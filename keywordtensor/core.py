
#wszystkie importy
import json
import torch
import torchaudio
import torchaudio.transforms as T
import onnxruntime as ort
from pathlib import Path
import os
import time
import uuid
from collections import deque, defaultdict
import numpy as np
import inspect
import queue
import threading
import random
import sys
import contextlib

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    sd = None

try:
    from fastai.vision.all import *
    from fastcore.foundation import L
    import itertools
    if not hasattr(L, 'starmap'):
        L.starmap = lambda self, f: L(itertools.starmap(f, self))
    from torch_audiomentations import Compose, Shift, Gain, PolarityInversion, AddColoredNoise, PitchShift, HighPassFilter, LowPassFilter
    from huggingface_hub import snapshot_download
    IS_EDGE_VERSION = False
except ImportError:
    IS_EDGE_VERSION = True
    class Transform:
        pass
    class TensorImage:
        def __new__(cls, x, *args, **kwargs): 
            return x
    def show_image(*args, **kwargs): 
        pass






#zamiana pliku na falę dźwiękową
class LoadAudio(Transform):
    def __init__(self, sr=16000, duration=3.0):
        self.sr = sr
        self.duration = duration

    def encodes(self, file: Path):
        waveform, sr = torchaudio.load(file)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sr)
            sr = self.sr
        target_length = int(self.duration * sr)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            pad_amount = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

#definicja wyświetlenia spektrogramu
class AudioSpectrogram(TensorImage):
    def show(self, ctx=None, **kwargs):
        return show_image(self, ctx=ctx, cmap='magma', **kwargs)

#definicja zamiany fali dźwiękowej na spektrogram
class WaveformToSpectrogram(Transform):
    def __init__(self, sr=16000):
        self.melspec = T.MelSpectrogram(sample_rate=sr, n_mels=128, n_fft=1024, hop_length=128)
        self.todb    = T.AmplitudeToDB(top_db=80.0)
        
    def encodes(self, waveform):
        spec = self.todb(self.melspec(waveform))
        return AudioSpectrogram(spec)
    
#definicja augmentacji audio
class AudioAugment(Transform):
    split_idx = 0 
    
    def __init__(self, sr=16000):
        self.sr = sr
        
        self.audio_augcompose = Compose([
            Shift(min_shift=-0.1, max_shift=0.1, sample_rate=self.sr, p=0.4, rollover=False, output_type="dict"),
            Gain(min_gain_in_db=-15.0, max_gain_in_db=15.0, p=0.6, output_type="dict"),
            PolarityInversion(p=0.5, output_type="dict"),
            AddColoredNoise(min_snr_in_db=20.0, max_snr_in_db=35.0, p=0.6, output_type="dict"),
            PitchShift(min_transpose_semitones=-8, max_transpose_semitones=8, sample_rate=self.sr, p=0.6, output_type="dict"),
            HighPassFilter(min_cutoff_freq=50, max_cutoff_freq=300, sample_rate=self.sr, p=0.3, output_type="dict"),
            LowPassFilter(min_cutoff_freq=5000, max_cutoff_freq=7000, sample_rate=self.sr, p=0.3, output_type="dict"),
        ], output_type="dict")

    def encodes(self, waveform):
        audio_input = waveform.unsqueeze(0)
        augment_dict = self.audio_augcompose(audio_input, sample_rate=self.sr)
        audio_output = augment_dict.samples
        return audio_output.squeeze(0)
    
#defdinicja augmentacji spektrogramu
class SpecAugment(Transform):
    split_idx = 0
    def __init__(self):
        self.tmask = T.TimeMasking(time_mask_param=4)
        self.fmask = T.FrequencyMasking(freq_mask_param=4)
        
    def encodes(self, spec: AudioSpectrogram):
        return self.fmask(self.tmask(spec))

#definicja normalizacji spektrogramu
class NormalizeSpec(Transform):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def setups(self, items):
        if len(items) > 1000:
            items = [items[i] for i in random.sample(range(len(items)), 1000)]
        sum_x, sum_x2, n = 0.0, 0.0, 0
        for x in progress_bar(items):
            sum_x += x.sum().item()
            sum_x2 += (x**2).sum().item()
            n += x.numel()
        self.mean = sum_x / n
        self.std = (sum_x2 / n - self.mean**2)**0.5

    def encodes(self, spec: AudioSpectrogram):
        normalized = (spec - self.mean) / self.std
        return AudioSpectrogram(normalized)

    #def decodes(self, spec: AudioSpectrogram):
        #denormalized = (spec * self.std) + self.mean
        #return AudioSpectrogram(denormalized)






#przygotowanie plikow do treningu i tworzenie klasy other gdy nazwa zawiera "mixed:"
def prepare_files(files_paths, classes, label_func):
    paths = [files_paths] if isinstance(files_paths, (str, Path)) else files_paths
    by_cls = defaultdict(list)
    
    for f in (file for p in paths for file in get_files(Path(p), extensions=['.wav', '.opus'])):
        by_cls[label_func(f)].append(f)
        
    if not classes: 
        return L((f, l) for l, fs in by_cls.items() for f in fs), list(by_cls.keys())
    
    bases = [c.replace("mixed:", "") for c in classes]
    for b in bases: 
        assert by_cls[b], f"Error: 0 files for class '{b}'"
    
    normal_classes = [c for c in classes if "mixed:" not in c]
    tgt_n = len(by_cls[normal_classes[0]]) if normal_classes else 0
    
    others = [f for k, v in by_cls.items() if k not in bases for f in v]
    out = L()
    
    for c, b in zip(classes, bases):
        noise = by_cls[b]
        if c == b: 
            out.extend([(f, c) for f in noise])
        else:
            pool = (random.choices(noise, k=tgt_n//2) + random.choices(others, k=tgt_n - tgt_n//2)) if (noise and others) else random.choices(noise or others, k=tgt_n)
            out.extend([(f, c) for f in pool])
            
    return out, classes

#pobieranie danych z: google, mswc, linku, hf, lub folderu
def resolve_dataset(dataset):
    urls = []
    paths = []
    label_func = parent_label
    
    if dataset == "google":
        label_func = lambda f: 'other' if 'ESC-50' in str(f) or 'TigreGotico' in str(f) else parent_label(f)
        urls = ["http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"]
        paths.append(Path(snapshot_download(repo_id="TigreGotico/ESC-50", repo_type="dataset")))
    elif dataset == "mswc":
        label_func = lambda f: 'other' if 'ESC-50' in str(f) or 'TigreGotico' in str(f) else f.name.split('_')[0]
        parts = list("012345")
        urls = [f"https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/main/data/opus/pl/train/audio/{p}.tar.gz" for p in parts]
        paths.append(Path(snapshot_download(repo_id="TigreGotico/ESC-50", repo_type="dataset")))
    elif dataset.startswith("http"):
        urls = [dataset]
        
    if urls:
        for u in urls: 
            paths.append(untar_data(u))
    elif dataset.startswith("hf:"):
        paths = [Path(snapshot_download(repo_id=dataset[3:], repo_type="dataset"))]
    else:
        paths = [Path(dataset)]

    return paths, label_func





    
#klasa gdy ktos chce trenowac swoj wlasny model, a nie korzystac z wbudowanych wytrenowanych przykladow 
class Engine:
    def __init__(self):
        self.model_name = None

    def train(self, dataset, classes: list = None, epochs=30, batch_size=32, wd=0.01, eps=0.01, valid_pct=0.1, model_path='myownmodel', duration=3.0, sr=16000):
        if IS_EDGE_VERSION:
            raise RuntimeError("This is the Edge version. To train, install: pip install keywordtensor[train]")

        dataset_path, label_func = resolve_dataset(dataset)
        items, classes = prepare_files(dataset_path, classes, label_func)
        self.model_name = model_path
        
        splits = RandomSplitter(valid_pct=valid_pct, seed=42)(items)
        norm_spec = NormalizeSpec()
        
        tfms = [
                [ItemGetter(0), LoadAudio(sr=sr, duration=duration), AudioAugment(sr=sr), WaveformToSpectrogram(sr=sr), norm_spec, SpecAugment()],
                [ItemGetter(1), Categorize()]
                ]
        dsets = Datasets(items, tfms, splits=splits)
        dls = dsets.dataloaders(bs=batch_size)

        model = xresnet18(c_in=1, n_out=len(dls.vocab), pretrained=False)
        learn = Learner(dls, model, wd=wd, metrics=accuracy, loss_func=LabelSmoothingCrossEntropy(eps=eps))
        
        res = learn.lr_find(show_plot=False)
        base_lr = res.valley
        learn.fit_one_cycle(epochs, lr_max=slice(base_lr/10, base_lr))

        config = {
            "labels": list(dls.vocab),
            "mean": float(norm_spec.mean),
            "std": float(norm_spec.std),
            "duration": float(duration),
            "sr": int(sr)
        }
        with open(f"{model_path}_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        x, y = learn.dls.one_batch()
        dummy_input = x[0].unsqueeze(0).cpu()

        model = learn.model.cpu()
        model.eval()

        torch.onnx.export(
            model, 
            dummy_input, 
            f"{model_path}.onnx",
            input_names=['input'],
            output_names=['output'],
            opset_version=12,
            dynamo=False
            )






    def listen(self, model_path, actions=None, min_confidence=0.6, n_averages=3, source="microphone", listen_time=0, threads: int = None, stop=None)):

        #wczytanie pliku config oraz modelu
        user_model_path = Path(f"{model_path}.onnx")
        user_config_path = Path(f"{model_path}_config.json")
        library_dir = os.path.dirname(os.path.abspath(__file__))
        builtin_base_path = os.path.join(library_dir, "pretrained", model_path)
        builtin_model_path = Path(f"{builtin_base_path}.onnx")
        builtin_config_path = Path(f"{builtin_base_path}_config.json")
        if user_model_path.exists() and user_config_path.exists():
            resolved_path = model_path
        elif builtin_model_path.exists() and builtin_config_path.exists():
            resolved_path = builtin_base_path
        with open(f"{resolved_path}_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        labels = cfg["labels"]
        sr = cfg["sr"]
        duration = cfg["duration"]
        mean = cfg["mean"]
        std = cfg["std"]

        #ilosć watkow do wykorzystania przez model
        if threads is not None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = threads
            sess_options.inter_op_num_threads = threads
            sess = ort.InferenceSession(f"{resolved_path}.onnx", sess_options=sess_options)
        else:
            sess = ort.InferenceSession(f"{resolved_path}.onnx")
        inp_name = sess.get_inputs()[0].name

        #wybor numeru mikrofonu
        device_id = None
        if isinstance(source, str) and source.startswith("microphone:"):
            try:
                device_id = int(source.split(":")[1])
            except ValueError:
                pass

        #tworzenie buforu audio z mikrofonu lub zmiennej uzytkownika
        length_in_samples = int(sr * duration)
        bufor_audio = deque([0.0] * length_in_samples, maxlen=length_in_samples)
        stream = None
        if isinstance(source, str) and source.startswith("microphone"):
            def audio_callback(indata, frames, time_info, status):
                bufor_audio.extend(indata[:, 0].tolist())
            stream = sd.InputStream(samplerate=sr, channels=1, dtype='float32', device=device_id, callback=audio_callback)
            stream.start()
        else:
            bufor_audio = source

        #predykcja z podanego bufora audio z uwzglednieniem resamplingu
        wav_to_spec = WaveformToSpectrogram(sr=sr)
        normalize_spec = NormalizeSpec(mean=mean, std=std)
        def predict(bufor_audio):
            if isinstance(bufor_audio, tuple):
                original_sr, audio_data = bufor_audio
                wav_tensor = torch.tensor(audio_data, dtype=torch.float32)
                if original_sr != sr:
                    wav_tensor = torchaudio.functional.resample(wav_tensor.unsqueeze(0), orig_freq=original_sr, new_freq=sr).squeeze(0)
            else:
                wav_tensor = torch.tensor(list(bufor_audio), dtype=torch.float32)
            spectrogram = wav_to_spec.encodes(wav_tensor)
            spectrogram = normalize_spec.encodes(spectrogram)
            onnx_data = spectrogram.unsqueeze(0).unsqueeze(0).numpy()
            logits = sess.run(None, {inp_name: onnx_data})[0][0]
            exp_res = np.exp(logits - np.max(logits))
            probs = exp_res / exp_res.sum()
            return {label: float(prob) for label, prob in zip(labels, probs)}

        #obsluga predykcji i wywolanie akcji
        prediction_history = deque(maxlen=n_averages)
        def process_actions(probs):
            if not probs:
                return
            prediction_history.append(probs)
            if len(prediction_history) < n_averages:
                return
            mean_probs = {}
            for label in labels:
                total = sum(hist[label] for hist in prediction_history)
                mean_probs[label] = total / n_averages
            best_word = max(mean_probs, key=mean_probs.get)
            confidence = mean_probs[best_word]
            if confidence > min_confidence:
                if not actions:
                    print(f"Recognized: {best_word} ({confidence:.2f})")
                    prediction_history.clear()
                    time.sleep(duration)
                elif best_word in actions:
                    actions[best_word]()
                    prediction_history.clear()

        #petla do obslugi nasluchu
        start_time = time.time()
        time.sleep(duration)
        try:
            if listen_time > 0:
                while (time.time() - start_time) <= listen_time:
                    if stop is not None and stop(): return
                    probs = predict(bufor_audio)
                    process_actions(probs)
                    time.sleep(0.05)
            elif listen_time == 0:
                probs = predict(bufor_audio)
                print(probs)
            elif listen_time == -1:
                while True:
                    if stop is not None and stop(): return
                    probs = predict(bufor_audio)
                    process_actions(probs)
                    time.sleep(0.05)
        finally:
            if stream is not None:
                stream.stop()
                stream.close()






    def record(self, target, classes: list, samples: int = 100, actions: dict = None, source="microphone", duration:float = 3.0, sr=16000, stop=None):
        length_in_samples = int(sr * duration)

        device_id = None
        if isinstance(source, str) and source.startswith("microphone:"):
            try:
                device_id = int(source.split(":")[1])
            except ValueError:
                pass

        if actions is None:
            actions = {}

        for cls in classes:
            for i in range(samples):

                if stop is not None and stop(): return

                bufor_audio = None 
                start_time = None  
                def record_in_background():
                    nonlocal bufor_audio
                    if isinstance(source, str) and source.startswith("microphone"):
                        bufor_audio = sd.rec(length_in_samples, samplerate=sr, channels=1, dtype='float32',device=device_id)
                        sd.wait() 
                    else:
                        time.sleep(duration)
                        bufor_audio = source

                t = threading.Thread(target=record_in_background)

                def start_recording():
                    nonlocal start_time
                    start_time = time.time()
                    t.start()
                    

                def current_time():
                    if start_time is None: return 0.0
                    return time.time() - start_time

                if cls in actions:
                    actions[cls](start_recording=start_recording, current_time=current_time, total_time=duration)
                else:
                    print(f"Recording sample {i} for [{cls}]...")
                    t.start()             
                t.join()

                if isinstance(bufor_audio, tuple):
                    original_sr, audio_data = bufor_audio
                    tensor_data = torch.tensor(audio_data, dtype=torch.float32).T
                    if original_sr != sr:
                        tensor_data = torchaudio.functional.resample(tensor_data, orig_freq=original_sr, new_freq=sr)
                else:
                    tensor_data = torch.tensor(bufor_audio, dtype=torch.float32).T

                if isinstance(target, str):
                    folder_path = os.path.join(target, cls)
                    os.makedirs(folder_path, exist_ok=True)
                    save_path = os.path.join(target, cls, f"{i}_{uuid.uuid4().hex[:6]}.wav")
                    torchaudio.save(save_path, tensor_data, sr)
                else:
                    target(cls, i, tensor_data, sr)
                



