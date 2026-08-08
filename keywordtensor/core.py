
import json
import urllib.request
from urllib.error import URLError, HTTPError
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
import librosa
import soundfile as sf

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    sd = None

try:
    import torch
    import torch.nn.functional
    import torch.nn as nn
    import torchaudio
    import torchaudio.transforms as T
    from fastai.vision.all import *
    from fastcore.foundation import L
    import itertools
    if not hasattr(L, 'starmap'):
        L.starmap = lambda self, f: L(itertools.starmap(f, self))
    from torch_audiomentations import *
    from huggingface_hub import snapshot_download
    from fastai.data.external import fastai_path
    from datasets import load_dataset
    from fastai.metrics import accuracy, Precision, Recall, F1Score
    from fastai.callback.data import WeightedDL
    from collections import Counter
    import math
    IS_EDGE_VERSION = False
    
    if not hasattr(torchaudio, 'info'):
        class _FakeAudioMetaData:
            def __init__(self, sample_rate, num_frames):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
        def _fast_info(filepath):
            info = sf.info(str(filepath))
            return _FakeAudioMetaData(info.samplerate, info.frames)
        torchaudio.info = _fast_info

except ImportError:
    IS_EDGE_VERSION = True
    class Transform:
        pass
    class TensorImage:
        def __new__(cls, x, *args, **kwargs): 
            return x
    def show_image(*args, **kwargs): 
        pass






class LoadAudio(Transform):
    def __init__(self, sr=16000):
        self.sr = sr

    def encodes(self, file: Path):
        waveform, sr = librosa.load(file, sr=None)
        if sr != self.sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sr)
        return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)


class CropAudioTrain(Transform):
    split_idx = 0
    def __init__(self, duration=1.0, sr=16000):
        self.target_len = int(duration * sr)

    def encodes(self, waveform):
        current_len = waveform.shape[1]
        if current_len > self.target_len:
            start_idx = random.randint(0, current_len - self.target_len)
            waveform = waveform[:, start_idx : start_idx + self.target_len]
        elif current_len < self.target_len:
            pad_total = self.target_len - current_len
            pad_left = random.randint(0, pad_total)
            pad_right = pad_total - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
        return waveform


class CropAudioValid(Transform):
    split_idx = 1
    def __init__(self, duration=1.0, sr=16000):
        self.target_len = int(duration * sr)

    def encodes(self, waveform):
        current_len = waveform.shape[1]
        if current_len > self.target_len:
            waveform = waveform[:, :self.target_len]
        elif current_len < self.target_len:
            pad_amount = self.target_len - current_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform


class AudioSpectrogram(TensorImage):
    def show(self, ctx=None, **kwargs):
        return show_image(self, ctx=ctx, cmap='magma', **kwargs)


class WaveformToSpectrogram(Transform):
    def __init__(self, sr=16000):
        self.sr = sr

    def encodes(self, waveform):
        is_tensor = hasattr(waveform, 'numpy') 
        audio_np = waveform.squeeze().numpy() if is_tensor else waveform
        mel = librosa.feature.melspectrogram(y=audio_np, sr=self.sr, n_fft=1024, hop_length=128, win_length=1024, n_mels=128, power=2.0, htk=True, center=True, pad_mode='reflect', norm=None)
        spec_db = librosa.power_to_db(mel, ref=1.0, top_db=80.0, amin=1e-10)
        if is_tensor:
            return AudioSpectrogram(torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0))
        return np.expand_dims(spec_db, axis=(0, 1)).astype(np.float32)
        

class AudioAugment(Transform):
    split_idx = 0 
    def __init__(self, sr=16000, bg_paths=None, rir_paths=None):
        self.sr = sr
        
        tfms = [
            Shift(min_shift=-0.1, max_shift=0.1, sample_rate=self.sr, p=0.4, rollover=False, output_type="dict"),
            PitchShift(min_transpose_semitones=-4, max_transpose_semitones=4, sample_rate=self.sr, p=0.5, output_type="dict"),
        ]
        
        if bg_paths:
            tfms.append(AddBackgroundNoise(background_paths=bg_paths, min_snr_in_db=10.0, max_snr_in_db=20.0, p=0.6, output_type="dict"))
        if rir_paths:
            tfms.append(ApplyImpulseResponse(ir_paths=rir_paths, p=0.3, compensate_for_propagation_delay=True, output_type="dict"))
        
        tfms.extend([
            LowPassFilter(min_cutoff_freq=5000, max_cutoff_freq=7000, sample_rate=self.sr, p=0.2, output_type="dict"),
            HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=300, sample_rate=self.sr, p=0.2, output_type="dict"),
            BandStopFilter(min_center_frequency=600, max_center_frequency=3500, min_bandwidth_fraction=0.02, max_bandwidth_fraction=0.1, sample_rate=self.sr, p=0.1, output_type="dict"),
            AddColoredNoise(min_snr_in_db=15.0, max_snr_in_db=30.0, p=0.2, output_type="dict"),
            PeakNormalization(apply_to="all", p=1.0, output_type="dict"),
            Gain(min_gain_in_db=-25.0, max_gain_in_db=0.0, p=0.6, output_type="dict")
        ])
        
        self.audio_augcompose = Compose(tfms, output_type="dict")

    def encodes(self, waveform):
        audio_input = waveform.unsqueeze(0)
        augment_dict = self.audio_augcompose(audio_input, sample_rate=self.sr)
        audio_output = augment_dict.samples
        return audio_output.squeeze(0)
    

class SpecAugment(Transform):
    split_idx = 0
    def __init__(self):
        self.tmask1 = T.TimeMasking(time_mask_param=2)
        self.tmask2 = T.TimeMasking(time_mask_param=2)
        self.fmask1 = T.FrequencyMasking(freq_mask_param=2)
        self.fmask2 = T.FrequencyMasking(freq_mask_param=2)
        
    def encodes(self, spec: AudioSpectrogram):
        return self.fmask2(self.fmask1(self.tmask2(self.tmask1(spec))))


class NormalizeSpec(Transform):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def setups(self, items):
        n_items = len(items)
        indices = random.sample(range(n_items), 1000) if n_items > 1000 else range(n_items)
        sum_x, sum_x2, n = 0.0, 0.0, 0
        for i in progress_bar(indices):
            x = items[i]
            sum_x += x.sum().item()
            sum_x2 += (x**2).sum().item()
            n += x.numel()
        self.mean = sum_x / n
        self.std = (sum_x2 / n - self.mean**2)**0.5

    def encodes(self, spec):
        normalized = (spec - self.mean) / self.std
        is_tensor = hasattr(spec, 'numpy')
        return AudioSpectrogram(normalized) if is_tensor else normalized

    #def decodes(self, spec: AudioSpectrogram):
        #denormalized = (spec * self.std) + self.mean
        #return AudioSpectrogram(denormalized)






def prepare_files(dataset_paths, label_func, classes, bg_classes=None, rir_classes=None, mixed=True):
    bg_cls = [bg_classes] if isinstance(bg_classes, str) else (bg_classes or [])
    rir_cls = [rir_classes] if isinstance(rir_classes, str) else (rir_classes or [])

    all_files = [f for p in dataset_paths for f in get_files(Path(p), extensions=['.wav', '.opus'])]

    by_label = defaultdict(list)
    for f in all_files:
        by_label[label_func(f)].append(f)

    bg_files = []
    for c in bg_cls: bg_files.extend(by_label.get(c, []))
        
    rir_files = []
    for c in rir_cls: rir_files.extend(by_label.get(c, []))

    bg_paths = list({str(f.parent.resolve()) for f in bg_files})
    rir_paths = list({str(f.parent.resolve()) for f in rir_files})

    if bg_cls: assert bg_paths, f"Error: No background files found for classes {bg_cls}"
    if rir_cls and not rir_paths: print(f"Warning: No RIR files found for classes {rir_cls}")

    classes_out = []
    for c in classes:
        for f in by_label.get(c, []):
            classes_out.append((f, c))
            
    for c in classes: assert any(cls == c for _, cls in classes_out), f"Error: 0 files found for target class '{c}'"

    ignored_labels = set(classes + bg_cls + rir_cls)
    other_out = []
    for lbl, files in by_label.items():
        if lbl not in ignored_labels:
            for f in files:
                other_out.append((f, lbl))

    out = L(classes_out)

    if bg_cls:
        tgt_n = sum(len(by_label.get(c, [])) for c in classes) // len(classes) if classes else 0
        
        if mixed:
            others = [f for f, _ in other_out]
            pool = (random.choices(others, k=tgt_n//2) + random.choices(bg_files, k=tgt_n - tgt_n//2)) if others else random.choices(bg_files, k=tgt_n)
        else:
            pool = random.choices(bg_files, k=tgt_n) if bg_files else []

        out.extend([(f, "other") for f in pool])
        
    print("Preparation summary:")
    unique_classes = list(set([c for _, c in out]))
    for c in sorted(unique_classes):
        class_count = len([f for f, cls in out if cls == c])
        print(f"Class '{c}': {class_count} files")
        
    print(f"Background files available for aug: {len(bg_files)}")
    print(f"RIR files available for aug: {len(rir_files)}")
    
    return out, bg_paths, rir_paths


def resolve_dataset(dataset):
    paths = []
    archive_base = Path("~/.fastai/archive").expanduser()
    data_base = Path("~/.fastai/data").expanduser()
    
    datasets = [dataset] if isinstance(dataset, (str, Path)) else dataset
    
    for d in datasets:
        d_str = str(d)
        if d_str == "google":
            dest = data_base / "speech_commands"
            fname = archive_base / "speech_commands"
            print("Downloading Google Speech Commands dataset...")
            try: paths.append(untar_data("http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", data=dest, archive=fname))
            except Exception: 
                print("Corrupted download detected for google. Forcing redownload...")
                paths.append(untar_data("http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", data=dest, archive=fname, force_download=True))
            print("Google Speech Commands dataset completed.")
        elif d_str == "caiman":
            caiman_cache = data_base / "CAIMAN-ASR"
            print("Downloading CAIMAN-ASR dataset...")
            if not caiman_cache.exists() or len(list(caiman_cache.glob("*.wav"))) < 1000:
                try: ds = load_dataset("Myrtle/CAIMAN-ASR-BackgroundNoise", split="train")
                except Exception: 
                    print("Corrupted download detected for caiman. Forcing redownload...")
                    ds = load_dataset("Myrtle/CAIMAN-ASR-BackgroundNoise", split="train", download_mode="force_redownload")
                caiman_cache.mkdir(parents=True, exist_ok=True)
                for i, item in progress_bar(enumerate(ds), total=len(ds)):
                    audio_data = item["audio"]
                    save_path = caiman_cache / f"noise_{i}.wav"
                    sf.write(str(save_path), audio_data["array"], audio_data["sampling_rate"])
            print("CAIMAN-ASR dataset completed.")
            paths.append(caiman_cache)
        elif d_str == "openrir":
            dest = data_base / "openslr-rir"
            fname = archive_base / "rir_noises"
            print("Downloading OpenSLR RIR dataset...")
            try: paths.append(untar_data("http://www.openslr.org/resources/28/rirs_noises.zip", data=dest, archive=fname))
            except Exception: 
                print("Corrupted download detected for openrir. Forcing redownload...")
                paths.append(untar_data("http://www.openslr.org/resources/28/rirs_noises.zip", data=dest, archive=fname, force_download=True))
            print("OpenSLR RIR dataset completed.")
        elif d_str == "mswc":
            parts = [str(i) for i in range(88)]
            print("Downloading MSWC English (88 parts)...")
            for p in parts:
                dest = data_base / f"mswc_en_{p}"
                fname = archive_base / f"mswc_en_{p}"
                url = f"https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/main/data/opus/en/train/audio/{p}.tar.gz"
                try: paths.append(untar_data(url, archive=fname, data=dest))
                except Exception: 
                    print(f"Corrupted download detected for mswc part {p}. Forcing redownload...")
                    paths.append(untar_data(url, archive=fname, data=dest, force_download=True))
            print("MSWC English dataset completed.")
        elif d_str == "mswc-pl":
            parts = list("012345")
            print("Downloading MSWC Polish (6 parts)...")
            for p in parts:
                dest = data_base / f"mswc_pl_{p}"
                fname = archive_base / f"mswc_pl_{p}"
                url = f"https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/main/data/opus/pl/train/audio/{p}.tar.gz"
                try: paths.append(untar_data(url, archive=fname, data=dest))
                except Exception: 
                    print(f"Corrupted download detected for mswc-pl part {p}. Forcing redownload...")
                    paths.append(untar_data(url, archive=fname, data=dest, force_download=True))
            print("MSWC Polish dataset completed.")
        elif d_str.startswith("http"):
            filename = d_str.split('/')[-1]
            dest = data_base / filename.split('.')[0]
            fname = archive_base / filename.split('.')[0]
            print(f"Downloading custom dataset: {filename}...")
            try: paths.append(untar_data(d_str, data=dest, archive=fname))
            except Exception: 
                print(f"Corrupted download detected for {filename}. Forcing redownload...")
                paths.append(untar_data(d_str, data=dest, archive=fname, force_download=True))
            print(f"Custom dataset {filename} completed.")
        elif d_str.startswith("hf:"):
            repo = d_str[3:]
            print(f"Downloading HuggingFace dataset: {repo}...")
            try: paths.append(Path(snapshot_download(repo_id=repo, repo_type="dataset")))
            except Exception: 
                print(f"Corrupted download detected for HF repo {repo}. Forcing redownload...")
                paths.append(Path(snapshot_download(repo_id=repo, repo_type="dataset", force_download=True)))
            print(f"HuggingFace dataset {repo} completed.")
        else:
            paths.append(Path(d))

    def label_func(f):
        if 'CAIMAN-ASR' in str(f) or 'Myrtle' in str(f): return 'background'
        if '_background_noise_' in f.parent.name: return 'background'
        if 'openslr' in str(f).lower() or 'rirs_noises' in str(f).lower(): return 'rir'
        if any("mswc" in str(d) for d in datasets) and f.suffix == '.opus' and '_' in f.name: return f.name.split('_')[0]
        return parent_label(f)

    return paths, label_func






class Engine:
    def __init__(self):
        self.model_name = None

    def train(self, dataset, classes: list = None, epochs=10, batch_size=32, wd=0.01, eps=0.05, valid_pct=0.1, model_path='myownmodel', duration=1.0, sr=16000, alpha=0.0, bg_classes=None, rir_classes=None, mixed=True, cbs=None):
        if IS_EDGE_VERSION:
            raise RuntimeError("This is the Edge version. To train, install: pip install keywordtensor[train]")

        if classes is not None and isinstance(classes, str):
            classes = [classes]

        assert classes is not None, "Error: 'classes' parameter is required."
        assert bg_classes is not None, "Error: 'bg_classes' parameter is required."

        model_dir = Path(model_path).parent
        if str(model_dir) != "." and not model_dir.exists():
            print(f"Error: Directory '{model_dir}' does not exist.")
            assert False, f"Please create the directory '{model_dir}' before training."
            
        if Path(f"{model_path}.onnx").exists() or Path(f"{model_path}_config.json").exists():
            print(f"Error: Model '{model_path}' already exists.")
            assert False, "Please choose a different model_path or delete existing files to prevent overwriting."

        dataset_paths, label_func = resolve_dataset(dataset)
        items, bg_paths, rir_paths = prepare_files(
            dataset_paths, label_func, classes,
            bg_classes=bg_classes, rir_classes=rir_classes, mixed=mixed
        )
        self.model_name = model_path
        
        splits = RandomSplitter(valid_pct=valid_pct, seed=42)(items)
        norm_spec = NormalizeSpec()
        
        tfms = [
                [ItemGetter(0), LoadAudio(sr=sr), CropAudioTrain(duration=duration, sr=sr), CropAudioValid(duration=duration, sr=sr), AudioAugment(sr=sr, bg_paths=bg_paths, rir_paths=rir_paths or None), WaveformToSpectrogram(sr=sr), norm_spec, SpecAugment()],
                [ItemGetter(1), Categorize()]
                ]
        dsets = Datasets(items, tfms, splits=splits)

        dls = dsets.dataloaders(bs=batch_size)

        train_labels = [items[i][1] for i in splits[0]]
        class_counts = Counter(train_labels)
        raw_weights = [1.0 / class_counts[c] for c in dls.vocab]
        weight_tensor = torch.tensor(raw_weights, dtype=torch.float32, device=dls.device)
        weight_tensor /= weight_tensor.mean()

        model = xresnet18(c_in=1, n_out=len(dls.vocab), pretrained=False)
        if torch.cuda.device_count() > 1:
            print(f"Detected {torch.cuda.device_count()} GPUs. Enabling DataParallel.")
            model = nn.DataParallel(model)

        base_cbs = [MixUp(alpha=alpha)] if alpha > 0 else []
        user_cbs = cbs if cbs is not None else []
        all_cbs = base_cbs + user_cbs

        metrics = [accuracy, Precision(average='macro'), Recall(average='macro'), F1Score(average='macro')]
        loss_func = LabelSmoothingCrossEntropy(weight=weight_tensor, eps=eps)
        learn = Learner(dls, model, wd=wd, metrics=metrics, loss_func=loss_func, cbs=all_cbs)
        
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

        model = learn.model
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        model = model.cpu()
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






    def listen(self, model_path, actions=None, min_confidence=0.6, n_averages=2, source="microphone", listen_time=-1, threads: int = None, stop=None):

        user_model_path = Path(f"{model_path}.onnx")
        user_config_path = Path(f"{model_path}_config.json")
        library_dir = os.path.dirname(os.path.abspath(__file__))
        builtin_base_path = os.path.join(library_dir, "pretrained", model_path)
        builtin_model_path = Path(f"{builtin_base_path}.onnx")
        builtin_config_path = Path(f"{builtin_base_path}_config.json")
        if user_model_path.exists() and user_config_path.exists():
            resolved_path = model_path
            
        elif not builtin_model_path.exists() or not builtin_config_path.exists():
            try:
                print(f"Downloading pre-trained '{model_path}' model (this may take a few seconds)...")
                base_url = f"https://raw.githubusercontent.com/fkondela/KeywordTensor/main/keywordtensor/pretrained/{model_path}"
                
                urllib.request.urlretrieve(f"{base_url}.onnx", str(builtin_model_path))
                urllib.request.urlretrieve(f"{base_url}_config.json", str(builtin_config_path))
                print("Download complete!")
            except (URLError, HTTPError):
                print(f"Error: Model '{model_path}' not found locally or on the server.")
                sys.exit(1)
                
            resolved_path = builtin_base_path
            
        else:
            resolved_path = builtin_base_path
        with open(f"{resolved_path}_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        labels = cfg["labels"]
        sr = cfg["sr"]
        duration = cfg["duration"]
        mean = cfg["mean"]
        std = cfg["std"]

        if threads is not None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = threads
            sess_options.inter_op_num_threads = threads
            sess = ort.InferenceSession(f"{resolved_path}.onnx", sess_options=sess_options)
        else:
            sess = ort.InferenceSession(f"{resolved_path}.onnx")
        inp_name = sess.get_inputs()[0].name

        device_id = None
        if isinstance(source, str) and source.startswith("microphone:"):
            try:
                device_id = int(source.split(":")[1])
            except ValueError:
                pass

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

        wav_to_spec = WaveformToSpectrogram(sr=sr)
        normalize_spec = NormalizeSpec(mean=mean, std=std)
        def predict(bufor_audio):
            if isinstance(bufor_audio, tuple):
                original_sr, audio_data = bufor_audio
                audio_np = np.array(audio_data, dtype=np.float32)
                if original_sr != sr:
                    audio_np = librosa.resample(audio_np, orig_sr=original_sr, target_sr=sr)
            else:
                audio_np = np.array(list(bufor_audio), dtype=np.float32)

            spec = wav_to_spec.encodes(audio_np)
            onnx_data = normalize_spec.encodes(spec)

            logits = sess.run(None, {inp_name: onnx_data})[0][0]
            exp_res = np.exp(logits - np.max(logits))
            probs = exp_res / exp_res.sum()
            return {label: float(prob) for label, prob in zip(labels, probs)}

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






    def record(self, target, classes: list, samples: int = 100, actions: dict = None, source="microphone", duration:float = 1.0, sr=16000, stop=None):
        if classes is not None and isinstance(classes, str):
            classes = [classes]
            
        length_in_samples = int(sr * duration)

        device_id = None
        if isinstance(source, str) and source.startswith("microphone:"):
            try:
                device_id = int(source.split(":")[1])
            except ValueError:
                pass

        if actions is None:
            actions = {}

        for i in range(samples):
            for cls in classes:

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
                    for countdown in [3, 2, 1]:
                        print(f"\rRecording sample {i} for [{cls.upper()}] in {countdown}...", end="", flush=True)
                        time.sleep(1)
                        
                    start_recording()
                    
                    while current_time() < duration:
                        print(f"\rRecording sample {i} for [{cls.upper()}] ({current_time():.1f}s / {duration:.1f}s)", end="", flush=True)
                        time.sleep(0.1)
                        
                    print(f"\rRecording sample {i} for [{cls.upper()}] ({duration:.1f}s / {duration:.1f}s) - DONE!          ")             
                t.join()

                if isinstance(bufor_audio, tuple):
                    original_sr, audio_data = bufor_audio
                    audio_np = np.array(audio_data, dtype=np.float32).squeeze()
                    if original_sr != sr:
                        audio_np = librosa.resample(audio_np, orig_sr=original_sr, target_sr=sr)
                else:
                    audio_np = np.array(bufor_audio, dtype=np.float32).squeeze()

                if isinstance(target, str):
                    folder_path = os.path.join(target, cls)
                    os.makedirs(folder_path, exist_ok=True)
                    save_path = os.path.join(target, cls, f"{i}_{uuid.uuid4().hex[:6]}.wav")
                    sf.write(save_path, audio_np, sr)

                else:
                    target(cls, i, audio_np, sr)
                



