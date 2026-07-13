import json
import torch
import torchaudio
import torchaudio.transforms as T
import onnxruntime as ort
from pathlib import Path
import os
import time
from collections import deque
import numpy as np
import inspect
import queue

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    sd = None
#klasy niezbedne do transformacji danych audio i spektrogramów

try:
    from fastai.vision.all import *
    from fastcore.foundation import L
    import itertools
    if not hasattr(L, 'starmap'):
        L.starmap = lambda self, f: L(itertools.starmap(f, self))
    from torch_audiomentations import Compose, Shift, Gain, PolarityInversion, AddColoredNoise, PitchShift, HighPassFilter, LowPassFilter
    HAS_FASTAI = True
except ImportError:
    HAS_FASTAI = False
    class Transform:
        pass
    class TensorImage:
        def __new__(cls, x, *args, **kwargs): 
            return x
    def show_image(*args, **kwargs): 
        pass


#zamiana pliku na falę dźwiękową
class LoadAudio(Transform):
    def __init__(self, duration=3.0):
        self.duration = duration

    def encodes(self, file: Path):
        waveform, sr = torchaudio.load(file)
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
        sum_x, sum_x2, n = 0.0, 0.0, 0
        for x in items:
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
        

#klasa gdy ktos chce trenowac swoj wlasny model, a nie korzystac z wbudowanych wytrenowanych przykladow 
class Engine:
    def __init__(self):
        self.model_name = None

    def train(self, dataset_path, epochs=30, batch_size=32, wd=0.01, eps=0.01, valid_pct=0.1, model_name='myownmodel', duration=3.0, sr=16000):
        if not HAS_FASTAI:
            raise RuntimeError("Training requires fastai. Install the full package: pip install keywordtensor")
        
        self.model_name = model_name
        
        get_audio_files = partial(get_files, extensions=['.wav'])
        files = get_audio_files(Path(dataset_path))
        splits = RandomSplitter(valid_pct=valid_pct, seed=42)(files)
        
        norm_spec = NormalizeSpec()
        
        tfms = [
                [LoadAudio(duration=duration), AudioAugment(sr=sr), WaveformToSpectrogram(sr=sr), norm_spec, SpecAugment()],
                [parent_label, Categorize()]
                ]
        dsets = Datasets(files, tfms, splits=splits)
        dls = dsets.dataloaders(bs=batch_size)

        model = xresnet18(c_in=1, n_out=len(dls.vocab), pretrained=False)
        learn = Learner(dls, model, wd=wd, metrics=accuracy, loss_func=LabelSmoothingCrossEntropy(eps=eps))
        
        res = learn.lr_find()
        base_lr = res.valley
        learn.fit_one_cycle(epochs, lr_max=slice(base_lr/10, base_lr))

        config = {
            "labels": list(dls.vocab),
            "mean": float(norm_spec.mean),
            "std": float(norm_spec.std),
            "duration": float(duration),
            "sr": int(sr)
        }
        with open(f"{model_name}_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        x, y = learn.dls.one_batch()
        dummy_input = x[0].unsqueeze(0).cpu()

        model = learn.model.cpu()
        model.eval()

        torch.onnx.export(
            model, 
            dummy_input, 
            f"{model_name}.onnx",
            input_names=['input'],
            output_names=['output'],
            opset_version=12,
            dynamo=False
            )


    def listen(self, model_name, actions=None, min_confidence=0.6, n_averages=3, device=None):
        
        if actions is None:
            actions = {}
            
        user_model_path = Path(f"{model_name}.onnx")
        user_config_path = Path(f"{model_name}_config.json")
        
        library_dir = os.path.dirname(os.path.abspath(__file__))
        builtin_base_path = os.path.join(library_dir, "pretrained", model_name)
        builtin_model_path = Path(f"{builtin_base_path}.onnx")
        builtin_config_path = Path(f"{builtin_base_path}_config.json")

        if user_model_path.exists() and user_config_path.exists():
            resolved_path = model_name
        elif builtin_model_path.exists() and builtin_config_path.exists():
            resolved_path = builtin_base_path
            
        with open(f"{resolved_path}_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        labels = cfg["labels"]
        sr = cfg["sr"]
        duration = cfg["duration"]
        cooldown = cfg.get("cooldown", 2.0)
        
        wav_to_spec = WaveformToSpectrogram(sr=sr)
        normalize_spec = NormalizeSpec(mean=cfg["mean"], std=cfg["std"])
        
        sess = ort.InferenceSession(f"{resolved_path}.onnx")
        inp_name = sess.get_inputs()[0].name
        
        buf_len = int(sr * duration)
        audio_buffer = deque([0.0] * buf_len, maxlen=buf_len)
        prediction_history = deque(maxlen=n_averages)
        last_trigger_times = {}

        def _run_inference(current_buffer, current_sr):
            wav_tensor = torch.tensor(current_buffer, dtype=torch.float32)
            
            if current_sr is not None and current_sr != sr:
                wav_tensor = torchaudio.functional.resample(wav_tensor, orig_freq=current_sr, new_freq=sr)
            
            if len(wav_tensor) > buf_len:
                wav_tensor = wav_tensor[-buf_len:]
            elif len(wav_tensor) < buf_len:
                return
                
            spectrogram = wav_to_spec.encodes(wav_tensor)
            spectrogram = normalize_spec.encodes(spectrogram)
            onnx_data = spectrogram.unsqueeze(0).unsqueeze(0).numpy()
            
            logits = sess.run(None, {inp_name: onnx_data})[0][0]
            exp_res = np.exp(logits - np.max(logits))
            probs = exp_res / exp_res.sum()
            prediction_history.append(probs)
            
            if len(prediction_history) == n_averages:
                mean_probs = np.mean(prediction_history, axis=0)
                idx = np.argmax(mean_probs)
                confidence = mean_probs[idx]
                predicted_class = labels[idx]

                if confidence > min_confidence:
                    if predicted_class in actions:
                        action_val = actions[predicted_class]
                        
                        if callable(action_val):
                            func = action_val
                            current_cooldown = 0.0
                        else:
                            func = action_val.get("function")
                            current_cooldown = action_val.get("cooldown", cooldown)
                            
                        last_time = last_trigger_times.get(predicted_class, 0.0)
                        
                        if func and (time.time() - last_time >= current_cooldown):
                            func()
                            prediction_history.clear()
                            last_trigger_times[predicted_class] = time.time()

        webrtc_ctx = None
        if "streamlit" in sys.modules:
            import streamlit as st
            if hasattr(st, "session_state") and "webrtc_ctx" in st.session_state:
                webrtc_ctx = st.session_state["webrtc_ctx"]

        current_sr = None
        if webrtc_ctx:
            def poll_audio():
                nonlocal current_sr
                try: frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.0)
                except queue.Empty: frames = []
                for frame in frames:
                    clean_frame = frame.reformat(format='flt', layout='mono', rate=sr)
                    if current_sr is None: current_sr = clean_frame.sample_rate
                    audio_buffer.extend(clean_frame.to_ndarray()[0].tolist())
                return current_sr
        else:
            if not HAS_SOUNDDEVICE:
                raise RuntimeError("Live listening requires sounddevice. Install it via pip: pip install sounddevice")
            def _audio_callback(indata, frames, time_info, status):
                audio_buffer.extend(indata[:, 0].tolist())
            stream = sd.InputStream(samplerate=sr, channels=1, callback=_audio_callback)
            stream.start()
            current_sr = sr
            def poll_audio():
                return current_sr

        try:
            while True:
                if webrtc_ctx and not webrtc_ctx.state.playing:
                    break
                    
                active_sr = poll_audio()
                
                if active_sr and len(audio_buffer) >= int(active_sr * duration):
                    _run_inference(list(audio_buffer), active_sr)
                    
                time.sleep(0.05)
        finally:
            if not webrtc_ctx:
                stream.stop()
                stream.close()