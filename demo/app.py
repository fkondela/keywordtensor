import sys, os, time, json, queue
from collections import deque
import torch
import torchaudio
import numpy as np
import onnxruntime as ort
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import keywordtensor
from keywordtensor.core import WaveformToSpectrogram, NormalizeSpec

st.set_page_config(page_title="KeywordTensor Live", layout="wide")
st.title("🎙️ KeywordTensor: Live WebRTC")

lib_path = os.path.dirname(keywordtensor.__file__)
config_path = os.path.join(lib_path, "pretrained", "prawda_falsz_config.json")
onnx_path = os.path.join(lib_path, "pretrained", "prawda_falsz.onnx")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

sr = cfg["sr"]
buf_len = int(sr * cfg["duration"])

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_ctx = webrtc_streamer(
    key="keyword-spotting",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "video": False, 
        "audio": {
            "autoGainControl": False,
            "echoCancellation": False,
            "noiseSuppression": False
        }
    },
    audio_receiver_size=256,
)

if webrtc_ctx.state.playing:
    wav_to_spec = WaveformToSpectrogram(sr=sr)
    normalize_spec = NormalizeSpec(mean=cfg["mean"], std=cfg["std"])
    sess = ort.InferenceSession(onnx_path)
    inp_name = sess.get_inputs()[0].name
    labels = cfg["labels"]
    
    if "audio_buffer" not in st.session_state:
        st.session_state.audio_buffer = deque([0.0] * buf_len, maxlen=buf_len)
    
    prediction_history = deque(maxlen=3)
    last_trigger_times = {label: 0.0 for label in labels}
    
    status_box = st.empty()
    log_box = st.empty()
    wykrycia = []

    while webrtc_ctx.state.playing:
        current_time = time.time()
        
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.01)
            except queue.Empty:
                audio_frames = []
            
            for frame in audio_frames:
                sound = frame.to_ndarray()
                
                if sound.ndim == 2:
                    sound = sound[0, :] if sound.shape[0] < sound.shape[1] else sound[:, 0]
                
                sound_float = sound.astype(np.float32) / 32768.0
                wav_chunk = torch.tensor(sound_float, dtype=torch.float32)
                
                if frame.sample_rate != sr:
                    wav_chunk = torchaudio.functional.resample(wav_chunk, orig_freq=frame.sample_rate, new_freq=sr)
                    
                st.session_state.audio_buffer.extend(wav_chunk.tolist())
                
        wav_tensor = torch.tensor(list(st.session_state.audio_buffer), dtype=torch.float32)
        spectrogram = wav_to_spec.encodes(wav_tensor)
        spectrogram = normalize_spec.encodes(spectrogram)
        onnx_data = spectrogram.unsqueeze(0).unsqueeze(0).numpy()
        
        logits = sess.run(None, {inp_name: onnx_data})[0][0]
        exp_res = np.exp(logits - np.max(logits))
        probs = exp_res / exp_res.sum()
        
        prediction_history.append(probs)
        
        if len(prediction_history) == 3:
            avg_probs = np.mean(prediction_history, axis=0)
            pred_idx = np.argmax(avg_probs)
            pred_label = labels[pred_idx]
            confidence = avg_probs[pred_idx]
            
            status_box.info(f"Ostatnia klasa: {pred_label} ({confidence:.0%})")
            
            if confidence > 0.6 and pred_label in ["prawda", "falsz"]:
                if current_time - last_trigger_times.get(pred_label, 0) > 2.0:
                    last_trigger_times[pred_label] = current_time
                    wykrycia.insert(0, f"✅ **{pred_label.upper()}** ({confidence:.0%})")
                    log_box.markdown("\n".join(wykrycia))
        
        time.sleep(0.05)
