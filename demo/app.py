import sys, os, time, json
from collections import deque
import torch
import numpy as np
import onnxruntime as ort
import torchaudio
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Dodanie katalogu glownego do sys.path by wczytac lokalny keywordtensor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import keywordtensor
from keywordtensor.core import WaveformToSpectrogram, NormalizeSpec

st.set_page_config(page_title="KeywordTensor Live", layout="wide")
st.title("🎙️ KeywordTensor: Live WebRTC")

lib_path = os.path.dirname(keywordtensor.__file__)
model_name = "prawda_falsz"
config_path = os.path.join(lib_path, "pretrained", f"{model_name}_config.json")
onnx_path = os.path.join(lib_path, "pretrained", f"{model_name}.onnx")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

sr = cfg["sr"]
duration = cfg["duration"]
buf_len = int(sr * duration)

if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = deque([0.0] * buf_len, maxlen=buf_len)

# Watek odbierajacy dzwiek z przegladarki 
def audio_frame_callback(frame):
    sound = frame.to_ndarray()
    if sound.ndim > 1 and sound.shape[1] > 1:
        mono = sound.mean(axis=1)
    else:
        mono = sound.flatten()
        
    st.session_state.audio_buffer.extend(mono.tolist())
    return frame

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
st.write("Wlacz mikrofon i uruchom analize ponizej.")

webrtc_ctx = webrtc_streamer(
    key="keyword-spotting",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_callback,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": False, "audio": True},
)

start_analysis = st.button("Uruchom analize")

if webrtc_ctx.state.playing and start_analysis:
    wav_to_spec = WaveformToSpectrogram(sr=sr)
    normalize_spec = NormalizeSpec(mean=cfg["mean"], std=cfg["std"])
    sess = ort.InferenceSession(onnx_path)
    inp_name = sess.get_inputs()[0].name
    labels = cfg["labels"]
    
    n_averages = 3
    prediction_history = deque(maxlen=n_averages)
    last_trigger_times = {label: 0.0 for label in labels}
    
    status_box = st.empty()
    log_box = st.empty()
    wykrycia = []

    # Glowna petla sprawdzajaca zapelniony bufor
    while webrtc_ctx.state.playing:
        current_time = time.time()
        current_buffer = list(st.session_state.audio_buffer)
        
        wav_tensor = torch.tensor(current_buffer, dtype=torch.float32)
        spectrogram = normalize_spec.encodes(wav_to_spec.encodes(wav_tensor))
        onnx_data = spectrogram.unsqueeze(0).numpy()
        
        logits = sess.run(None, {inp_name: onnx_data})[0][0]
        exp_res = np.exp(logits - np.max(logits))
        probs = exp_res / exp_res.sum()
        
        prediction_history.append(probs)
        
        if len(prediction_history) == n_averages:
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
        
        time.sleep(0.1)
