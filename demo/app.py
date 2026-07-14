import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from keywordtensor.core import Engine
import av
import queue

st.set_page_config(page_title="KeywordTensor Web", layout="wide")
st.title("KeywordTensor - prawda_falsz model")

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
)

def pokaz_prawde(): 
    st.success("✅ Predicted: PRAWDA")

def pokaz_falsz():  
    st.error("❌ Predicted: FAŁSZ")

actions = {
    "prawda": {"function": pokaz_prawde, "cooldown": 2.0}, 
    "falsz": {"function": pokaz_falsz, "cooldown": 2.0}
}

def get_webrtc_stream(ctx, sr=16000):
    resampler = av.AudioResampler(format='flt', layout='mono', rate=sr)
    while ctx.state.playing:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=0.0)
        except queue.Empty:
            frames = []
            
        clean_audio = []
        for frame in frames:
            for clean_frame in resampler.resample(frame):
                clean_audio.extend(clean_frame.to_ndarray()[0].tolist())
                
        if clean_audio:
            yield clean_audio
        else:
            yield None

if webrtc_ctx.state.playing:
    engine = Engine()
    audio_source = get_webrtc_stream(webrtc_ctx)
    engine.listen("prawda_falsz", actions=actions, source=audio_source)
