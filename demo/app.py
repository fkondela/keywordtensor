import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from keywordtensor.core import Engine

st.set_page_config(page_title="KeywordTensor Web", layout="wide")
st.title("KeywordTensor - Rozpoznawanie w Przeglądarce")

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
)

def pokaz_prawde(): 
    st.success("✅ WYKRYTO: PRAWDA")

def pokaz_falsz():  
    st.error("❌ WYKRYTO: FAŁSZ")

actions = {
    "prawda": pokaz_prawde, 
    "falsz": pokaz_falsz
}

if webrtc_ctx.state.playing:
    engine = Engine()
    engine.listen("prawda_falsz", actions=actions, source=webrtc_ctx)
