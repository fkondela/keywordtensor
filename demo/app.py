import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from keywordtensor.core import Engine
import av
import queue
import time

st.set_page_config(page_title="KeywordTensor Web", layout="wide")
st.title("KeywordTensor - prawda_falsz model")

mode = st.radio("Tryb aplikacji:", ["🎙️ Przetestuj Model", "🛠️ Dodaj Próbki (Admin)"], horizontal=True)

# Przesuwamy WebRTC na samą górę, żeby zawsze renderował się stabilnie na stronie.
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
            if ctx.audio_receiver is None:
                frames = []
            else:
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

# Obsługa stanu nagrywania - odporna na migotanie interfejsu (przeładowania)
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

if mode == "🎙️ Przetestuj Model":
    # ORYGINALNY TRYB
    if webrtc_ctx.state.playing:
        engine = Engine()
        audio_source = get_webrtc_stream(webrtc_ctx)
        engine.listen("prawda_falsz", actions=actions, source=audio_source)

else:
    # TRYB ADMINA
    st.markdown("### Panel Administracyjny (Crowdsourcing)")
    haslo = st.text_input("Hasło administracyjne:", type="password")
    
    try:
        oczekiwane_haslo = st.secrets["ADMIN_PASS"]
    except Exception:
        oczekiwane_haslo = "dev123"
        
    if haslo == oczekiwane_haslo:
        
        if webrtc_ctx.state.playing:
            engine = Engine()
            audio_source = get_webrtc_stream(webrtc_ctx)
            
            ekran = st.empty()
            
            def clear_queue():
                if webrtc_ctx.audio_receiver:
                    try:
                        while True:
                            webrtc_ctx.audio_receiver.get_frames(timeout=0.0)
                    except queue.Empty:
                        pass

            def akcja_prawda():
                clear_queue()
                ekran.error("🔴 MÓW TERAZ: PRAWDA (Trwa nagrywanie 3.0s...)")
                
            def akcja_falsz():
                clear_queue()
                ekran.error("🔴 MÓW TERAZ: FAŁSZ (Trwa nagrywanie 3.0s...)")

            moje_akcje = {
                "prawda": akcja_prawda,
                "falsz": akcja_falsz
            }

            def zapisz_i_wyslij(klasa, index, tensor_data, sr):
                ekran.warning(f"⏳ Zapisuję próbkę dla: {klasa.upper()}...")
                
                import torchaudio
                torchaudio.save("temp.wav", tensor_data, sr)
                
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=st.secrets["HF_TOKEN"])
                    nazwa_pliku = f"{klasa}/probka_{int(time.time())}_{index}.wav"
                    api.upload_file(
                        path_or_fileobj="temp.wav",
                        path_in_repo=nazwa_pliku,
                        repo_id="fkondela/testowy_zbior_audio", # ZMIEŃ NA SWOJE REPOZYTORIUM
                        repo_type="dataset"
                    )
                except Exception as e:
                    ekran.error(f"Błąd wysyłania (Brak HF_TOKEN lub awaria HF): {e}")
                    time.sleep(2)

            if not st.session_state.is_recording:
                if st.button("Rozpocznij automatyczną sesję (4 próbki)"):
                    st.session_state.is_recording = True
                    st.rerun()
            else:
                # Rozpoczęto pętlę nagrywania - to się wykona odpornie na refresh
                engine.record(
                    target=zapisz_i_wyslij,
                    classes=["prawda", "falsz"],
                    samples=2,
                    actions=moje_akcje,
                    source=audio_source,
                    duration=3.0
                )
                ekran.success("✅ Koniec sesji! Wszystko wysłane.")
                # Resetujemy stan po skończeniu sesji
                st.session_state.is_recording = False
                
            # Idle loop podtrzymujący WebRTC przy życiu kiedy nie ma nagrywania
            if not st.session_state.is_recording:
                while webrtc_ctx.state.playing:
                    try:
                        if webrtc_ctx.audio_receiver:
                            webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
                    except queue.Empty:
                        pass
                    time.sleep(0.05)
                
    elif haslo != "":
        st.error("Błędne hasło!")
