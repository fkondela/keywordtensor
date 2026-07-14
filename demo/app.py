import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from keywordtensor.core import Engine
import av
import queue
import time
import torchaudio
import random
import json
from faker import Faker
import os
from twilio.rest import Client

st.set_page_config(page_title="KeywordTensor Web", layout="wide")
st.title("KeywordTensor - prawda_falsz model")

@st.cache_data
def get_ice_servers():
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    except KeyError:
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers

with st.expander("🔴 Krok 1: Aktywuj Mikrofon (Kliknij, aby rozwinąć)", expanded=True):
    st.markdown("*Wybierz swój mikrofon z listy (Select device) i kliknij **START**, aby połączyć się z serwerem.*")

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )

if not webrtc_ctx.state.playing:
    st.warning("⚠️ Zanim przejdziesz dalej, musisz aktywować mikrofon. Rozwiń Krok 1 powyżej i kliknij START.")
    st.stop()

st.markdown("---")
st.markdown("### 🟢 Krok 2: Wybierz tryb")
mode = st.radio("Tryb aplikacji:", ["🎙️ Przetestuj Model", "🛠️ Dodaj Próbki (Admin)"], horizontal=True, label_visibility="collapsed")

if "historia_detekcji" not in st.session_state:
    st.session_state.historia_detekcji = []

ekran_logow = st.empty()

def odswiez_ekran():
    ostatnie_logi = st.session_state.historia_detekcji[-10:]
    ekran_logow.markdown("<br>".join(ostatnie_logi), unsafe_allow_html=True)

def pokaz_prawde(): 
    st.session_state.historia_detekcji.append("✅ Predicted: <b style='color:green'>PRAWDA</b>")
    odswiez_ekran()

def pokaz_falsz():  
    st.session_state.historia_detekcji.append("❌ Predicted: <b style='color:red'>FAŁSZ</b>")
    odswiez_ekran()

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

if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

if mode == "🎙️ Przetestuj Model":
    engine = Engine()
    audio_source = get_webrtc_stream(webrtc_ctx)
    engine.listen("prawda_falsz", actions=actions, source=audio_source)

else:
    st.markdown("### Panel Administracyjny (Crowdsourcing)")
    haslo = st.text_input("Hasło administracyjne:", type="password")
    
    try:
        oczekiwane_haslo = st.secrets["ADMIN_PASS"]
    except Exception:
        oczekiwane_haslo = "dev123"
        
    if haslo == oczekiwane_haslo:
        
        if True: # Stan mikrofonu jest już gwarantowany przez st.stop() na górze
            engine = Engine()
            audio_source = get_webrtc_stream(webrtc_ctx)
            
            ekran = st.empty()
            
            WLASNE_SLOWA = [
                "prawie", "prawo", "prawnik", "sprawdzam", "sprawa",
                "fauna", "fala", "fałda", "farsz", "flaszka",
                "pradawny", "falsyfikat", "szum", "wdowa", "owca",
                "trawa", "brawa", "wada", "rada", "praca", "prasa", "pstrąg", "broda", "fraza",
                "klosz", "masz", "nasz", "plusz", "gulasz", "fakt", "fart", "kosz",
                "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć", "dziesięć",
                "biały", "czarny", "czerwony", "zielony", "niebieski", "żółty", "szary",
                "dom", "drzewo", "auto", "rzeka", "krzesło", "ekran", "telefon", "woda", "słońce", "chmura", 
                "ptak", "okno", "drzwi", "książka", "lampa", "biurko", "kubek", "buty", "szkoła", "ulica"
            ]
            
            fake = Faker('pl_PL')
            ZAKAZANE_SLOWA = ["prawda", "fałsz", "falsz", "prawdę", "prawde"]

            def bezpieczne_slowo():
                if random.random() < 0.35:
                    return random.choice(WLASNE_SLOWA)
                else:
                    while True:
                        slowo = fake.word().lower()
                        if slowo not in ZAKAZANE_SLOWA:
                            return slowo
            
            def wyznacz_czasy(liczba_elementow):
                if liczba_elementow == 0: return []
                while True:
                    czasy = [random.uniform(0.1, 1.8) for _ in range(liczba_elementow)]
                    czasy.sort()
                    if liczba_elementow == 1: return czasy
                    if all(czasy[i] - czasy[i-1] >= 0.65 for i in range(1, liczba_elementow)):
                        return czasy

            def zbuduj_timeline(klasa):
                elementy = []
                L = random.choice([2, 3]) if klasa != "other" else random.choice([1, 2, 3])
                czasy = wyznacz_czasy(L)
                for t in czasy:
                    elementy.append({"start": t, "tekst": bezpieczne_slowo(), "odczytane": False, "docelowe": False})
                        
                if klasa in ["prawda", "falsz"] and L > 0:
                    idx_komendy = random.randint(0, L - 1)
                    elementy[idx_komendy]["tekst"] = "PRAWDA" if klasa == "prawda" else "FAŁSZ"
                    elementy[idx_komendy]["docelowe"] = True
                
                return elementy
            
            def clear_queue():
                if webrtc_ctx.audio_receiver:
                    try:
                        while True:
                            webrtc_ctx.audio_receiver.get_frames(timeout=0.0)
                    except queue.Empty:
                        pass

            def tworz_akcje(klasa):
                def akcja(start_callback):
                    clear_queue()
                    timeline = zbuduj_timeline(klasa)
                    
                    plan_str = " | ".join([f"**[{z['start']:.1f}s]** {z['tekst']}" for z in timeline])
                    ekran.info(f"📘 **Plan słów na to nagranie (zapoznaj się przez 5 sekund):**\n\n{plan_str}")
                    time.sleep(5.0)
                    
                    for i in [3, 2, 1]:
                        ekran.warning(f"⏳ Start za {i}...")
                        time.sleep(1.0)
                        
                    start_callback()
                    start_nagrania = time.time()
                    
                    while (t := time.time() - start_nagrania) < 3.0:
                        for z in timeline:
                            if not z["odczytane"] and t >= z["start"]:
                                if z["docelowe"]:
                                    ekran.error(f"🔴 MÓW: **{z['tekst']}**")
                                else:
                                    ekran.warning(f"⚪ MÓW: {z['tekst']}")
                                z["odczytane"] = True
                        time.sleep(0.05)
                        
                    ekran.success("✅ Nagrywanie zakończone! Trwa wysyłanie...")
                return akcja

            moje_akcje = {
                "prawda": tworz_akcje("prawda"),
                "falsz": tworz_akcje("falsz")
            }

            def zapisz_i_wyslij(klasa, index, tensor_data, sr):
                torchaudio.save("temp.wav", tensor_data, sr)
                
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=st.secrets["HF_TOKEN"])
                    baza_nazwy = f"{klasa}/probka_{int(time.time())}_{random.randint(1000, 9999)}_{index}"
                    api.upload_file(
                        path_or_fileobj="temp.wav",
                        path_in_repo=f"{baza_nazwy}.wav",
                        repo_id="fkondela/KeywordTensor_prawda_falsz", 
                        repo_type="dataset"
                    )
                except Exception as e:
                    pass
                    time.sleep(2)

            if not st.session_state.is_recording:
                if st.button("▶️ START - Rozpocznij automatyczną sesję (4 próbki)", type="primary"):
                    st.session_state.is_recording = True
                    st.rerun()
            else:
                engine.record(
                    target=zapisz_i_wyslij,
                    classes=["prawda", "falsz"],
                    samples=2,
                    actions=moje_akcje,
                    source=audio_source,
                    duration=3.0
                )
                ekran.success("✅ Koniec sesji! Wszystko wysłane.")
                st.session_state.is_recording = False
                
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
