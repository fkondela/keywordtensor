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

st.markdown("---")
st.markdown("### 🟢 Krok 2: Wybierz tryb")
mode = st.radio("Tryb aplikacji:", ["🎙️ Przetestuj Model", "🛠️ Dodaj Próbki (Admin)", "🎮 Zagraj w Quiz"], horizontal=True, label_visibility="collapsed")

if "previous_mode" not in st.session_state:
    st.session_state.previous_mode = mode

if st.session_state.previous_mode != mode:
    st.session_state.previous_mode = mode
    st.session_state.quiz_started = False
    st.session_state.is_recording = False

webrtc_ctx = webrtc_streamer(
    key=f"speech-to-text-{mode}",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=2048,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
)

if "historia_detekcji" not in st.session_state:
    st.session_state.historia_detekcji = []

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

class QuizAnswerDetected(Exception):
    def __init__(self, answer):
        self.answer = answer

def ans_prawda():
    if time.time() - st.session_state.get("quiz_listen_start", 0) > 2.5:
        raise QuizAnswerDetected("prawda")

def ans_falsz():
    if time.time() - st.session_state.get("quiz_listen_start", 0) > 2.5:
        raise QuizAnswerDetected("falsz")

quiz_actions = {
    "prawda": {"function": ans_prawda, "cooldown": 0.0},
    "falsz": {"function": ans_falsz, "cooldown": 0.0}
}

def timed_webrtc_stream(ctx, ekran_statusu, timeout=10.0, sr=16000):
    resampler = av.AudioResampler(format='flt', layout='mono', rate=sr)
    start_time = time.time()
    
    if ctx.audio_receiver:
        try:
            while True:
                ctx.audio_receiver.get_frames(timeout=0.0)
        except queue.Empty:
            pass

    while ctx.state.playing:
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            break
            
        pozostalo = int(timeout - elapsed)
        ekran_statusu.info(f"⏳ Słucham... Masz {pozostalo} sekund. Odpowiedz wyraźnie: PRAWDA lub FAŁSZ.")
        
        try:
            if ctx.audio_receiver is None:
                frames = []
            else:
                frames = ctx.audio_receiver.get_frames(timeout=0.1)
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

if mode == "🎙️ Przetestuj Model":
    if not webrtc_ctx.state.playing:
        st.warning("⚠️ Zanim przejdziesz dalej, musisz włączyć mikrofon klikając przycisk START powyżej.")
    else:
        if "test_started" not in st.session_state:
            st.session_state.test_started = False
            
        if not st.session_state.test_started:
            if st.button("▶️ Rozpocznij Testowanie Modelu na żywo", type="primary"):
                st.session_state.test_started = True
                st.rerun()
        else:
            st.success("Nasłuchiwanie aktywne...")
            
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

            test_actions = {
                "prawda": {"function": pokaz_prawde, "cooldown": 2.0}, 
                "falsz": {"function": pokaz_falsz, "cooldown": 2.0}
            }
            
            odswiez_ekran()
            
            engine = Engine()
            audio_source = get_webrtc_stream(webrtc_ctx)
            engine.listen("prawda_falsz", actions=test_actions, source=audio_source)

elif mode == "🎮 Zagraj w Quiz":
    st.subheader("🪐 Kosmiczny Quiz Głosowy AI")
    
    ekran_audio = st.empty()
    
    def play_tts(text):
        from gtts import gTTS
        import io
        import base64
        tts = gTTS(text=text, lang='pl')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        b64 = base64.b64encode(fp.getvalue()).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        ekran_audio.markdown(md, unsafe_allow_html=True)
        
    pytania = [
        {"q": "Czarna dziura świeci tak jasno, że widać ją z daleka.", "a": "falsz", "exp": "Czarna dziura pochłania światło i jest niewidoczna."},
        {"q": "Słońce to ponad 99 procent masy całego układu słonecznego.", "a": "prawda", "exp": "Dokładnie tak, Słońce jest gigantyczne."},
        {"q": "Księżyc ma własne źródło światła.", "a": "falsz", "exp": "Księżyc tylko odbija światło słoneczne."},
        {"q": "Droga Mleczna to nasza galaktyka.", "a": "prawda", "exp": "Zgadza się, żyjemy w Drodze Mlecznej."},
        {"q": "Woda w kosmosie w ogóle nie występuje.", "a": "falsz", "exp": "Woda w postaci lodu jest bardzo powszechna w kosmosie."},
        {"q": "Ziemia krąży wokół Słońca.", "a": "prawda", "exp": "Ziemia robi pełne okrążenie w 365 dni."},
        {"q": "Wenus jest najzimniejszą planetą.", "a": "falsz", "exp": "Wenus to najgorętsza planeta ze względu na atmosferę."},
        {"q": "Gwiazda Polarna zawsze wskazuje północ.", "a": "prawda", "exp": "Na naszej półkuli to prawda."},
        {"q": "Mars jest nazywany Błękitną Planetą.", "a": "falsz", "exp": "Mars to Czerwona Planeta, a Błękitną jest Ziemia."},
        {"q": "Na Księżycu jest grawitacja.", "a": "prawda", "exp": "Jest, ale około 6 razy słabsza niż na Ziemi."}
    ]
    
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
        
    if not webrtc_ctx.state.playing:
        st.warning("⚠️ Zanim przejdziesz dalej, musisz włączyć mikrofon klikając przycisk START powyżej.")
    elif not st.session_state.quiz_started:
        st.markdown("Witaj w interaktywnym quizie! Asystent AI zada Ci 10 pytań o kosmosie. Po każdym pytaniu masz 10 sekund na odpowiedź (Prawda lub Fałsz) do mikrofonu.")
        if st.button("▶️ Rozpocznij Quiz (Włącz dźwięk!)", type="primary"):
            st.session_state.quiz_started = True
            st.session_state.quiz_q_idx = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_state = "ASKING"
            if "quiz_end_tts" in st.session_state: del st.session_state.quiz_end_tts
            st.rerun()
    else:
        idx = st.session_state.quiz_q_idx
        if idx >= len(pytania):
            st.success(f"🎉 Koniec Quizu! Twój wynik to: {st.session_state.quiz_score} / {len(pytania)}")
            if "quiz_end_tts" not in st.session_state:
                play_tts(f"Koniec quizu! Twój wynik to {st.session_state.quiz_score} na {len(pytania)}. Dziękuję za grę!")
                st.session_state.quiz_end_tts = True
            if st.button("🔄 Zagraj ponownie"):
                st.session_state.quiz_started = False
                st.rerun()
        else:
            q = pytania[idx]
            st.markdown(f"### Pytanie {idx+1}/10")
            st.markdown(f"**{q['q']}**")
            ekran_statusu = st.empty()
            ekran_feedback = st.empty()
            
            if st.session_state.quiz_state == "ASKING":
                ekran_statusu.info("Lektor czyta pytanie...")
                slowne_numery = ["pierwsze", "drugie", "trzecie", "czwarte", "piąte", "szóste", "siódme", "ósme", "dziewiąte", "dziesiąte"]
                play_tts(f"Pytanie {slowne_numery[idx]}. {q['q']}")
                time.sleep(6.0) # Czas na przeczytanie pytania
                st.session_state.quiz_state = "LISTENING"
                st.rerun()
                
            elif st.session_state.quiz_state == "LISTENING":
                engine = Engine()
                audio_source = timed_webrtc_stream(webrtc_ctx, ekran_statusu, timeout=10.0)
                odpowiedz = "brak"
                st.session_state.quiz_listen_start = time.time()
                try:
                    engine.listen("prawda_falsz", actions=quiz_actions, source=audio_source)
                except QuizAnswerDetected as e:
                    odpowiedz = e.answer
                
                if odpowiedz == "brak":
                    st.session_state.quiz_feedback = "Brak odpowiedzi. Czas minął."
                    st.session_state.quiz_feedback_tts = f"Brak odpowiedzi. {q['exp']}"
                elif odpowiedz == q["a"]:
                    st.session_state.quiz_score += 1
                    st.session_state.quiz_feedback = f"✅ Świetnie! (Zrozumiano: {odpowiedz.upper()})"
                    st.session_state.quiz_feedback_tts = f"Świetnie! {q['exp']}"
                else:
                    st.session_state.quiz_feedback = f"❌ Pudło! (Zrozumiano: {odpowiedz.upper()})"
                    st.session_state.quiz_feedback_tts = f"Pudło. {q['exp']}"
                
                st.session_state.quiz_state = "FEEDBACK"
                st.rerun()
                
            elif st.session_state.quiz_state == "FEEDBACK":
                with ekran_feedback.container():
                    st.info(st.session_state.quiz_feedback)
                    st.markdown(f"*(Wyjaśnienie: {q['exp']})*")
                
                if "feedback_played" not in st.session_state or st.session_state.feedback_played != idx:
                    play_tts(st.session_state.quiz_feedback_tts)
                    st.session_state.feedback_played = idx
                    time.sleep(5.0) 
                    st.session_state.quiz_q_idx += 1
                    st.session_state.quiz_state = "ASKING"
                    st.rerun()

else:
    st.markdown("### Panel Administracyjny (Crowdsourcing)")
    if not webrtc_ctx.state.playing:
        st.warning("⚠️ Zanim przejdziesz dalej, musisz włączyć mikrofon klikając przycisk START powyżej.")
    else:
        if not st.session_state.is_recording:
            haslo = st.text_input("Hasło administracyjne:", type="password")
            
            try:
                oczekiwane_haslo = st.secrets["ADMIN_PASS"]
            except Exception:
                oczekiwane_haslo = "dev123"
                
            if haslo == oczekiwane_haslo:
                if st.button("▶️ Rozpocznij automatyczną sesję dodawania próbek", type="primary"):
                    st.session_state.is_recording = True
                    st.rerun()
            elif haslo:
                st.error("Nieprawidłowe hasło!")
        else:
            st.success("Sesja nagrywania aktywna. Wykonuj polecenia z ekranu...")
            
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
            time.sleep(2.0)
            st.rerun()
