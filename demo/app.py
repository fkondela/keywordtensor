import os
import time
import random
import gradio as gr
import numpy as np
import torchaudio
import torch
import sys
import queue
import threading
from faker import Faker
import gradio_client.utils as client_utils

# Monkey-patch naprawiający błąd boolean schema w Gradio (Wymagany z powodu nowszego Pydantic na Pythonie 3.14)
oryginalny_get_type = client_utils.get_type

def patched_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return oryginalny_get_type(schema)

client_utils.get_type = patched_get_type

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from keywordtensor.core import Engine

engine = Engine()
fake = Faker('pl_PL')

audio_queues = {}

def process_stream_detect(new_audio_chunk, state_buffer, model_name):
    if new_audio_chunk is None:
        return "Nasłuchuję...", state_buffer
        
    sr, y = new_audio_chunk
    if y.dtype != np.float32:
        y = y.astype(np.float32) / 32768.0
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    state_buffer = np.concatenate([state_buffer, y])
    max_len = int(sr * 3.0)
    state_buffer = state_buffer[-max_len:]
    
    wyniki = engine.listen(model_name, source=state_buffer.tolist(), listen_time=-1)
    
    if wyniki:
        najlepsza_klasa = max(wyniki, key=wyniki.get)
        if wyniki[najlepsza_klasa] > 0.6 and najlepsza_klasa != "other":
            kolor = "green" if najlepsza_klasa == "prawda" else "red"
            return f"<h2>🔥 Wykryto: <span style='color:{kolor}'>{najlepsza_klasa.upper()}</span> ({wyniki[najlepsza_klasa]:.2f})</h2>", state_buffer
    
    return "<h2>Nasłuchuję...</h2>", state_buffer


def process_admin_stream(new_audio_chunk, request: gr.Request):
    if new_audio_chunk is not None:
        sr, y = new_audio_chunk
        if y.dtype != np.float32:
            y = y.astype(np.float32) / 32768.0
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        session_hash = request.session_hash
        if session_hash not in audio_queues:
            audio_queues[session_hash] = queue.Queue()
        audio_queues[session_hash].put(y.tolist())
    return None

def get_audio_stream(session_hash):
    q = audio_queues.get(session_hash)
    while True:
        if q is None:
            yield None
            time.sleep(0.1)
            continue
        try:
            chunk = q.get(timeout=1.0)
            if chunk == "STOP":
                break
            yield chunk
        except queue.Empty:
            pass

def automatic_session_gradio(haslo, request: gr.Request):
    oczekiwane_haslo = os.environ.get("ADMIN_PASS", "dev123")
    if haslo != oczekiwane_haslo:
        yield "<h3>❌ Błędne hasło!</h3>"
        return
        
    session_hash = request.session_hash
    if session_hash not in audio_queues:
        audio_queues[session_hash] = queue.Queue()
        
    ui_queue = queue.Queue()
    
    while not audio_queues[session_hash].empty():
        audio_queues[session_hash].get()
        
    ZAKAZANE_SLOWA = ["prawda", "fałsz", "falsz", "prawdę", "prawde"]
    WLASNE_SLOWA = ["prawie", "prawo", "prawnik", "sprawdzam", "sprawa", "fauna", "fala", "szum", "wdowa", "owca"]
    
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

    def tworz_akcje(klasa):
        def akcja(start_callback):
            while not audio_queues[session_hash].empty():
                audio_queues[session_hash].get()
                
            timeline = zbuduj_timeline(klasa)
            plan_str = " | ".join([f"**[{z['start']:.1f}s]** {z['tekst']}" for z in timeline])
            ui_queue.put(f"<h3>📘 Plan słów (zapoznaj się przez 5s):</h3><p>{plan_str}</p>")
            time.sleep(5.0)
            
            for i in [3, 2, 1]:
                ui_queue.put(f"<h3>⏳ Start za {i}...</h3>")
                time.sleep(1.0)
                
            start_callback()
            start_nagrania = time.time()
            
            while (t := time.time() - start_nagrania) < 3.0:
                html_out = ""
                for z in timeline:
                    if not z["odczytane"] and t >= z["start"]:
                        if z["docelowe"]:
                            html_out = f"<h2>🔴 MÓW: <span style='color:red'>{z['tekst']}</span></h2>"
                        else:
                            html_out = f"<h2>⚪ MÓW: {z['tekst']}</h2>"
                        z["odczytane"] = True
                if html_out:
                    ui_queue.put(html_out)
                time.sleep(0.05)
                
            ui_queue.put("<h3>✅ Nagrywanie zakończone! Trwa wysyłanie...</h3>")
        return akcja

    moje_akcje = {
        "prawda": tworz_akcje("prawda"),
        "falsz": tworz_akcje("falsz")
    }

    def zapisz_i_wyslij(klasa, index, tensor_data, sr):
        torchaudio.save("temp.wav", tensor_data, sr)
        try:
            from huggingface_hub import HfApi
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                ui_queue.put("<h3>❌ Brak tokenu HF_TOKEN!</h3>")
                return
            api = HfApi(token=hf_token)
            baza_nazwy = f"{klasa}/probka_{int(time.time())}_{random.randint(1000, 9999)}_{index}"
            api.upload_file(
                path_or_fileobj="temp.wav",
                path_in_repo=f"{baza_nazwy}.wav",
                repo_id="fkondela/KeywordTensor_prawda_falsz", 
                repo_type="dataset"
            )
            ui_queue.put(f"<h3>✅ Próbka dla {klasa} wysłana na serwer!</h3>")
            time.sleep(2.0)
        except Exception as e:
            ui_queue.put(f"<h3>❌ Błąd wysyłania: {str(e)}</h3>")
            time.sleep(2.0)

    def run_engine():
        engine.record(
            target=zapisz_i_wyslij,
            classes=["prawda", "falsz"],
            samples=2,
            actions=moje_akcje,
            source=get_audio_stream(session_hash),
            duration=3.0
        )
        ui_queue.put("ZAKONCZONO")

    t = threading.Thread(target=run_engine)
    t.start()

    ostatni_tekst = "<h3>Rozpoczynamy sesję...</h3>"
    yield ostatni_tekst
    
    while t.is_alive() or not ui_queue.empty():
        try:
            msg = ui_queue.get(timeout=0.1)
            if msg == "ZAKONCZONO":
                break
            ostatni_tekst = msg
            yield ostatni_tekst
        except queue.Empty:
            pass

    yield "<h3>✅ Koniec sesji automatycznej! Dziękujemy.</h3>"


theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

with gr.Blocks(theme=theme, title="KeywordTensor Web") as demo:
    gr.Markdown("# 🎙️ KeywordTensor - Wersja Chmurowa (Gradio)")
    
    with gr.Tab("Live Streaming (Detekcja)"):
        gr.Markdown("Włącz mikrofon i powiedz **'prawda'** lub **'fałsz'**.")
        model_dropdown = gr.Dropdown(choices=["prawda_falsz"], value="prawda_falsz", label="Aktywny Model ONNX")
        user_buffer = gr.State(value=np.array([], dtype=np.float32))
        
        with gr.Row():
            audio_in = gr.Audio(sources=["microphone"], streaming=True, label="Twój Mikrofon")
            wynik_out = gr.HTML("<h2>Zacznij mówić...</h2>")
            
        audio_in.stream(fn=process_stream_detect, inputs=[audio_in, user_buffer, model_dropdown], outputs=[wynik_out, user_buffer])
        
    with gr.Tab("🛠️ Crowdsourcing Próbek (Admin)"):
        gr.Markdown("Zostaw włączony mikrofon i naciśnij Start. Aplikacja przeprowadzi Cię przez sesję automatycznie tak jak stara wersja.")
        
        with gr.Row():
            haslo_input = gr.Textbox(label="Hasło Administracyjne", type="password")
            
        admin_audio_in = gr.Audio(sources=["microphone"], streaming=True, label="Utrzymuj mikrofon włączony!")
        start_btn = gr.Button("🚀 Rozpocznij automatyczną sesję (4 próbki)", variant="primary")
        ekran_admina = gr.HTML("<h3>Oczekuję...</h3>")
        
        admin_audio_in.stream(fn=process_admin_stream, inputs=[admin_audio_in], outputs=[])
        
        start_btn.click(
            fn=automatic_session_gradio,
            inputs=[haslo_input],
            outputs=[ekran_admina]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)
