import os
import sys
import time
import random
import queue
import threading
import numpy as np
import torch
import torchaudio
import gradio as gr
import av
from faker import Faker
from keywordtensor.core import Engine

engine = Engine()
fake = Faker('pl_PL')

audio_queues = {}
ui_queues = {}
is_live = {}
session_resamplers = {}

def get_session_audio(session_hash):
    q = audio_queues[session_hash]
    while is_live.get(session_hash, False):
        try:
            chunk = q.get(timeout=0.1)
            if chunk == "STOP":
                break
            yield chunk
        except queue.Empty:
            yield None

def handle_audio_stream(chunk, request: gr.Request):
    session_hash = request.session_hash
    if session_hash not in audio_queues:
        audio_queues[session_hash] = queue.Queue()
    
    if chunk is not None:
        sr, y = chunk
        if y.dtype != np.float32:
            y = y.astype(np.float32) / 32768.0
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        if sr != 16000:
            if session_hash not in session_resamplers:
                session_resamplers[session_hash] = av.AudioResampler(format='flt', layout='mono', rate=16000)
            
            y = y.reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(y, format='flt', layout='mono')
            frame.sample_rate = sr
            
            resampler = session_resamplers[session_hash]
            clean_audio = []
            for clean_frame in resampler.resample(frame):
                clean_audio.extend(clean_frame.to_ndarray()[0].tolist())
            
            audio_queues[session_hash].put(clean_audio)
        else:
            audio_queues[session_hash].put(y.tolist())

def live_mode_generator(request: gr.Request):
    session_hash = request.session_hash
    is_live[session_hash] = True
    
    if session_hash not in audio_queues:
        audio_queues[session_hash] = queue.Queue()
    ui_queues[session_hash] = queue.Queue()
    
    def prawda_cb():
        ui_queues[session_hash].put("<h2>🔥 Wykryto: <span style='color:green'>PRAWDA</span></h2>")
    def falsz_cb():
        ui_queues[session_hash].put("<h2>🔥 Wykryto: <span style='color:red'>FAŁSZ</span></h2>")
        
    def thread_func():
        engine.listen(
            "prawda_falsz", 
            actions={
                "prawda": {"function": prawda_cb, "cooldown": 2.0}, 
                "falsz": {"function": falsz_cb, "cooldown": 2.0}
            }, 
            source=get_session_audio(session_hash)
        )
        
    t = threading.Thread(target=thread_func)
    t.start()
    
    yield "<h2>Oczekuję na detekcję... (Mów do mikrofonu)</h2>"
    
    while is_live.get(session_hash, False):
        try:
            msg = ui_queues[session_hash].get(timeout=0.1)
            yield msg
            time.sleep(1.5)
            yield "<h2>Nasłuchuję...</h2>"
        except queue.Empty:
            pass
            
    t.join()
    yield "<h2>Zatrzymano.</h2>"

def admin_mode_generator(haslo, request: gr.Request):
    session_hash = request.session_hash
    oczekiwane_haslo = os.environ.get("ADMIN_PASS", "dev123")
    if haslo != oczekiwane_haslo:
        yield "<h3>❌ Błędne hasło!</h3>"
        return
        
    is_live[session_hash] = True
    if session_hash not in audio_queues:
        audio_queues[session_hash] = queue.Queue()
    ui_queues[session_hash] = queue.Queue()
    
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
            ui_queues[session_hash].put(f"<h3>📘 Plan (5s):</h3><p>{plan_str}</p>")
            time.sleep(5.0)
            
            for i in [3, 2, 1]:
                ui_queues[session_hash].put(f"<h3>⏳ Start za {i}...</h3>")
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
                    ui_queues[session_hash].put(html_out)
                time.sleep(0.05)
                
            ui_queues[session_hash].put("<h3>✅ Zakończono! Wysyłanie...</h3>")
        return akcja

    moje_akcje = {
        "prawda": tworz_akcje("prawda"),
        "falsz": tworz_akcje("falsz")
    }

    def zapisz_i_wyslij(klasa, index, tensor_data, sr):
        torchaudio.save("/tmp/temp.wav", tensor_data, sr)
        try:
            from huggingface_hub import HfApi
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                ui_queues[session_hash].put("<h3>❌ Brak tokenu HF_TOKEN!</h3>")
                return
            api = HfApi(token=hf_token)
            baza_nazwy = f"{klasa}/probka_{int(time.time())}_{random.randint(1000, 9999)}_{index}"
            api.upload_file(
                path_or_fileobj="/tmp/temp.wav",
                path_in_repo=f"{baza_nazwy}.wav",
                repo_id="fkondela/KeywordTensor_prawda_falsz", 
                repo_type="dataset"
            )
            ui_queues[session_hash].put(f"<h3>✅ Wysłano próbkę: {klasa}</h3>")
        except Exception as e:
            ui_queues[session_hash].put(f"<h3>❌ Błąd: {str(e)}</h3>")

    def thread_func():
        engine.record(
            target=zapisz_i_wyslij,
            classes=["prawda", "falsz"],
            samples=2,
            actions=moje_akcje,
            source=get_session_audio(session_hash),
            duration=3.0
        )
        ui_queues[session_hash].put("ZAKONCZONO")

    t = threading.Thread(target=thread_func)
    t.start()
    
    yield "<h3>Rozpoczynamy...</h3>"
    
    while is_live.get(session_hash, False):
        try:
            msg = ui_queues[session_hash].get(timeout=0.1)
            if msg == "ZAKONCZONO":
                break
            yield msg
        except queue.Empty:
            pass
            
    is_live[session_hash] = False
    yield "<h3>✅ Koniec sesji.</h3>"

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

with gr.Blocks(title="KeywordTensor Web", theme=theme) as demo:
    gr.Markdown("# 🎙️ KeywordTensor - Wersja Chmurowa")
    
    with gr.Group(visible=True) as gate_group:
        gr.Markdown("### 👆 Krok 1: Zezwól na dostęp i włącz mikrofon poniżej, aby odblokować aplikację.")
        audio_in = gr.Audio(sources=["microphone"], streaming=True, label="Mikrofon Główny")
        
    with gr.Group(visible=False) as menu_group:
        gr.Markdown("### 🎛️ Wybierz tryb działania:")
        with gr.Row():
            btn_menu_live = gr.Button("🔴 Detekcja na żywo", variant="primary")
            btn_menu_admin = gr.Button("🛠️ Panel Administratora", variant="secondary")
            
    with gr.Group(visible=False) as live_group:
        btn_back_live = gr.Button("🔙 Zatrzymaj i Wróć do menu", variant="stop")
        gr.Markdown("---")
        btn_start_live = gr.Button("🚀 Rozpocznij Detekcję", variant="primary")
        live_output = gr.HTML("<h2>Oczekuję na start...</h2>")
        
    with gr.Group(visible=False) as admin_group:
        btn_back_admin = gr.Button("🔙 Zatrzymaj i Wróć do menu", variant="stop")
        gr.Markdown("---")
        admin_password = gr.Textbox(label="Hasło Administracyjne", type="password")
        btn_start_admin = gr.Button("🚀 Rozpocznij Sesję Próbek", variant="primary")
        admin_output = gr.HTML("<h3>Oczekuję na start...</h3>")

    audio_in.start_recording(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[gate_group, menu_group]
    )
    
    def nav_to_live(): return gr.update(visible=False), gr.update(visible=True)
    def nav_to_admin(): return gr.update(visible=False), gr.update(visible=True)
    
    def nav_back(request: gr.Request):
        session = request.session_hash
        if session in is_live:
            is_live[session] = False
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

    btn_menu_live.click(nav_to_live, outputs=[menu_group, live_group])
    btn_menu_admin.click(nav_to_admin, outputs=[menu_group, admin_group])
    
    btn_back_live.click(nav_back, outputs=[menu_group, live_group, admin_group, btn_start_live, btn_start_admin])
    btn_back_admin.click(nav_back, outputs=[menu_group, live_group, admin_group, btn_start_live, btn_start_admin])
    
    audio_in.stream(
        fn=handle_audio_stream,
        inputs=[audio_in]
    )
    
    btn_start_live.click(
        fn=lambda: gr.update(visible=False),
        outputs=[btn_start_live]
    ).then(
        fn=live_mode_generator,
        outputs=[live_output]
    )
    
    btn_start_admin.click(
        fn=lambda: gr.update(visible=False),
        outputs=[btn_start_admin]
    ).then(
        fn=admin_mode_generator,
        inputs=[admin_password],
        outputs=[admin_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000, theme=theme)
