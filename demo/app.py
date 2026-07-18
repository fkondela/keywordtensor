import os
import time
import random
import queue
import threading
import numpy as np
import torch
import torchaudio
import gradio as gr
from faker import Faker
from keywordtensor.core import Engine

torch.set_num_threads(1)

engine = Engine()
fake = Faker('pl_PL')

class SharedAudioState:
    def __init__(self):
        self.raw_buffer = []
        self.latest_resampled_3s = None
        self.lock = threading.Lock()
        self.new_data_event = threading.Event()
        
    def clear(self):
        with self.lock:
            self.raw_buffer = []
            self.latest_resampled_3s = None
        self.new_data_event.clear()

def get_audio_stream(shared_state, live_flag, sliding_window=True):
    while live_flag[0]:
        shared_state.new_data_event.wait(timeout=0.1)
        shared_state.new_data_event.clear()
        
        with shared_state.lock:
            freshest_buf = shared_state.latest_resampled_3s
            
        if freshest_buf is not None:
            yield freshest_buf

def consume_ui_events(ui_queue, thread, live_flag):
    error_occurred = False
    try:
        while live_flag[0]:
            try:
                msg = ui_queue.get(timeout=0.1)
                if msg == "ZAKONCZONO":
                    break
                yield msg, gr.update(visible=False)
                if "ERROR" in msg:
                    error_occurred = True
                    live_flag[0] = False
                    break
            except queue.Empty:
                pass
    finally:
        live_flag[0] = False
        thread.join(timeout=1.0)
        
    if not error_occurred:
        yield "<h3>Zakończono bezpiecznie.</h3>", gr.update(visible=True)

def handle_audio_stream(chunk, shared_state):
    if chunk is not None:
        sr, y = chunk
        if y.dtype != np.float32:
            y = y.astype(np.float32) / 32768.0
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        y_chunk = y.tolist()
        
        with shared_state.lock:
            shared_state.raw_buffer.extend(y_chunk)
            target = int(sr * 3.0)
            
            if len(shared_state.raw_buffer) > target:
                shared_state.raw_buffer = shared_state.raw_buffer[-target:]
                
            if len(shared_state.raw_buffer) >= target:
                process_buf = shared_state.raw_buffer
                y_tensor = torch.tensor(process_buf, dtype=torch.float32)
                if y_tensor.abs().max() > 1.0:
                    y_tensor = y_tensor / 32768.0
                if sr != 16000:
                    y_tensor = torchaudio.functional.resample(y_tensor, orig_freq=sr, new_freq=16000)
                
                shared_state.latest_resampled_3s = y_tensor.numpy().tolist()
                shared_state.new_data_event.set()

def live_mode_generator(shared_state, ui_queue, live_flag):
    live_flag[0] = True
    shared_state.clear()
    with ui_queue.mutex:
        ui_queue.queue.clear()
        
    def prawda_cb():
        ui_queue.put("<h2>Detected: <span style='color:green'>PRAWDA</span></h2>")
    def falsz_cb():
        ui_queue.put("<h2>Detected: <span style='color:red'>FAŁSZ</span></h2>")
    def other_cb():
        ui_queue.put("<h2>Detected: <span style='color:gray'>OTHER</span></h2>")
        
    def thread_func():
        try:
            engine.listen(
                "prawda_falsz", 
                actions={
                    "prawda": {"function": prawda_cb, "cooldown": 3.0}, 
                    "falsz": {"function": falsz_cb, "cooldown": 3.0},
                    "other": {"function": other_cb, "cooldown": 0.0}
                }, 
                source=get_audio_stream(shared_state, live_flag, sliding_window=True),
                min_confidence=0.55,
                n_averages=1,
                threads=1
            )
        except Exception as e:
            ui_queue.put(f"<h2>ERROR: {str(e)}</h2>")
            
    t = threading.Thread(target=thread_func)
    t.start()
    
    yield "<h2>Detected: <span style='color:gray'>OTHER</span></h2>", gr.update(visible=False)
    yield from consume_ui_events(ui_queue, t, live_flag)

def admin_mode_generator(password, shared_state, ui_queue, live_flag):
    expected_password = os.environ.get("ADMIN_PASS", "dev123")
    if password != expected_password:
        yield "<h3>Invalid Password!</h3>", gr.update(visible=True)
        return
        
    live_flag[0] = True
    shared_state.clear()
    with ui_queue.mutex:
        ui_queue.queue.clear()
        
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
            shared_state.clear()
                
            timeline = zbuduj_timeline(klasa)
            plan_str = " | ".join([f"**[{z['start']:.1f}s]** {z['tekst']}" for z in timeline])
            ui_queue.put(f"<h3>Plan (5s):</h3><p>{plan_str}</p>")
            
            for _ in range(50):
                if not live_flag[0]: return
                time.sleep(0.1)
            
            for i in [3, 2, 1]:
                if not live_flag[0]: return
                ui_queue.put(f"<h3>Start in {i}...</h3>")
                time.sleep(1.0)
            
            shared_state.clear()
                
            start_callback()
            start_nagrania = time.time()
            
            while (t := time.time() - start_nagrania) < 3.0:
                if not live_flag[0]: return
                html_out = ""
                for z in timeline:
                    if not z["odczytane"] and t >= z["start"]:
                        if z["docelowe"]:
                            html_out = f"<h2>SPEAK: <span style='color:red'>{z['tekst']}</span></h2>"
                        else:
                            html_out = f"<h2>SPEAK: {z['tekst']}</h2>"
                        z["odczytane"] = True
                if html_out:
                    ui_queue.put(html_out)
                time.sleep(0.05)
                
            ui_queue.put("<h3>Recording finished! Uploading...</h3>")
        return akcja

    moje_akcje = {
        "prawda": tworz_akcje("prawda"),
        "falsz": tworz_akcje("falsz")
    }

    def zapisz_i_wyslij(klasa, index, tensor_data, sr):
        import wave
        import numpy as np
        audio_np = tensor_data.squeeze().numpy()
        audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
        
        temp_file_path = f"/tmp/temp_{int(time.time())}_{random.randint(10000, 99999)}_{index}.wav"
        
        with wave.open(temp_file_path, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(audio_np.tobytes())
        try:
            from huggingface_hub import HfApi
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                ui_queue.put("<h3>ERROR: Missing HF_TOKEN!</h3>")
                return
            api = HfApi(token=hf_token)
            baza_nazwy = f"{klasa}/probka_{int(time.time())}_{random.randint(1000, 9999)}_{index}"
            api.upload_file(
                path_or_fileobj=temp_file_path,
                path_in_repo=f"{baza_nazwy}.wav",
                repo_id="fkondela/KeywordTensor_prawda_falsz", 
                repo_type="dataset"
            )
            ui_queue.put(f"<h3>Successfully uploaded: {klasa}</h3>")
        except Exception as e:
            ui_queue.put(f"<h3>ERROR: {str(e)}</h3>")
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

    def thread_func():
        try:
            engine.record(
                target=zapisz_i_wyslij,
                classes=["prawda", "falsz"],
                samples=2,
                actions=moje_akcje,
                source=get_audio_stream(shared_state, live_flag, sliding_window=False),
                duration=3.0
            )
            ui_queue.put("ZAKONCZONO")
        except Exception as e:
            ui_queue.put(f"<h3>ERROR: {str(e)}</h3>")
            ui_queue.put("ZAKONCZONO")

    t = threading.Thread(target=thread_func)
    t.start()
    
    yield "<h3>Starting...</h3>", gr.update(visible=False)
    yield from consume_ui_events(ui_queue, t, live_flag)

custom_css = """
.header-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
    margin-bottom: 20px;
    margin-top: 10px;
}
.header-container img {
    height: 50px;
    border-radius: 12px;
    filter: drop-shadow(0px 2px 4px rgba(0, 0, 0, 0.2));
}
.header-container h1 {
    margin: 0;
    color: #3b82f6 !important;
    font-weight: bold !important;
    font-size: 2.2em;
}
footer {
    display: none !important;
}
button[aria-label="Settings"] {
    display: none !important;
}
"""

head_html = '<link rel="icon" type="image/png" href="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.png">'
with gr.Blocks(title="KeywordTensor", head=head_html) as demo:
    gr.HTML('''
    <div class="header-container">
        <img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.png" alt="KeywordTensor Logo">
        <h1>KeywordTensor - prawda_falsz model</h1>
    </div>
    ''')
    
    audio_queue_state = gr.State(lambda: SharedAudioState())
    ui_queue_state = gr.State(lambda: queue.Queue())
    live_flag_state = gr.State(lambda: [False])
    
    with gr.Accordion("Step 1: Select Microphone", open=True) as mic_group:
        audio_in = gr.Audio(sources=["microphone"], streaming=True, label="Audio Stream")
        btn_confirm_mic = gr.Button("Next", variant="primary", interactive=False)
        
    with gr.Group(visible=False) as menu_group:
        gr.Markdown("### Select Mode:")
        with gr.Row():
            btn_menu_live = gr.Button("Live Mode", variant="primary")
            btn_menu_admin = gr.Button("Admin Panel", variant="secondary")
            
    with gr.Group(visible=False) as live_group:
        btn_back_live = gr.Button("Stop", variant="stop")
        btn_start_live = gr.Button("Start", variant="primary")
        live_output = gr.HTML("<h2>Awaiting start...</h2>")
        
    with gr.Group(visible=False) as admin_group:
        btn_back_admin = gr.Button("Stop", variant="stop")
        admin_password = gr.Textbox(label="Password", type="password")
        btn_start_admin = gr.Button("Start", variant="primary")
        admin_output = gr.HTML("<h3>Awaiting start...</h3>")

    audio_in.start_recording(
        fn=lambda: gr.update(interactive=True),
        outputs=[btn_confirm_mic]
    )

    def confirm_mic():
        return gr.Accordion(open=False), gr.update(visible=False), gr.update(visible=True)

    btn_confirm_mic.click(
        fn=confirm_mic,
        outputs=[mic_group, btn_confirm_mic, menu_group]
    )

    def on_mic_stop(live_flag):
        live_flag[0] = False
        return gr.Accordion(open=True), gr.update(visible=True, interactive=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    audio_in.stop_recording(
        fn=on_mic_stop,
        inputs=[live_flag_state],
        outputs=[mic_group, btn_confirm_mic, menu_group, live_group, admin_group]
    )

    btn_menu_live.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[menu_group, live_group])
    btn_menu_admin.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[menu_group, admin_group])
    
    def nav_back(live_flag):
        live_flag[0] = False
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

    btn_back_live.click(nav_back, inputs=[live_flag_state], outputs=[menu_group, live_group, admin_group, btn_start_live, btn_start_admin])
    btn_back_admin.click(nav_back, inputs=[live_flag_state], outputs=[menu_group, live_group, admin_group, btn_start_live, btn_start_admin])
    
    audio_in.stream(
        fn=handle_audio_stream,
        inputs=[audio_in, audio_queue_state],
        concurrency_limit=100
    )
    
    btn_start_live.click(
        fn=live_mode_generator,
        inputs=[audio_queue_state, ui_queue_state, live_flag_state],
        outputs=[live_output, btn_start_live],
        concurrency_limit=100
    )
    
    btn_start_admin.click(
        fn=admin_mode_generator,
        inputs=[admin_password, audio_queue_state, ui_queue_state, live_flag_state],
        outputs=[admin_output, btn_start_admin],
        concurrency_limit=100
    )
    
    gr.HTML("""
    <div style="position: fixed; bottom: 15px; left: 0; right: 0; text-align: center; font-size: 14px; opacity: 0.5; display: flex; align-items: center; justify-content: center; gap: 8px; z-index: 100; pointer-events: none;">
        <svg height="18" width="18" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
        </svg>
        <a href="https://github.com/fkondela/keywordtensor" target="_blank" style="color: inherit; text-decoration: none; font-weight: 500; pointer-events: auto;">View on GitHub</a>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000, theme=gr.themes.Soft(primary_hue="blue"), css=custom_css)
