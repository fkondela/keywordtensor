import os
import time
import random
import queue
import threading
import itertools
import collections
import numpy as np
import gradio as gr
from faker import Faker
from keywordtensor.core import Engine

engine = Engine()
fake = Faker('pl_PL')

CUSTOM_WORDS = [
    "wrak", "hak", "smak", "znak", "mak", "rak", "brak", "szlak", "fakt",
    "krok", "skok", "blok", "smok", "sok", "bok", "rok", "szok", "tok",
    "nic", "nić", "nikt", "niech", "miecz", "sieć", "piec", "pień", "cień",
    "śni", "dni", "dno", "noc", "moc", "koc",
    "psy", "sny", "my", "wy", "ty", "by", "co", "po", "kto", "sto",
    "to", "do", "bo", "no", "go", "za", "na", "ma", "jak",
    "gdzie", "kiedy", "teraz", "zaraz", "potem", "tutaj", "tam", "stąd",
    "nos", "los", "włos", "głos", "byt", "mit", "hit", "szczyt",
    "dal", "bal", "stal", "szal", "pan", "stan", "plan",
    "kran", "dom", "tom", "prom", "grom", "kot", "lot",
    "las", "czas", "bas", "kwas", "głaz", "syn",
    "młyn", "płyn", "pech", "cud",
    "lud", "trud", "mróz", "wóz", "król", "sól", "ul"
]
_other_cycle = itertools.cycle(["bg", "fn", "bg", "hn"])
_current_word = ""

class AudioState:
    def __init__(self):
        self.sr = None
        self.buffer = []
        self.duration = 1.0
        self.lock = threading.Lock()

def consume_ui_events(ui_queue, thread, live_flag):
    error = False
    finished = False
    try:
        while live_flag[0]:
            try:
                msg = ui_queue.get(timeout=0.1)
                if msg == "DONE":
                    finished = True
                    break
                yield msg, gr.update(visible=False)
                if "ERROR" in msg:
                    error = True
                    live_flag[0] = False
                    break
            except queue.Empty:
                pass
    finally:
        cancelled = not live_flag[0] and not finished
        live_flag[0] = False
        thread.join(timeout=1.0)
        
    if not error:
        if cancelled:
            yield gr.update(), gr.update()
        else:
            yield "<h3>Finished safely.</h3>", gr.update(visible=True)

def handle_audio_stream(chunk, state):
    if chunk:
        sr, y = chunk
        with state.lock:
            if state.sr is None:
                state.sr = sr
                target = int(sr * state.duration)
                state.buffer = collections.deque([0.0] * target, maxlen=target)
                
            if y.dtype != np.float32:
                y = y.astype(np.float32) / 32768.0
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
                
            y_list = y.tolist()
            shift = len(y_list)
            
            if shift > 0:
                state.buffer.extend(y_list)

def live_mode(state, ui_queue, live_flag):
    live_flag[0] = True
    with state.lock:
        state.duration = 1.0
        state.buffer.clear()
        state.sr = None
    with ui_queue.mutex:
        ui_queue.queue.clear()
        
    def tak_cb():
        ui_queue.put("<h2>Detected: <span style='color:green'>TAK</span></h2>")
        time.sleep(1.0)
        
    def nie_cb():
        ui_queue.put("<h2>Detected: <span style='color:red'>NIE</span></h2>")
        time.sleep(1.0)
        
    def other_cb():
        ui_queue.put("<h2>Detected: <span style='color:gray'>OTHER</span></h2>")
        
    def worker():
        while state.sr is None and live_flag[0]:
            time.sleep(0.05)
        if not live_flag[0]: return
            
        try:
            engine.listen(
                "tak_nie", 
                actions={
                    "tak": tak_cb, 
                    "nie": nie_cb,
                    "other": other_cb
                }, 
                source=(state.sr, state.buffer),
                min_confidence=0.7,
                n_averages=1,
                listen_time=-1,
                stop=lambda: not live_flag[0]
            )
        except Exception as e:
            ui_queue.put(f"<h2>ERROR: {str(e)}</h2>")
            
    t = threading.Thread(target=worker)
    t.start()
    
    yield "<h2>Awaiting audio...</h2>", gr.update(visible=False)
    yield from consume_ui_events(ui_queue, t, live_flag)

def admin_mode(password, state, ui_queue, live_flag):
    global _current_word
    if password != os.environ.get("ADMIN_PASS", "dev123"):
        yield "<h3>Invalid Password!</h3>", gr.update(visible=True)
        return

    live_flag[0] = True
    with state.lock:
        state.duration = 1.0
        state.buffer.clear()
        state.sr = None
    with ui_queue.mutex:
        ui_queue.queue.clear()

    def get_other_prompt():
        global _current_word
        kind = next(_other_cycle)
        if kind == "bg":
            _current_word = "background"
            return None
        elif kind == "fn":
            _current_word = fake.word().lower()
            return _current_word
        else:
            _current_word = random.choice(CUSTOM_WORDS)
            return _current_word

    def create_action(cls_name):
        def action(start_recording, current_time, total_time):
            word = get_other_prompt() if cls_name == "other" else cls_name
            
            if not word:
                word_display = "<span style='color:gray'>🔇 NIC NIE MÓW (Cisza)</span>"
            elif word.lower() == "tak":
                word_display = "<span style='color:#22c55e'>TAK</span>"
            elif word.lower() == "nie":
                word_display = "<span style='color:#ef4444'>NIE</span>"
            else:
                word_display = f"<span style='color:#a855f7'>{word.upper()}</span>"

            for i in [3, 2, 1]:
                if not live_flag[0]: return
                ui_queue.put(f"<h3>Recording <b>{word_display}</b> - get ready <span style='color:#3b82f6'>{i}</span>...</h3>")
                time.sleep(0.5)

            start_recording()

            while current_time() < total_time:
                if not live_flag[0]: return
                ui_queue.put(f"<h3>Recording <b>{word_display}</b> (<span style='color:#f97316'>{current_time():.1f}s</span> / {total_time:.1f}s)</h3>")
                time.sleep(0.1)

            ui_queue.put("<h3><span style='color:#22c55e'>Done, sending to server...</span></h3>")
        return action

    def save_and_upload(cls_name, idx, audio_np, sr):
        import soundfile as sf
        label = _current_word if cls_name == "other" else cls_name
        filename = f"{label}_{int(time.time())}_{random.randint(1000, 9999)}_{idx}.wav"
        tmp = f"/tmp/{filename}"
        sf.write(tmp, audio_np.squeeze(), sr)
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                ui_queue.put("<h3>ERROR: Missing HF_TOKEN!</h3>")
                return
            from huggingface_hub import HfApi
            HfApi(token=hf_token).upload_file(
                path_or_fileobj=tmp,
                path_in_repo=f"users_dataset/{cls_name}/{filename}",
                repo_id="fkondela/KeywordTensor_tak_nie",
                repo_type="dataset",
                commit_message=f"Add {cls_name} sample"
            )
            ui_queue.put(f"<h3>Uploaded: {cls_name}/{filename}</h3>")
        except Exception as e:
            ui_queue.put(f"<h3>ERROR: {e}</h3>")
        finally:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass

    def worker():
        while state.sr is None and live_flag[0]:
            time.sleep(0.05)
        if not live_flag[0]: return
        try:
            engine.record(
                target=save_and_upload,
                classes=["tak", "nie", "other"],
                samples=2,
                actions={
                    "tak": create_action("tak"),
                    "nie":  create_action("nie"),
                    "other":  create_action("other"),
                },
                source=(state.sr, state.buffer),
                duration=1.0,
                stop=lambda: not live_flag[0]
            )
            ui_queue.put("DONE")
        except Exception as e:
            ui_queue.put(f"<h3>ERROR: {e}</h3>")
            ui_queue.put("DONE")

    t = threading.Thread(target=worker)
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
footer { display: none !important; }
button[aria-label="Settings"] { display: none !important; }
"""

head_html = '<link rel="icon" type="image/png" href="/favicon.ico?v=999">'
with gr.Blocks(title="KeywordTensor") as demo:
    gr.HTML('''
    <div class="header-container">
        <img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.png?v=999" alt="KeywordTensor Logo">
        <h1>KeywordTensor - tak_nie model</h1>
    </div>
    ''')
    
    audio_state = gr.State(lambda: AudioState())
    ui_queue = gr.State(lambda: queue.Queue())
    live_flag = gr.State(lambda: [False])
    
    with gr.Accordion("Step 1: Select Microphone", open=True) as mic_group:
        audio_in = gr.Audio(sources=["microphone"], streaming=True, label="Audio Stream")
        btn_confirm = gr.Button("Next", variant="primary", interactive=False)
        
    with gr.Group(visible=False) as menu_group:
        gr.Markdown("### Select Mode:")
        with gr.Row():
            btn_live = gr.Button("Live Mode", variant="primary")
            btn_admin = gr.Button("Admin Panel", variant="secondary")
            
    with gr.Group(visible=False) as live_group:
        btn_stop_live = gr.Button("Stop", variant="stop")
        btn_start_live = gr.Button("Start", variant="primary")
        live_output = gr.HTML("<h2>Awaiting start...</h2>")
        
    with gr.Group(visible=False) as admin_group:
        btn_stop_admin = gr.Button("Stop", variant="stop")
        admin_pass = gr.Textbox(label="Password", type="password")
        btn_start_admin = gr.Button("Start", variant="primary")
        admin_output = gr.HTML("<h3>Awaiting start...</h3>")

    audio_in.start_recording(fn=lambda: gr.update(interactive=True), outputs=[btn_confirm])
    btn_confirm.click(fn=lambda: (gr.Accordion(open=False), gr.update(visible=False), gr.update(visible=True)), outputs=[mic_group, btn_confirm, menu_group])

    def on_stop(flag):
        flag[0] = False
        return gr.Accordion(open=True), gr.update(visible=True, interactive=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    audio_in.stop_recording(fn=on_stop, inputs=[live_flag], outputs=[mic_group, btn_confirm, menu_group, live_group, admin_group])

    btn_live.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)), outputs=[menu_group, live_group, btn_start_live])
    btn_admin.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)), outputs=[menu_group, admin_group, btn_start_admin])
    
    def nav_back(flag):
        flag[0] = False
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

    btn_stop_live.click(nav_back, inputs=[live_flag], outputs=[menu_group, live_group, admin_group, btn_start_live, btn_start_admin])
    btn_stop_admin.click(nav_back, inputs=[live_flag], outputs=[menu_group, live_group, admin_group, btn_start_live, btn_start_admin])
    
    audio_in.stream(fn=handle_audio_stream, inputs=[audio_in, audio_state], concurrency_limit=100)
    btn_start_live.click(fn=live_mode, inputs=[audio_state, ui_queue, live_flag], outputs=[live_output, btn_start_live], concurrency_limit=100)
    btn_start_admin.click(fn=admin_mode, inputs=[admin_pass, audio_state, ui_queue, live_flag], outputs=[admin_output, btn_start_admin], concurrency_limit=100)
    
    gr.HTML("""
    <div style="position: fixed; bottom: 15px; left: 0; right: 0; text-align: center; font-size: 14px; opacity: 0.5; display: flex; align-items: center; justify-content: center; gap: 8px; z-index: 100; pointer-events: none;">
        <svg height="18" width="18" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
        <a href="https://github.com/fkondela/keywordtensor" target="_blank" style="color: inherit; text-decoration: none; font-weight: 500; pointer-events: auto;">View on GitHub</a>
    </div>
    """)

if __name__ == "__main__":
    import os
    # Jeśli app.py na Azure jest w roocie obok assets, bierzemy assets stąd. W środowisku lokalnym jest poziom wyżej.
    base = os.path.dirname(__file__)
    favicon_path = os.path.join(base if os.path.exists(os.path.join(base, "assets")) else os.path.dirname(base), "assets", "logo.png")
    
    demo.launch(server_name="0.0.0.0", server_port=8000, theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, favicon_path=favicon_path, head=head_html)
