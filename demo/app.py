import os
import time
import random
import queue
import threading
import numpy as np
import gradio as gr
from faker import Faker
from keywordtensor.core import Engine

engine = Engine()
fake = Faker('pl_PL')

class AudioState:
    def __init__(self):
        self.sr = None
        self.buffer = []
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
        if state.sr is None:
            state.sr = sr
            state.buffer = [0.0] * int(sr * 3.0)
            
        if y.dtype != np.float32:
            y = y.astype(np.float32) / 32768.0
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        y_list = y.tolist()
        shift = len(y_list)
        target = int(state.sr * 3.0)
        
        with state.lock:
            if shift >= target:
                state.buffer[:] = y_list[-target:]
            else:
                state.buffer[:-shift] = state.buffer[shift:]
                state.buffer[-shift:] = y_list

def live_mode(state, ui_queue, live_flag):
    live_flag[0] = True
    with state.lock:
        state.buffer.clear()
        state.sr = None
    with ui_queue.mutex:
        ui_queue.queue.clear()
        
    def truth_cb():
        ui_queue.put("<h2>Detected: <span style='color:green'>PRAWDA</span></h2>")
        time.sleep(3.0)
        
    def false_cb():
        ui_queue.put("<h2>Detected: <span style='color:red'>FAŁSZ</span></h2>")
        time.sleep(3.0)
        
    def other_cb():
        ui_queue.put("<h2>Detected: <span style='color:gray'>OTHER</span></h2>")
        
    def worker():
        while state.sr is None and live_flag[0]:
            time.sleep(0.05)
        if not live_flag[0]: return
            
        try:
            engine.listen(
                "prawda_falsz", 
                actions={
                    "prawda": truth_cb, 
                    "falsz": false_cb,
                    "other": other_cb
                }, 
                source=(state.sr, state.buffer),
                min_confidence=0.55,
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
    if password != os.environ.get("ADMIN_PASS", "dev123"):
        yield "<h3>Invalid Password!</h3>", gr.update(visible=True)
        return
        
    live_flag[0] = True
    with state.lock:
        state.buffer.clear()
        state.sr = None
    with ui_queue.mutex:
        ui_queue.queue.clear()
        
    BANNED_WORDS = ["prawda", "fałsz", "falsz", "prawdę", "prawde"]
    CUSTOM_WORDS = ["prawie", "prawo", "prawnik", "sprawdzam", "sprawa", "fauna", "fala", "szum", "wdowa", "owca"]
    
    def safe_word():
        if random.random() < 0.35:
            return random.choice(CUSTOM_WORDS)
        while True:
            word = fake.word().lower()
            if word not in BANNED_WORDS:
                return word
                
    def get_times(count):
        if count == 0: return []
        while True:
            times = [random.uniform(0.1, 1.8) for _ in range(count)]
            times.sort()
            if count == 1: return times
            if all(times[i] - times[i-1] >= 0.65 for i in range(1, count)):
                return times

    def build_timeline(cls_name):
        items = []
        count = random.choice([2, 3]) if cls_name != "other" else random.choice([1, 2, 3])
        times = get_times(count)
        for t in times:
            items.append({"start": t, "text": safe_word(), "read": False, "target": False})
            
        if cls_name in ["prawda", "falsz"] and count > 0:
            idx = random.randint(0, count - 1)
            items[idx]["text"] = "PRAWDA" if cls_name == "prawda" else "FAŁSZ"
            items[idx]["target"] = True
        return items

    def create_action(cls_name):
        def action(start_recording, current_time, total_time):
            timeline = build_timeline(cls_name)
            plan_str = " | ".join([f"**[{z['start']:.1f}s]** {z['text']}" for z in timeline])
            ui_queue.put(f"<h3>Plan (5s):</h3><p>{plan_str}</p>")
            
            for _ in range(50):
                if not live_flag[0]: return
                time.sleep(0.1)
            
            for i in [3, 2, 1]:
                if not live_flag[0]: return
                ui_queue.put(f"<h3>Start in {i}...</h3>")
                time.sleep(1.0)
                
            start_recording()
            
            while (t := current_time()) < total_time:
                if not live_flag[0]: return
                html_out = ""
                for z in timeline:
                    if not z["read"] and t >= z["start"]:
                        if z["target"]:
                            html_out = f"<h2>SPEAK: <span style='color:red'>{z['text']}</span></h2>"
                        else:
                            html_out = f"<h2>SPEAK: {z['text']}</h2>"
                        z["read"] = True
                if html_out:
                    ui_queue.put(html_out)
                time.sleep(0.05)
                
            ui_queue.put("<h3>Recording finished! Uploading...</h3>")
        return action

    def save_and_upload(cls_name, idx, tensor_data, sr):
        import wave
        import numpy as np
        audio_np = tensor_data.squeeze()
        audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
        
        temp_file = f"/tmp/temp_{int(time.time())}_{random.randint(10000, 99999)}_{idx}.wav"
        
        with wave.open(temp_file, "w") as f:
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
            base_name = f"{cls_name}/probka_{int(time.time())}_{random.randint(1000, 9999)}_{idx}"
            api.upload_file(
                path_or_fileobj=temp_file,
                path_in_repo=f"{base_name}.wav",
                repo_id="fkondela/KeywordTensor_prawda_falsz", 
                repo_type="dataset"
            )
            ui_queue.put(f"<h3>Successfully uploaded: {cls_name}</h3>")
        except Exception as e:
            ui_queue.put(f"<h3>ERROR: {str(e)}</h3>")
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    def worker():
        while state.sr is None and live_flag[0]:
            time.sleep(0.05)
        if not live_flag[0]: return
        
        try:
            engine.record(
                target=save_and_upload,
                classes=["prawda", "falsz"],
                samples=2,
                actions={"prawda": create_action("prawda"), "falsz": create_action("falsz")},
                source=(state.sr, state.buffer),
                duration=3.0,
                stop=lambda: not live_flag[0]
            )
            ui_queue.put("DONE")
        except Exception as e:
            ui_queue.put(f"<h3>ERROR: {str(e)}</h3>")
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

with gr.Blocks(title="KeywordTensor") as demo:
    gr.HTML('''
    <div class="header-container">
        <img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.png" alt="KeywordTensor Logo">
        <h1>KeywordTensor - prawda_falsz model</h1>
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
    import urllib.request
    import tempfile
    
    favicon = os.path.join(tempfile.gettempdir(), "keywordtensor_logo.png")
    if not os.path.exists(favicon):
        try:
            urllib.request.urlretrieve("https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.png", favicon)
        except Exception:
            favicon = None
            
    demo.launch(server_name="0.0.0.0", server_port=8000, theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, favicon_path=favicon)
