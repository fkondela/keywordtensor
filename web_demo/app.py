import os
import urllib.request
from pathlib import Path
import io
import time
import subprocess
import tempfile
import random
import threading
import itertools
import collections
import numpy as np
import gradio as gr
import requests
import html
import uuid
import base64
from gtts import gTTS
import soundfile as sf
from huggingface_hub import HfApi
from faker import Faker
from keywordtensor.core import Engine

def speak_sync(text):
    try:
        tts = gTTS(text=text, lang='pl')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio = sf.info(fp)
        duration = audio.duration
        
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode('utf-8')
        return b64, duration
    except Exception as e:
        print(f"TTS Error: {e}")
        return "", max(1.0, len(text) * 0.08)


quiz_questions = [
    ("Czy gepardy potrafią biegać szybciej niż 100 kilometrów na godzinę?", "tak"),
    ("Czy rekiny mają kości?", "nie"),
    ("Czy Wielki Mur Chiński widać z kosmosu gołym okiem?", "nie"),
    ("Czy na Wenus dzień trwa dłużej niż tamtejszy rok?", "tak"),
    ("Czy Mount Everest to jedyna góra na Ziemi o wysokości ponad 8 tysięcy metrów?", "nie"),
    ("Czy serce płetwala błękitnego jest wielkości małego samochodu?", "tak"),
    ("Czy Antarktyda to największa pustynia na naszej planecie?", "tak"),
    ("Czy nietoperze są całkowicie ślepe?", "nie"),
    ("Czy Jowisz ma twardą powierzchnię, na której można wylądować?", "nie"),
    ("Czy Słońce jest gwiazdą?", "tak"),
    ("Czy orki należą do rodziny delfinów?", "tak"),
    ("Czy na Księżycu jest wiatr?", "nie"),
    ("Czy kameleon zmienia kolor tylko po to, żeby wtopić się w otoczenie?", "nie"),
    ("Czy woda na Ziemi jest starsza niż nasze Słońce?", "tak"),
    ("Czy pajęczyna jest mocniejsza od drutu stalowego o tej samej grubości?", "tak"),
    ("Czy niedźwiedzie polarne mają białą skórę pod futrem?", "nie"),
    ("Czy piorun jest gorętszy od powierzchni Słońca?", "tak"),
    ("Czy ośmiornice mają trzy serca?", "tak"),
    ("Czy Wenus jest najbliższą Słońca planetą w naszym układzie?", "nie"),
    ("Czy diamenty można stopić w bardzo wysokiej temperaturze?", "tak")
]

def create_game_iframe(game_id, url, base_url, bg_color):
    raw_html = requests.get(url).text.replace("<head>", f'<head>\n    <base href="{base_url}">')
    style = f"<style>body {{ display: flex; justify-content: center; align-items: center; margin: 0; padding: 0; background-color: {bg_color};}} main, canvas, .container {{ margin-top: 20px !important; box-shadow: 0px 0px 20px rgba(0,0,0,0.5); border-radius: 8px; padding: 20px; background: white; }}</style>"
    game_html = raw_html.replace("</head>", f"{style}\n</head>")
    return f'<iframe id="{game_id}" srcdoc="{html.escape(game_html)}" width="100%" height="800" style="border:none; overflow:hidden;"></iframe>'

iframe_soko = create_game_iframe("game_iframe_sokoban", "https://raw.githubusercontent.com/taniarascia/sokoban/master/index.html", "https://rawcdn.githack.com/taniarascia/sokoban/master/", "#2c3e50")
iframe_2048 = create_game_iframe("game_iframe_2048", "https://raw.githubusercontent.com/gabrielecirulli/2048/master/index.html", "https://rawcdn.githack.com/gabrielecirulli/2048/master/", "#faf8ef")

custom_css = """
.header-container { display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 20px; margin-top: 10px; }
.header-container img { height: 50px; border-radius: 12px; filter: drop-shadow(0px 2px 4px rgba(0, 0, 0, 0.2)); }
.header-container h1 { margin: 0; color: #3b82f6 !important; font-weight: bold !important; font-size: 2.2em; }
footer { display: none !important; }
button[aria-label="Settings"] { display: none !important; }
"""

def handle_audio_stream(chunk, state):
    if chunk:
        sr, y = chunk
        if state["sr"] is None:
            state["sr"] = sr
            target = int(sr * state["duration"])
            state["buffer"] = collections.deque([0.0] * target, maxlen=target)
        if y.dtype != np.float32: y = y.astype(np.float32) / 32768.0
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        if len(y_list := y.tolist()) > 0: state["buffer"].extend(y_list)

def run_game(state, live_flag, iframe_code):
    live_flag[0] = True
    state["duration"], state["sr"] = 1.0, None
    target_len = state["buffer"].maxlen
    if target_len: state["buffer"].extend([0.0] * target_len)
    
    last_action = [None]
    def trigger(direction): last_action[0] = direction; time.sleep(1.0)
    
    def worker():
        while state["sr"] is None and live_flag[0]: time.sleep(0.05)
        if not live_flag[0]: return
        try:
            state["engine"].listen(
                "spatial_nav",
                actions={"up": lambda: trigger("up"), "down": lambda: trigger("down"), "left": lambda: trigger("left"), "right": lambda: trigger("right")},
                min_confidence=0.7, n_averages=1, listen_time=-1,
                source=(state["sr"], state["buffer"]), threads=1, stop=lambda: not live_flag[0]
            )
        except: pass

    threading.Thread(target=worker, daemon=True).start()

    iframe_div = f"<div style='display:flex; justify-content:center;'>{iframe_code}</div>"
    yield gr.update(visible=False), iframe_div, ""
    
    while live_flag[0]:
        if last_action[0]:
            msg = last_action[0]
            last_action[0] = None
            yield gr.update(visible=False), iframe_div, f"{msg}_{uuid.uuid4()}"
        else:
            time.sleep(0.1)
            
    yield gr.update(), gr.update(), ""

def get_js_handler(iframe_id, use_which_prop=False):
    key_prop = "which" if use_which_prop else "key"
    key_val = "key" if use_which_prop else """(key===38) ? 'ArrowUp' : (key===40) ? 'ArrowDown' : (key===37) ? 'ArrowLeft' : 'ArrowRight'"""
    
    return f"""
    (word) => {{ 
        if (!word) return;
        let key = 0;
        if (word.startsWith('up')) key = 38;
        if (word.startsWith('down')) key = 40;
        if (word.startsWith('left')) key = 37;
        if (word.startsWith('right')) key = 39;
        
        if (key !== 0) {{
            let iframe = document.getElementById("{iframe_id}");
            if (iframe && iframe.contentWindow) {{
                let e_down = new KeyboardEvent("keydown", {{bubbles: true}});
                Object.defineProperty(e_down, 'keyCode', {{get: () => key}});
                Object.defineProperty(e_down, '{key_prop}', {{get: () => {key_val}}});
                iframe.contentWindow.document.dispatchEvent(e_down);
                setTimeout(() => {{
                    let e_up = new KeyboardEvent("keyup", {{bubbles: true}});
                    Object.defineProperty(e_up, 'keyCode', {{get: () => key}});
                    Object.defineProperty(e_up, '{key_prop}', {{get: () => {key_val}}});
                    iframe.contentWindow.document.dispatchEvent(e_up);
                }}, 50);
            }}
        }}
    }}
    """

js_handler_soko = get_js_handler("game_iframe_sokoban", use_which_prop=False)
js_handler_2048 = get_js_handler("game_iframe_2048", use_which_prop=True)



def live_mode_quiz(state, live_flag):
    live_flag[0] = True
    state["duration"], state["sr"] = 1.0, None
    target_len = state["buffer"].maxlen
    if target_len: state["buffer"].extend([0.0] * target_len)
    
    last_html = [""]
    def on_tak(): last_html[0] = "<h2>Detected: <span style='color:green'>TAK</span></h2>"; time.sleep(1.0)
    def on_nie(): last_html[0] = "<h2>Detected: <span style='color:red'>NIE</span></h2>"; time.sleep(1.0)
    def on_other(): last_html[0] = "<h2>Detected: <span style='color:gray'>OTHER</span></h2>"

    def worker():
        while state["sr"] is None and live_flag[0]: time.sleep(0.05)
        if not live_flag[0]: return
        try:
            state["engine"].listen(
                "tak_nie", actions={"tak": on_tak, "nie": on_nie, "other": on_other},
                min_confidence=0.6, n_averages=1, listen_time=-1,
                source=(state["sr"], state["buffer"]), threads=1, stop=lambda: not live_flag[0]
            )
        except: pass

    threading.Thread(target=worker, daemon=True).start()

    yield "<h2>Awaiting audio...</h2>", gr.update(visible=False)
    while live_flag[0]:
        if last_html[0]:
            html_msg = last_html[0]
            last_html[0] = ""
            yield html_msg, gr.update(visible=False)
        time.sleep(0.1)
    yield "<h2>Finished.</h2>", gr.update(visible=True)

def admin_mode_quiz(password, state, live_flag):
    if password != os.environ.get("ADMIN_PASS", "dev123"):
        yield "<h2>Invalid Password!</h2>", gr.update(visible=True)
        return

    live_flag[0] = True
    state["duration"], state["sr"] = 1.0, None
    target_len = state["buffer"].maxlen
    if target_len: state["buffer"].extend([0.0] * target_len)
    
    current_status = ["<h2>Starting...</h2>"]
    done_flag = [False]

    def get_other_prompt():
        kind = next(state["other_cycle"])
        if kind == "bg": state["current_word"] = "background"; return None
        if kind == "fn": state["current_word"] = fake.word().lower(); return state["current_word"]
        state["current_word"] = random.choice(CUSTOM_WORDS); return state["current_word"]

    def create_action(cls_name):
        def action(start_recording, current_time, total_time):
            word = get_other_prompt() if cls_name == "other" else cls_name
            if not word: word_display = "<span style='color:gray'>🔇 SILENCE</span>"
            elif word.lower() == "tak": word_display = "<span style='color:#22c55e'>TAK</span>"
            elif word.lower() == "nie": word_display = "<span style='color:#ef4444'>NIE</span>"
            else: word_display = f"<span style='color:#a855f7'>{word.upper()}</span>"

            for i in [3, 2, 1]:
                if not live_flag[0]: return
                current_status[0] = f"<h2>Recording <b>{word_display}</b> - get ready <span style='color:#3b82f6'>{i}</span>...</h2>"
                time.sleep(1.0)

            start_recording()
            while current_time() < total_time:
                if not live_flag[0]: return
                current_status[0] = f"<h2>Recording <b>{word_display}</b> (<span style='color:#f97316'>{current_time():.1f}s</span> / {total_time:.1f}s)</h2>"
                time.sleep(0.25)

            current_status[0] = "<h2><span style='color:#22c55e'>Done, sending to server...</span></h2>"
        return action

    def save_and_upload(cls_name, idx, audio_np, sr):
        if not live_flag[0]: return
        label = state["current_word"] if cls_name == "other" else cls_name
        filename = f"{label}_{int(time.time())}_{random.randint(1000, 9999)}_{idx}.wav"
        tmp = f"/tmp/{filename}"
        sf.write(tmp, audio_np.squeeze(), sr)
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token: current_status[0] = "<h2>ERROR: Missing HF_TOKEN!</h2>"; return
            HfApi(token=hf_token).upload_file(
                path_or_fileobj=tmp, path_in_repo=f"users_dataset/{cls_name}/{filename}",
                repo_id="fkondela/KeywordTensor_tak_nie", repo_type="dataset", commit_message=f"Add {cls_name} sample"
            )
            current_status[0] = f"<h2>Uploaded: {cls_name}/{filename}</h2>"
        except Exception as e: current_status[0] = f"<h2>ERROR: {e}</h2>"
        finally:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass

    def worker():
        while state["sr"] is None and live_flag[0]: time.sleep(0.05)
        if not live_flag[0]: return
        try:
            state["engine"].record(
                target=save_and_upload, classes=["tak", "nie", "other"], samples=4,
                actions={"tak": create_action("tak"), "nie": create_action("nie"), "other": create_action("other")},
                source=(state["sr"], state["buffer"]), duration=1.0, threads=1, stop=lambda: not live_flag[0]
            )
        except Exception as e:
            current_status[0] = f"<h2>ERROR: {str(e)}</h2>"
        finally:
            done_flag[0] = True
            
    threading.Thread(target=worker, daemon=True).start()
    
    yield current_status[0], gr.update(visible=False)
    last_yielded = current_status[0]
    while live_flag[0] and not done_flag[0]:
        if current_status[0] != last_yielded:
            last_yielded = current_status[0]
            yield last_yielded, gr.update(visible=False)
        time.sleep(0.25)
    yield "<h2>Finished.</h2>", gr.update(visible=True)

def game_mode_quiz(state, live_flag):
    live_flag[0] = True
    state["duration"], state["sr"] = 1.0, None
    target_len = state["buffer"].maxlen
    if target_len: state["buffer"].extend([0.0] * target_len)
    
    yield "<h2>Awaiting start...</h2>", gr.update(visible=False), None, None
    
    score = 0
    total = min(10, len(quiz_questions))
    selected_questions = random.sample(quiz_questions, total)
    
    for idx, (question_text, correct_answer) in enumerate(selected_questions):
        if not live_flag[0]: break
        
        b64, duration = speak_sync(question_text)
        
        def render_ui(time_left_str, feedback_html=""):
            html = f"<div style='background-color:#f0f9ff; padding:20px; border-radius:12px; border:2px solid #bae6fd; text-align:center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>"
            html += f"<h2 style='margin-bottom:10px; color:#1e293b;'>Score: <span style='color:#8b5cf6'>{score}/{total}</span> &nbsp;&nbsp;|&nbsp;&nbsp; Time Left: <span id='qt' style='color:#3b82f6'>{time_left_str}</span></h2>"
            html += f"<h3 style='color:#64748b; margin-bottom:5px;'>Question {idx+1}/{total}</h3>"
            html += f"<h2 style='color:#1e3a8a; font-size:1.8em; margin-bottom:20px;'>{question_text}</h2>"
            if feedback_html: html += f"<div style='margin-top:20px;'>{feedback_html}</div>"
            html += "</div>"
            return html
            
        yield render_ui("Speaking..."), gr.update(visible=False), b64, "stop"
        
        time_start_sleep = time.time()
        while time.time() - time_start_sleep < (duration + 0.1) and live_flag[0]:
            time.sleep(0.1)
            
        if target_len: state["buffer"].extend([0.0] * target_len)
        
        answer_detected = [None]
        def on_tak(): answer_detected[0] = "tak"
        def on_nie(): answer_detected[0] = "nie"
        def on_other(): pass
        
        def worker():
            while state["sr"] is None and live_flag[0]: time.sleep(0.05)
            if not live_flag[0]: return
            try:
                state["engine"].listen(
                    "tak_nie", actions={"tak": on_tak, "nie": on_nie, "other": on_other}, 
                    source=(state["sr"], state["buffer"]), min_confidence=0.6, n_averages=1, listen_time=10.0, threads=1, stop=lambda: not live_flag[0] or answer_detected[0] is not None
                )
            except: pass
        
        threading.Thread(target=worker, daemon=True).start()
        
        start_t = time.time()
        yield render_ui("10.0s"), gr.update(visible=False), "", "10.0"
        
        while time.time() - start_t < 10.0 and live_flag[0] and answer_detected[0] is None:
            time.sleep(0.1)
            
        if not live_flag[0]: break
        
        if answer_detected[0] == correct_answer:
            score += 1
            feedback = "<h2><span style='color:green'>Correct!</span></h2>"
            text_to_speak = "Brawo, to poprawna odpowiedź!"
        elif answer_detected[0] is None:
            feedback = "<h2><span style='color:gray'>Time's up!</span></h2>"
            text_to_speak = "Czas minął!"
        else:
            feedback = "<h2><span style='color:red'>Wrong!</span></h2>"
            text_to_speak = "Niestety, zła odpowiedź."
            
        # 1. BŁYSKAWICZNA REAKCJA UI (Zatrzymanie timera i pokazanie wyniku bez czekania na audio)
        yield render_ui("0.0s", feedback_html=feedback), gr.update(visible=False), "", "stop"
        
        # Wymuszamy oddanie kontroli (GIL release), aby serwer Gradio zdążył fizycznie wysłać powyższą klatkę przez sieć zanim zablokujemy go na 2 sekundy pobieraniem gTTS!
        time.sleep(0.05)
        
        # 2. GENEROWANIE AUDIO W TLE (UI już się odświeżyło)
        b64_f, d = speak_sync(text_to_speak)
        
        # 3. WYSŁANIE GOTOWEGO DŹWIĘKU
        yield render_ui("0.0s", feedback_html=feedback), gr.update(visible=False), b64_f, "stop"
        
        time_start_sleep = time.time()
        while time.time() - time_start_sleep < (d + 0.5) and live_flag[0]:
            time.sleep(0.1)
            
    final_html = f"<div style='background-color:#f0f9ff; padding:20px; border-radius:12px; border:2px solid #bae6fd; text-align:center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>"
    final_html += f"<h2 style='color:#1e3a8a; font-size:2.5em; margin-bottom:15px;'>Thanks for playing! 🎉</h2>"
    final_html += f"<h3 style='color:#64748b; font-size:1.5em;'>Your Score: <span style='color:#8b5cf6; font-weight:bold;'>{score}/{total}</span></h3>"
    b64_final, d = speak_sync(f"Koniec gry! Twój wynik to {score} punktów.")
    final_html += "</div>"
    
    yield final_html, gr.update(visible=True), b64_final, "stop"

with gr.Blocks(title="KeywordTensor") as demo:
    gr.HTML('''
    <div class="header-container">
        <img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.png?v=999" alt="KeywordTensor Logo">
        <h1>KeywordTensor</h1>
    </div>
    ''')
    
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

    def init_user_state():
        return {
            "sr": None, 
            "buffer": collections.deque(), 
            "duration": 1.0,
            "engine": Engine(),
            "other_cycle": itertools.cycle(["bg", "fn", "bg", "hn"]),
            "current_word": ""
        }

    state = gr.State(init_user_state)
    live_flag = gr.State(lambda: [False])
    
    with gr.Accordion("Step 1: Select Microphone", open=True) as mic_group:
        audio_in = gr.Audio(sources=["microphone"], streaming=True, label="Audio Stream")
        btn_confirm = gr.Button("Next", variant="primary", interactive=False)
        
    with gr.Group(visible=False) as menu_group_main:
        gr.Markdown("### Select App:")
        with gr.Row():
            btn_quiz_menu = gr.Button("Quiz", variant="primary")
            btn_2048 = gr.Button("2048", variant="primary")
            btn_sokoban = gr.Button("Sokoban", variant="primary")

    with gr.Group(visible=False) as menu_group_quiz:
        gr.Markdown("### Select Mode:")
        btn_quiz_back = gr.Button("Back to Menu", variant="stop")
        with gr.Row():
            btn_quiz_game = gr.Button("Play Game", variant="primary")
            btn_quiz_live = gr.Button("Live Mode", variant="primary")
            btn_quiz_admin = gr.Button("Admin Panel", variant="primary")

    with gr.Group(visible=False) as soko_group:
        gr.Markdown("### Voice-Controlled Sokoban (Puzzle Game)")
        btn_stop_soko = gr.Button("Back to Menu", variant="stop")
        btn_start_soko = gr.Button("START GAME", variant="primary")
        soko_html = gr.HTML("<h2>Awaiting start...</h2>")
        soko_out = gr.Textbox(visible=False)

    with gr.Group(visible=False) as g2048_group:
        gr.Markdown("### Voice-Controlled 2048")
        btn_stop_2048 = gr.Button("Back to Menu", variant="stop")
        btn_start_2048 = gr.Button("START GAME", variant="primary")
        g2048_html = gr.HTML("<h2>Awaiting start...</h2>")
        g2048_out = gr.Textbox(visible=False)

    with gr.Group(visible=False) as quiz_live_group:
        btn_stop_quiz_live = gr.Button("Back to Menu", variant="stop")
        btn_start_quiz_live = gr.Button("Start", variant="primary")
        quiz_live_html = gr.HTML("<h2>Awaiting start...</h2>")

    with gr.Group(visible=False) as quiz_admin_group:
        btn_stop_admin = gr.Button("Back to Menu", variant="stop")
        admin_pass = gr.Textbox(label="Password", type="password")
        btn_start_admin = gr.Button("Start", variant="primary")
        admin_html = gr.HTML("<h2>Awaiting start...</h2>")
        
    with gr.Group(visible=False) as quiz_game_group:
        btn_stop_quiz_game = gr.Button("Back to Menu", variant="stop")
        btn_start_quiz_game = gr.Button("Start", variant="primary")
        quiz_game_html = gr.HTML("<h2>Awaiting start...</h2>")
        quiz_game_tts_out = gr.Textbox(visible=False)
        quiz_game_timer_out = gr.Textbox(visible=False)

    audio_in.start_recording(fn=lambda: gr.update(interactive=True), outputs=[btn_confirm])
    audio_in.stream(handle_audio_stream, inputs=[audio_in, state], outputs=None, concurrency_limit=100)
    
    btn_confirm.click(fn=lambda: (gr.Accordion(open=False), gr.update(visible=False), gr.update(visible=True)), outputs=[mic_group, btn_confirm, menu_group_main])

    audio_in.stop_recording(
        fn=lambda flag: (flag.__setitem__(0, False), gr.Accordion(open=True), gr.update(visible=True, interactive=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
        inputs=[live_flag],
        outputs=[mic_group, btn_confirm, menu_group_main, menu_group_quiz, soko_group, g2048_group, quiz_live_group, quiz_admin_group, quiz_game_group]
    )

    btn_sokoban.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)), outputs=[menu_group_main, soko_group, btn_start_soko])
    btn_2048.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)), outputs=[menu_group_main, g2048_group, btn_start_2048])
    btn_quiz_menu.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[menu_group_main, menu_group_quiz])
    
    btn_quiz_back.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[menu_group_quiz, menu_group_main])
    btn_quiz_live.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)), outputs=[menu_group_quiz, quiz_live_group, btn_start_quiz_live])
    btn_quiz_admin.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)), outputs=[menu_group_quiz, quiz_admin_group, btn_start_admin])
    btn_quiz_game.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)), outputs=[menu_group_quiz, quiz_game_group, btn_start_quiz_game])

    def run_sokoban_game(state, live_flag): yield from run_game(state, live_flag, iframe_soko)
    def run_2048_game(state, live_flag): yield from run_game(state, live_flag, iframe_2048)

    btn_start_soko.click(run_sokoban_game, inputs=[state, live_flag], outputs=[btn_start_soko, soko_html, soko_out], concurrency_limit=100)
    soko_out.change(None, inputs=[soko_out], js=js_handler_soko)

    btn_start_2048.click(run_2048_game, inputs=[state, live_flag], outputs=[btn_start_2048, g2048_html, g2048_out], concurrency_limit=100)
    g2048_out.change(None, inputs=[g2048_out], js=js_handler_2048)

    btn_start_quiz_live.click(live_mode_quiz, inputs=[state, live_flag], outputs=[quiz_live_html, btn_start_quiz_live], concurrency_limit=100)
    btn_start_admin.click(admin_mode_quiz, inputs=[admin_pass, state, live_flag], outputs=[admin_html, btn_start_admin], concurrency_limit=100)
    
    btn_start_quiz_game.click(game_mode_quiz, inputs=[state, live_flag], outputs=[quiz_game_html, btn_start_quiz_game, quiz_game_tts_out, quiz_game_timer_out], concurrency_limit=100, js="() => { let a=document.getElementById('ap')||document.createElement('audio'); a.id='ap'; a.style.display='none'; document.body.appendChild(a); a.src='data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4LjEyLjEwMAAAAAAAAAAAAAAA//OEAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAEAAABIAD+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+AAAAAExhdmM1OC4xMzQAAAAAAAAAAAAAAAAkAEQAAAAAAAASIQAAAABJRU5E'; a.play().catch(()=>{}); return []; }")
    
    js_play = "(b) => { if(b){ let a=document.getElementById('ap'); if(a){ a.src='data:audio/mp3;base64,'+b; a.play().catch(console.error); } } }"
    quiz_game_tts_out.change(None, inputs=[quiz_game_tts_out], js=js_play)
    
    js_timer = "(d) => { clearInterval(window.qT); if(!d||d==='stop')return; let t=parseFloat(d), el=document.getElementById('qt'); window.qT=setInterval(()=>{t=Math.max(0,t-0.1); if(el)el.innerText=t.toFixed(1)+'s'; if(t<=0)clearInterval(window.qT)},100); }"
    quiz_game_timer_out.change(None, inputs=[quiz_game_timer_out], js=js_timer)

    def stop_and_return_menu(flag, state_dict, to_main):
        flag[0] = False
        state_dict["buffer"].clear()
        state_dict["sr"] = None
        return (
            gr.update(visible=to_main), gr.update(visible=not to_main), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(value="<h2>Awaiting start...</h2>"), gr.update(value="<h2>Awaiting start...</h2>"), gr.update(value="<h2>Awaiting start...</h2>"), gr.update(value="<h2>Awaiting start...</h2>"), gr.update(value="<h2>Awaiting start...</h2>")
        )

    cancel_js = "(f, s) => { let a=document.getElementById('ap'); if(a) a.pause(); clearInterval(window.qT); return [f, s]; }"

    btn_stop_soko.click(lambda f, s: stop_and_return_menu(f, s, True), inputs=[live_flag, state], outputs=[menu_group_main, menu_group_quiz, soko_group, g2048_group, quiz_live_group, quiz_admin_group, quiz_game_group, soko_html, g2048_html, quiz_live_html, admin_html, quiz_game_html], js=cancel_js)
    btn_stop_2048.click(lambda f, s: stop_and_return_menu(f, s, True), inputs=[live_flag, state], outputs=[menu_group_main, menu_group_quiz, soko_group, g2048_group, quiz_live_group, quiz_admin_group, quiz_game_group, soko_html, g2048_html, quiz_live_html, admin_html, quiz_game_html], js=cancel_js)
    btn_stop_quiz_live.click(lambda f, s: stop_and_return_menu(f, s, False), inputs=[live_flag, state], outputs=[menu_group_main, menu_group_quiz, soko_group, g2048_group, quiz_live_group, quiz_admin_group, quiz_game_group, soko_html, g2048_html, quiz_live_html, admin_html, quiz_game_html], js=cancel_js)
    btn_stop_admin.click(lambda f, s: stop_and_return_menu(f, s, False), inputs=[live_flag, state], outputs=[menu_group_main, menu_group_quiz, soko_group, g2048_group, quiz_live_group, quiz_admin_group, quiz_game_group, soko_html, g2048_html, quiz_live_html, admin_html, quiz_game_html], js=cancel_js)
    btn_stop_quiz_game.click(lambda f, s: stop_and_return_menu(f, s, False), inputs=[live_flag, state], outputs=[menu_group_main, menu_group_quiz, soko_group, g2048_group, quiz_live_group, quiz_admin_group, quiz_game_group, soko_html, g2048_html, quiz_live_html, admin_html, quiz_game_html], js=cancel_js)

if __name__ == "__main__":
    base = os.path.dirname(__file__)

    favicon_path = os.path.join(base if os.path.exists(os.path.join(base, "assets")) else os.path.dirname(base), "assets", "logo.png")
    demo.launch(
        server_name="0.0.0.0", server_port=8000, theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, favicon_path=favicon_path
    )
