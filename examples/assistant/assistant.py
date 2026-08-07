import keywordtensor as kt
import time
import sys
import signal
import os
import webbrowser

def signal_handler(sig, frame):
    print("\nInterrupted by user")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
model = kt.Engine()

def open_gemini():
    print("Assistant activated -> Opening Gemini")
    webbrowser.open("https://gemini.google.com")
    time.sleep(2.0)

print("Listening...")

try:
    model.listen(
        model_path="assistant",
        actions={
            "marvin": open_gemini,
            "sheila": open_gemini
        },
        min_confidence=0.6,
        n_averages=2
    )
except KeyboardInterrupt:
    print("\nStopped listening")
    sys.exit(0)
