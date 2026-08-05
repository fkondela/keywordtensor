import keywordtensor as kt
from pynput.keyboard import Key, Controller
import time
import sys
import signal
import os

def signal_handler(sig, frame):
    print("\nInterrupted by user")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

keyboard = Controller()
model = kt.Engine()

def next_slide():
    print("Detected: 'next' -> Next slide")
    keyboard.tap(Key.right)
    time.sleep(1.5)

def prev_slide():
    print("Detected: 'back' -> Previous slide")
    keyboard.tap(Key.left)
    time.sleep(1.5)

print("Listening...")

try:
    model.listen(
        model_path="media_control",
        actions={
            "next": next_slide,
            "back": prev_slide
        },
        min_confidence=0.7,
        n_averages=1
    )
except KeyboardInterrupt:
    print("\nStopped listening")
    sys.exit(0)
