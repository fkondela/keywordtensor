import keywordtensor as kt
import time
import sys
import signal
import os
from pynput.keyboard import Key, Controller

def signal_handler(sig, frame):
    print("\nInterrupted by user")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

model = kt.Engine()
keyboard = Controller()

def play_music():
    print("Detected: 'play' -> Play/Pause")
    keyboard.tap(Key.media_play_pause)
    time.sleep(1.5)

def stop_music():
    print("Detected: 'stop' -> Play/Pause")
    keyboard.tap(Key.media_play_pause)
    time.sleep(1.5)

def next_track():
    print("Detected: 'next' -> Next track")
    keyboard.tap(Key.media_next)
    time.sleep(1.5)

def prev_track():
    print("Detected: 'back' -> Previous track")
    keyboard.tap(Key.media_previous)
    time.sleep(1.5)

print("Listening...")

try:
    model.listen(
        model_path="media_control",
        actions={
            "play": play_music,
            "stop": stop_music,
            "next": next_track,
            "back": prev_track
        },
        min_confidence=0.7,
        n_averages=3
    )
except KeyboardInterrupt:
    print("\nStopped listening")
    sys.exit(0)
