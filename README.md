<div align="center">
  <img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.png" width="300" alt="KeywordTensor Logo">
  <br>
  <p><strong>A Python library for training custom keyword spotting models and running real-time voice command detection.</strong></p>

  [![PyPI - Version](https://img.shields.io/pypi/v/keywordtensor?style=flat-square&color=blue)](https://pypi.org/project/keywordtensor/)
  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)
</div>

---

## ⚡ About KeywordTensor
KeywordTensor is built for developers who want to integrate voice commands into their Python projects without requiring deep knowledge of audio processing. 

- **Download public datasets** so you can choose your own words, and let KeywordTensor automatically distil them into a tiny personalized model for your device.
- **Bring your own audio files**: Just put your audio files into folders (e.g., `dataset/hello/`, `dataset/stop/`).
- **Trigger custom Python actions**: Easily map recognized words directly to your own Python functions. No Speech-to-Text required—KeywordTensor detects predefined commands and directly triggers Python callbacks.
- **Automated Export & Config**: Training automatically generates your optimized model and its configuration file. This allows you to launch live inference with a single command later. No manual saving required!
- **Lightweight Edge Variant**: A lightweight variant without heavy training dependencies. It allows running models on microcontrollers and IoT devices.
- **Built-in Audio Augmentation**: We automatically mutate your audio files during training to improve robustness in noisy environments.
- **SpecAugment Pipeline**: Raw audio is converted to Mel-spectrograms and augmented with masking techniques. The model learns to recognize commands even if the microphone crackles or the audio drops out.
- **Continuous Listening**: A rolling buffer averages predictions over time to prevent sudden false positive clicks.
- **Full Control**: We hide the complexity by default, but give you full access to all training, recording, and listening parameters.

---

## 📦 Pre-trained Models
Don't have time to record your own dataset? You can use our ready-to-go models.

- **`prawda_falsz`**
  <a href="https://keywordtensor-hqang5gnfte7hrhn.polandcentral-01.azurewebsites.net"><img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.png" height="28" align="top" alt="KeywordTensor Logo"><img src="https://img.shields.io/badge/Live_Demo-0089D6?style=for-the-badge" align="top" alt="Live Demo"></a>
  
  [SOON] A highly robust model trained specifically to handle high-pitched children's voices and extremely noisy environments. Designed for live public demonstration - *"Noc Naukowców"* (Researchers' Night) event.
- **More models coming soon!**

---

## 💻 Quick Start & API

### 1. Installation (Choose your variant)
The library is available in two variants on PyPI depending on your needs:

- **`pip install keywordtensor`**
  Installs the full training environment. Use this on your PC or laptop to train your models.

- **`pip install keywordtensor-edge`**
  A lightweight runtime variant. It completely strips out heavy training dependencies, providing only what is needed for real-time inference and dataset collection (`listen()` and `record()`). Perfect for microcontrollers or IoT devices.

---

### 2. Creating your own dataset
If you don't want to use public datasets, you can easily record your own voice to build a custom dataset using the built-in `.record()` tool.

```python
import time
import keywordtensor as kt

model = kt.Engine()

# Define a custom recording action to print progress
def on_record(start_recording, current_time, total_time):
    print("Get ready... speak now!")
    start_recording() # You MUST call this to actually start listening
    
    # Simple loop to print the current time
    while current_time() < total_time:
        print(f"Recording: {current_time():.1f}s / {total_time}s", end="\r")
        time.sleep(0.1)
    print("\nSaved!")

# Record 50 samples of the word "hello" and "stop"
model.record(
    target="my_dataset", 
    classes=["hello", "stop"], 
    samples=50,
    actions={
        "hello": on_record,
        "stop": on_record
    }
)
```

**Record parameters:**
Available parameters in `.record()`:
- `target` *(required)*: Path where the audio folders will be saved.
- `classes` *(required)*: List of strings. Words you want to record.
- `samples` *(default: 100)*: Number of audio samples to record per class.
- `duration` *(default: 1.0)*: The exact duration of each audio clip in seconds.
- `source` *(default: "microphone")*: Audio input source. 
  - `"microphone"` uses the default system microphone. 
  - `"microphone:1"` uses a specific microphone ID. 
  - `my_variable`: You can pass your own audio buffer directly or a tuple `(sample_rate, audio_buffer)`. If you pass a tuple with a different sample rate, KeywordTensor will automatically resample it to `sr` under the hood!
- `sr` *(default: 16000)*: Sample rate for the recorded audio files.
- `actions` *(default: prints recording progress)*: Optional dictionary mapping words to custom callbacks. If provided, your callback will receive three kwargs: `start_recording` (a callable you must execute to begin recording), `current_time` (a callable returning elapsed seconds), and `total_time` (the target duration).
- `stop` *(default: None)*: Optional callback function that returns `True` to stop the recording loop.

---

### 3. Training your model
The `.train()` method takes your audio files and trains a highly optimized neural network under the hood.

```python
import keywordtensor

model = keywordtensor.Engine()

# The engine automatically applies audio & spectrogram augmentations during training
model.train(
    dataset="google",
    classes=["up", "down", "mixed:other"],
    model_path="my_custom_model"
)
```

**Training parameters:**
You have total control over the pipeline. Available parameters in `.train()`:
- `dataset` *(required)*: Path to your audio dataset. You can provide a local folder path, a direct download link, a built-in dataset (`"google"`, `"mswc"`), or any Hugging Face dataset (`"hf:username/repo"`).
- `classes` *(default: all subfolders)*: List of specific words (folders) you want to recognize. If not provided, it trains on all available folders. **Pro-tip:** Add `"mixed:other"` to the list, and the engine will automatically aggregate random words from your dataset to create a robust background noise class!
- `model_path` *(default: 'myownmodel')*: Name of the final exported model.
- `duration` *(default: 1.0)*: The exact duration of your audio clips in seconds. If an audio clip is shorter, it will be automatically padded with zeros (silence). If it is longer, it will be accurately truncated to match this length.
- `sr` *(default: 16000)*: The target sample rate of the trained model. If your audio files have a different sample rate, the engine will automatically resample them under the hood.
- `epochs` *(default: 10)*: Number of training cycles over your dataset.
- `batch_size` *(default: 32)*: Number of audio samples processed simultaneously.
- `valid_pct` *(default: 0.1)*: Percentage of data reserved for validation.
- `learning_rate` *(Automatic)*: The engine dynamically searches for the optimal learning rate for your specific dataset and automatically applies the One-Cycle Policy. We do not expose manual LR tuning because the entire process is fully automated.
- `wd` *(default: 0.01)*: Weight decay (L2 penalty) to prevent overfitting.
- `eps` *(default: 0.01)*: Label smoothing epsilon to improve generalization.

---

### 4. Live Inference & Custom Actions
Once trained (or using a pre-trained model like `prawda_falsz`), you can run real-time inference using your microphone.

```python
import time
import keywordtensor as kt

model = kt.Engine()

# Define your custom actions
def on_hello():
    print("Action triggered: 'Hello' detected!")
    # We add a 2 second cooldown so the engine doesn't trigger 
    # this multiple times during a single utterance
    time.sleep(2)

def on_stop():
    print("Action triggered: Stopping the robot!")
    time.sleep(2)

# Map keywords to your Python functions
model.listen(
    model_path="my_custom_model",
    actions={
        "hello": on_hello,
        "stop": on_stop
    },
    min_confidence=0.6,
    n_averages=3
)
```

**Listen parameters:**
The `.listen()` method itself accepts the following runtime arguments:
- `model_path` *(required)*: The name of the model to load. You can provide the path to your own trained model, or use the built-in `"prawda_falsz"` model which is highly robust to noise and pitched voices.
- `actions` *(default: prints detection and pauses)*: Optional dictionary mapping detected keywords to your own Python callbacks. If not provided, the engine simply prints the detected word and waits for the sample duration to avoid spam. If you provide callbacks, they execute immediately upon detection, and you must implement any required "cooldown" inside your function (the listen core will pause while your function runs).
- `min_confidence` *(default: 0.6)*: The probability threshold (0.0 to 1.0) required to trigger the action.
- `n_averages` *(default: 3)*: Temporal smoothing. Averages the last *N* predictions to prevent false positive clicks.
- `listen_time` *(default: -1)*: How long to listen in seconds. `-1` means listen forever, `0` performs a single prediction, and `>0` sets a specific duration.
- `stop` *(default: None)*: Optional callback function that returns `True` to stop the listening loop.
- `source` *(default: "microphone")*: Audio input source. 
  - `"microphone"` uses the default system microphone. 
  - `"microphone:1"` uses a specific microphone ID. 
  - `my_variable`: You can pass your own audio buffer directly or a tuple `(sample_rate, audio_buffer)` for automatic resampling.
- `threads` *(default: None)*: Number of CPU threads to use for ONNX inference.

**Config file parameters:**
The rest of the underlying parameters are loaded automatically from the `<model_path>_config.json` file. When you run `.train()`, this configuration is automatically generated for you. However, **if you trained an ONNX model entirely outside of KeywordTensor**, simply drop it into your folder, create a matching `your_model_config.json` file next to it, and `.listen()` will load and run it seamlessly!

```json
{
    "labels": ["hello", "stop"],
    "duration": 1.0,
    "sr": 16000,
    "mean": -40.15,
    "std": 17.35
}
```

This file dictates the rules for the inference engine:
- `labels`: The list of keywords the model was trained on.
- `duration`: The size of the rolling audio buffer in seconds.
- `sr`: The target sample rate for the microphone.
- `mean` / `std`: Normalization statistics for the Mel-spectrogram.
