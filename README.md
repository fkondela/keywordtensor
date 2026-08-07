<div align="center">
  <img alt="KeywordTensor Logo" src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo_light.svg#gh-light-mode-only" width="700">
  <img alt="KeywordTensor Logo" src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo_dark.svg#gh-dark-mode-only" width="700">
</div>

---

<div align="center">
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
- **Trigger custom Python actions**: Easily map recognized words directly to your own Python functions. KeywordTensor detects predefined commands and triggers your custom callbacks.
- **Automated Export & Config**: Training automatically generates your optimized model and its configuration file. This allows you to launch live inference with a single command later. No manual saving required!
- **Lightweight Edge Variant**: A lightweight variant without heavy training dependencies. It allows running models on microcontrollers and IoT devices.
- **Built-in Audio Augmentation**: We automatically mutate your audio files during training to improve robustness in noisy environments.
- **SpecAugment Pipeline**: Raw audio is converted to Mel-spectrograms and augmented with masking techniques. The model learns to recognize commands even if the microphone crackles or the audio drops out.
- **Continuous Listening**: A rolling buffer averages predictions over time to prevent sudden false positive clicks.
- **Full Control**: We hide the complexity by default, but give you full access to all training, recording, and listening parameters.

---

## 📦 Pre-trained Models
Don't have time to record your own dataset? You can use our ready-to-go models.

- **`tak_nie`** (`tak`, `nie`, `other`)
  - **Example — Quiz:** <a href="https://keywordtensor-hqang5gnfte7hrhn.polandcentral-01.azurewebsites.net"><img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.svg" height="28" align="absmiddle" alt="KeywordTensor Logo"><img src="https://img.shields.io/badge/Live_Demo-0089D6?style=for-the-badge" align="absmiddle" alt="Live Demo"></a>
    
    [SOON] A highly robust model trained specifically to handle high-pitched children's voices and extremely noisy environments. Designed for live public demonstration - *"Noc Naukowców"* (Researchers' Night) event.

- **`spatial_nav`** (`backward`, `down`, `forward`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes`, `other`)
  - **Example — Sokoban:** <a href="https://keywordtensor-hqang5gnfte7hrhn.polandcentral-01.azurewebsites.net"><img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.svg" height="28" align="absmiddle" alt="KeywordTensor Logo"><img src="https://img.shields.io/badge/Live_Demo-0089D6?style=for-the-badge" align="absmiddle" alt="Live Demo"></a>
  - **Example — 2048:** <a href="https://keywordtensor-hqang5gnfte7hrhn.polandcentral-01.azurewebsites.net"><img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.svg" height="28" align="absmiddle" alt="KeywordTensor Logo"><img src="https://img.shields.io/badge/Live_Demo-0089D6?style=for-the-badge" align="absmiddle" alt="Live Demo"></a>

- **`numbers`** (`zero`, `one`, `two`, `three`, `four`, `five`, `six`, `seven`, `eight`, `nine`, `other`)

- **`media_control`** (`play`, `stop`, `next`, `back`, `other`)
  - **Example — Media Player Control:** *(Mac/Linux: `python3`, Windows: `python`)*
    ```bash
    python3 -c "import urllib.request, subprocess, sys, os; os.makedirs('media_player_control', exist_ok=True); os.chdir('media_player_control'); urllib.request.urlretrieve('https://raw.githubusercontent.com/fkondela/KeywordTensor/main/examples/media_player_control/media_player_control.py', 'app.py'); urllib.request.urlretrieve('https://raw.githubusercontent.com/fkondela/KeywordTensor/main/examples/media_player_control/requirements.txt', 'req.txt'); subprocess.run([sys.executable, '-m', 'venv', '.venv']); p = r'.venv\Scripts\python.exe' if sys.platform=='win32' else '.venv/bin/python'; subprocess.run([p, '-m', 'pip', 'install', '-r', 'req.txt']); subprocess.run([p, 'app.py'])"
    ```
  - **Example — Presentation Controller:** *(Mac/Linux: `python3`, Windows: `python`)*
    ```bash
    python3 -c "import urllib.request, subprocess, sys, os; os.makedirs('presentation_controller', exist_ok=True); os.chdir('presentation_controller'); urllib.request.urlretrieve('https://raw.githubusercontent.com/fkondela/KeywordTensor/main/examples/presentation_controller/presentation_controller.py', 'app.py'); urllib.request.urlretrieve('https://raw.githubusercontent.com/fkondela/KeywordTensor/main/examples/presentation_controller/requirements.txt', 'req.txt'); subprocess.run([sys.executable, '-m', 'venv', '.venv']); p = r'.venv\Scripts\python.exe' if sys.platform=='win32' else '.venv/bin/python'; subprocess.run([p, '-m', 'pip', 'install', '-r', 'req.txt']); subprocess.run([p, 'app.py'])"
    ```

- **`assistant`** (`marvin`, `sheila`, `other`)
  - **Example — Desktop Assistant:** *(Mac/Linux: `python3`, Windows: `python`)*
    ```bash
    python3 -c "import urllib.request, subprocess, sys, os; os.makedirs('assistant', exist_ok=True); os.chdir('assistant'); urllib.request.urlretrieve('https://raw.githubusercontent.com/fkondela/KeywordTensor/main/examples/assistant/assistant.py', 'app.py'); urllib.request.urlretrieve('https://raw.githubusercontent.com/fkondela/KeywordTensor/main/examples/assistant/requirements.txt', 'req.txt'); subprocess.run([sys.executable, '-m', 'venv', '.venv']); p = r'.venv\Scripts\python.exe' if sys.platform=='win32' else '.venv/bin/python'; subprocess.run([p, '-m', 'pip', 'install', '-r', 'req.txt']); subprocess.run([p, 'app.py'])"
    ```

- **`smarthome_control`** - **SOON**

> **Acknowledgments:** Pre-trained models in this repository may utilize data from [MSWC](https://huggingface.co/datasets/MLCommons/ml_spoken_words) (CC-BY 4.0), [Google Speech Commands](https://research.google/blog/launching-the-speech-commands-dataset/) (CC-BY 4.0), [CAIMAN-ASR-BackgroundNoise](https://huggingface.co/datasets/Myrtle/CAIMAN-ASR-BackgroundNoise) (CC-BY 4.0), and [OpenSLR RIR & Noises](https://www.openslr.org/28/) (CC-BY 4.0).

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
import keywordtensor as kt

model = kt.Engine()

# Record 50 samples of the word "hello" and "stop"
model.record(
    target="my_dataset", 
    classes=["hello", "stop"], 
    samples=50
)
```

**Record parameters:**
Available parameters in `.record()`:
- `target` *(required)*: Path where the audio folders will be saved.
- `classes` *(required)*: List of strings (or a single string). Words you want to record.
- `samples` *(default: 100)*: Number of audio samples to record per class.
- `duration` *(default: 1.0)*: The exact duration of each audio clip in seconds.
- `source` *(default: "microphone")*: Audio input source. 
  - `"microphone"` uses the default system microphone. 
  - `"microphone:1"` uses a specific microphone ID. 
  - `my_variable`: You can pass your own audio buffer directly or a tuple `(sample_rate, audio_buffer)`. If you pass a tuple with a different sample rate, KeywordTensor will automatically resample it to `sr` under the hood!
- `sr` *(default: 16000)*: Sample rate for the recorded audio files.
- `actions` *(default: prints live countdown and progress bar)*: Optional dictionary mapping words to custom callbacks. If provided, your callback will receive three kwargs: `start_recording` (a callable you must execute to begin recording), `current_time` (a callable returning elapsed seconds), and `total_time` (the target duration).
- `stop` *(default: None)*: Optional callback function that returns `True` to stop the recording loop.

---

### 3. Training your model
The `.train()` method takes your audio files and trains a highly optimized neural network under the hood.

```python
import keywordtensor

model = keywordtensor.Engine()

# The engine automatically applies audio & spectrogram augmentations during training
model.train(
    dataset=["google", "caiman", "openrir"],
    classes=["up", "down"],
    bg_classes="background",
    rir_classes="rir",
    mixed=True,
    model_path="my_custom_model"
)
```

**Training parameters:**
You have total control over the pipeline. Available parameters in `.train()`:
- `dataset` *(required)*: Path to your audio dataset. You can provide a local folder path, a direct download link, a built-in dataset ([`"google"`](https://huggingface.co/datasets/speech_commands) (~2.3GB), [`"mswc"`](https://huggingface.co/datasets/MLCommons/ml_spoken_words) (~35GB), [`"mswc-pl"`](https://huggingface.co/datasets/MLCommons/ml_spoken_words) (~2GB), [`"caiman"`](https://huggingface.co/datasets/Myrtle/CAIMAN-ASR-BackgroundNoise) (~4GB), [`"openrir"`](https://www.openslr.org/28/) (~1GB)), or any Hugging Face dataset (`"hf:username/repo"`). **You can also provide a list** (e.g. `["google", "caiman", "openrir", "my_folder"]`) to automatically merge multiple sources!
- `classes` *(required)*: List of strings (or a single string). Only the words you want the model to detect. Remaining folders in the dataset are handled automatically by `bg_classes` and `mixed`.
- `bg_classes` *(required)*: List of classes (or a single string) to use as background noise for audio augmentation. For the built-in `"caiman"` dataset, pass `"background"`.
- `rir_classes` *(default: None)*: List of classes (or a single string) to use as Room Impulse Responses for acoustic environment augmentation. For the built-in `"openrir"` dataset, pass `"rir"`.
- `mixed` *(default: True)*: If `True`, the engine generates an `"other"` class made of 50% background noise and 50% random unused words from your dataset. If `False`, `"other"` contains only background noise files.
- `model_path` *(default: 'myownmodel')*: Name of the final exported model.
- `duration` *(default: 1.0)*: The exact duration of your audio clips in seconds. During training, the engine automatically performs random temporal cropping (for long files) and random zero-padding (for short files) to dramatically improve robustness against temporal shifts. Validation files are statically cropped to ensure consistent evaluation.
- `sr` *(default: 16000)*: The target sample rate of the trained model. If your audio files have a different sample rate, the engine will automatically resample them under the hood.
- `epochs` *(default: 10)*: Number of training cycles over your dataset.
- `batch_size` *(default: 32)*: Number of audio samples processed simultaneously.
- `valid_pct` *(default: 0.1)*: Percentage of data reserved for validation.
- `learning_rate` *(Automatic)*: The engine dynamically searches for the optimal learning rate for your specific dataset and automatically applies the One-Cycle Policy. We do not expose manual LR tuning because the entire process is fully automated.
- `Class Balancing` *(Automatic)*: The engine detects class imbalances in your training set and weights samples accordingly, so smaller classes are not drowned out by larger ones.
- `Metrics` *(Automatic)*: In addition to accuracy, the engine tracks Precision, Recall, and F1 Score (macro) during training.
- `wd` *(default: 0.01)*: Weight decay (L2 penalty) to prevent overfitting.
- `eps` *(default: 0.05)*: Label smoothing epsilon to improve generalization.
- `alpha` *(default: 0.0)*: MixUp augmentation parameter. The library already applies background audio augmentation during training, so enabling this is not recommended. Set to `0.0` to disable.
- `cbs` *(default: None)*: Optional list of fastai Callbacks to inject into the training loop (e.g., for logging to Weights & Biases or TensorBoard).

---

### 4. Live Inference & Custom Actions
Once trained (or using a pre-trained model like `media_control`), you can run real-time inference using your microphone.

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
    n_averages=2
)
```

**Listen parameters:**
The `.listen()` method itself accepts the following runtime arguments:
- `model_path` *(required)*: The name of the model to load. You can provide the path to your own trained model, or use the built-in `"tak_nie"` model which is highly robust to noise and pitched voices. 
- `actions` *(default: prints detection and pauses)*: Optional dictionary mapping detected keywords to your own Python callbacks. If not provided, the engine simply prints the detected word and waits for the sample duration to avoid spam. If you provide callbacks, they execute immediately upon detection, and you must implement any required "cooldown" inside your function (the listen core will pause while your function runs).
- `min_confidence` *(default: 0.6)*: The probability threshold (0.0 to 1.0) required to trigger the action.
- `n_averages` *(default: 2)*: Temporal smoothing. Averages the last *N* predictions to prevent false positive clicks.
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
