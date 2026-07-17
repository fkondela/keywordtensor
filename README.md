<div align="center">
  <img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.png" width="300" alt="KeywordTensor Logo">
  <br>
  <p><strong>A Python library for training custom keyword spotting models and running real-time voice command detection.</strong></p>

  [![PyPI - Version](https://img.shields.io/pypi/v/keywordtensor?style=flat-square&color=blue)](https://pypi.org/project/keywordtensor/)
  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
</div>

---

## ⚡ About KeywordTensor
KeywordTensor is built for developers who want to integrate voice commands into their Python projects without requiring deep knowledge of audio processing. 

- **Bring your own `.wav` files**: Just put your audio files into folders (e.g., `dataset/hello/`, `dataset/stop/`).
- **Trigger custom Python actions**: Easily map recognized words directly to your own Python functions. No Speech-to-Text required—KeywordTensor detects predefined commands and directly triggers Python callbacks.
- **Automated Export & Config**: Training automatically generates your optimized model and its configuration file. This allows you to launch live inference with a single command later. No manual saving required!
- **Built-in Audio Augmentation**: We automatically mutate your `.wav` files during training (PitchShift, Gain & Polarity Inversion, Colored Noise) to improve robustness in noisy environments.
- **SpecAugment Pipeline**: Raw audio is converted to Mel-spectrograms with Time and Frequency Masking applied. The model learns to recognize commands even if the microphone crackles or the audio drops out.
- **Continuous Listening**: A rolling buffer averages predictions over time to prevent sudden false positive clicks.
- **Full Control**: We hide the complexity by default, but give you full access to all deep learning training and listening parameters (training configuration, validation settings, and inference thresholds).

---

## 📦 Pre-trained Models
Don't have time to record your own dataset? You can use our ready-to-go models.

- **`prawda_falsz`**
  <a href="https://keywordtensor-hqang5gnfte7hrhn.polandcentral-01.azurewebsites.net"><img src="https://raw.githubusercontent.com/fkondela/keywordtensor/main/assets/logo.png" height="28" align="top" alt="KeywordTensor Logo"><img src="https://img.shields.io/badge/Live_Demo-0089D6?style=for-the-badge" align="top" alt="Live Demo"></a>
  
  A highly robust model trained specifically to handle high-pitched children's voices and extremely noisy environments. This model was successfully deployed in a live public demonstration during the *"Noc Naukowców"* (Researchers' Night) event.
- **More models coming soon!**

---

## 💻 Quick Start & API

### 1. Installation (Choose your variant)
The library is available in two variants on PyPI depending on your needs:

- **`pip install keywordtensor`**
  Installs the full training environment. Use this on your PC or Server to train your models.

- **`pip install keywordtensor-edge`**
  A lightweight runtime variant. It strips out heavy training dependencies (like `fastai`), providing only what is needed for real-time inference (`listen()`). Perfect for Raspberry Pi or IoT devices.

---

### 2. Training your model
The `.train()` method takes your audio files and trains a neural network using PyTorch and FastAI under the hood. 

```python
import keywordtensor

model = keywordtensor.Engine()

# The engine automatically applies audio & spectrogram augmentations during training
model.train(
    dataset_path="path/to/audio/dataset",
    model_name="my_custom_model",
    epochs=30,
    batch_size=32
)
```

**Training parameters:**
You have total control over the pipeline. Available parameters in `.train()`:
- `dataset_path` *(required)*: Path to your dataset folder. Any number of folders (classes) is supported.
- `model_name` *(default: 'myownmodel')*: Name of the final exported model.
- `epochs` *(default: 30)*: Number of training cycles over your dataset.
- `batch_size` *(default: 32)*: Number of audio samples processed simultaneously.
- `learning_rate` *(Automatic)*: The engine dynamically searches for the optimal learning rate for your specific dataset and automatically applies the One-Cycle Policy.
- `wd` *(default: 0.01)*: Weight decay (L2 penalty) to prevent overfitting.
- `eps` *(default: 0.01)*: Label smoothing epsilon to improve generalization.
- `valid_pct` *(default: 0.1)*: Percentage of data reserved for validation.
- `duration` *(default: 3.0)*: The exact duration of your audio clips in seconds. If an audio clip is shorter, it will be automatically padded with zeros (silence). If it is longer, it will be accurately truncated to match this length.
- `sr` *(default: 16000)*: Sample rate of your audio files.

---

### 3. Live Inference & Custom Actions
Once trained (or using a pre-trained model like `prawda_falsz`), you can run real-time inference using your microphone.

```python
import keywordtensor as kt

model = kt.Engine()

# Define your custom actions
def on_hello():
    print("Action triggered: 'Hello' detected!")

def on_stop():
    print("Action triggered: Stopping the robot!")

# Map keywords to your Python functions with per-keyword cooldowns
model.listen(
    model_name="my_custom_model",
    actions={
        "hello": {"function": on_hello, "cooldown": 2.0},
        "stop": {"function": on_stop, "cooldown": 3.0}
    },
    min_confidence=0.6,
    n_averages=3,
    source="microphone"
)
```

**Listen parameters:**
The `.listen()` method itself accepts the following runtime arguments:
- `model_name` *(required)*: The name of the model to load. You can provide the path to your own trained model, or use the built-in `"prawda_falsz"` model which is highly robust to noise and pitched voices.
- `actions` *(default: None)*: Optional dictionary mapping detected keywords to Python callbacks. You can pass just a function (defaults to 0.0s cooldown), or a dictionary for precise control: `{"function": your_function, "cooldown": 2.0}`. Cooldowns are tracked individually per keyword!
- `min_confidence` *(default: 0.6)*: The probability threshold (0.0 to 1.0) required to trigger the action.
- `n_averages` *(default: 3)*: Temporal smoothing. Averages the last *N* predictions to prevent false positive clicks.
- `source` *(default: "microphone")*: Audio input source. 
  - `"microphone"` uses the default system microphone. 
  - `"microphone:1"` uses a specific microphone ID. 
  - `my_variable` You can also pass your own Python variable (a generator) that yields current microphone audio data (arrays of Float32 at 16000Hz) to easily plug KeywordTensor into your own custom apps!

**Config file parameters:**
The rest of the underlying parameters are loaded automatically from the `<model_name>_config.json` file! 
When you run `.train()`, this file is automatically generated for you. It looks like this:

```json
{
    "labels": ["hello", "stop"],
    "mean": -40.15,
    "std": 17.35,
    "duration": 3.0,
    "sr": 16000
}
```

This file dictates the rules for the inference engine:
- `labels`: The list of keywords the model was trained on.
- `duration`: The size of the rolling audio buffer in seconds.
- `sr`: The microphone sample rate.
- `mean` / `std`: Normalization statistics for the Mel-spectrogram.

> 💡 **Total Flexibility:** 
> Want to adjust the microphone sample rate or buffer duration without retraining? Just open the JSON file and edit it! 
> 
> **Bringing your own model?** No problem! If you trained an ONNX model entirely outside of KeywordTensor, simply drop it into your folder, create a matching `your_model_config.json` file next to it with the parameters above, and the `.listen()` method will load and run your external model.
