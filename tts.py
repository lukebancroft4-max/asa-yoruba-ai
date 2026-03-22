"""tts.py — Yoruba Text-to-Speech (dual engine)

Engines:
  "mms"        — facebook/mms-tts-yor (Meta MMS-TTS, VITS-based)
  "farmerline" — FarmerlineML/yoruba_tts-2025 (community VITS model)

Both are VITS models. MMS uses AutoProcessor; FarmerlineML uses AutoTokenizer.
NFC normalization ensures consistent Unicode encoding for Yoruba diacritics.
Returns (sample_rate, numpy_audio_array) suitable for Gradio's gr.Audio output.
"""

import unicodedata

import torch
import numpy as np
from transformers import AutoProcessor, AutoTokenizer, VitsModel

# Fix: system cuDNN 9.0.0.312 is broken on this machine
torch.backends.cudnn.enabled = False

ENGINES = {
    "mms": {
        "model_id": "facebook/mms-tts-yor",
        "preprocessor": "processor",
    },
    "farmerline": {
        "model_id": "FarmerlineML/yoruba_tts-2025",
        "preprocessor": "tokenizer",
    },
}

DEFAULT_ENGINE = "farmerline"


def load_tts(device: str = "cuda", engine: str = DEFAULT_ENGINE) -> tuple:
    """Load TTS preprocessor and model.

    Args:
        device: "cuda" or "cpu".
        engine: "mms" or "farmerline" (default).

    Returns:
        (preprocessor, model, sample_rate) tuple.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown TTS engine '{engine}'. Choose from: {list(ENGINES.keys())}")

    config = ENGINES[engine]
    model_id = config["model_id"]
    print(f"[TTS] Loading {model_id} (engine={engine})...")

    if config["preprocessor"] == "tokenizer":
        preprocessor = AutoTokenizer.from_pretrained(model_id)
    else:
        preprocessor = AutoProcessor.from_pretrained(model_id)

    model = VitsModel.from_pretrained(model_id).to(device)
    model.eval()
    sample_rate = model.config.sampling_rate
    print(f"[TTS] Loaded. Sample rate: {sample_rate} Hz")
    return preprocessor, model, sample_rate


def synthesize(
    preprocessor,
    model,
    sample_rate: int,
    text: str,
    device: str = "cuda",
) -> tuple[int, np.ndarray]:
    """Convert Yoruba text to audio waveform.

    Args:
        preprocessor: AutoProcessor or AutoTokenizer (both support text= kwarg).
        model: Loaded VitsModel.
        sample_rate: Model's native sample rate (typically 16000).
        text: Yoruba text to speak (diacritics preserved).
        device: "cuda" or "cpu".

    Returns:
        (sample_rate, float32 numpy array) for Gradio.
    """
    text = unicodedata.normalize("NFC", text)
    inputs = preprocessor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        waveform = model(**inputs).waveform

    audio = waveform.squeeze(0).cpu().numpy().astype(np.float32)
    return sample_rate, audio
