"""tts_engines.py — Dual TTS engine selector

Wraps the base tts module and adds support for multiple VITS TTS engines:
  "farmerline" — FarmerlineML/yoruba_tts-2025 (default, via base tts.py)
  "mms"        — facebook/mms-tts-yor (Meta MMS-TTS)

Both are VITS models returning (sample_rate, numpy_audio_array) for Gradio.
"""

import unicodedata

import torch
import numpy as np
from transformers import AutoProcessor, VitsModel

# Re-export base engine functions
from tts import load_tts as _load_farmerline, synthesize

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
    """Load TTS preprocessor and model for the specified engine.

    Args:
        device: "cuda" or "cpu".
        engine: "mms" or "farmerline" (default).

    Returns:
        (preprocessor, model, sample_rate) tuple.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown TTS engine '{engine}'. Choose from: {list(ENGINES.keys())}")

    if engine == "farmerline":
        return _load_farmerline(device=device)

    # MMS engine — uses AutoProcessor instead of AutoTokenizer
    model_id = ENGINES["mms"]["model_id"]
    print(f"[TTS] Loading {model_id} (engine=mms)...")
    preprocessor = AutoProcessor.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(device)
    model.eval()
    sample_rate = model.config.sampling_rate
    print(f"[TTS] Loaded. Sample rate: {sample_rate} Hz")
    return preprocessor, model, sample_rate
