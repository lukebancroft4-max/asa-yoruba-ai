"""tts.py — Yoruba Text-to-Speech via FarmerlineML/yoruba_tts-2025

Uses VITS architecture with AutoTokenizer, which correctly handles Yoruba
tonal diacritics (á à é è ó ò etc.). NFC normalization ensures consistent
Unicode encoding regardless of LLM output encoding.
"""

import unicodedata

import torch
import numpy as np
from transformers import AutoTokenizer, VitsModel

TTS_MODEL_ID = "FarmerlineML/yoruba_tts-2025"


def load_tts(device: str = "cuda") -> tuple:
    """Load FarmerlineML TTS tokenizer and model.

    Returns:
        (tokenizer, model, sample_rate) tuple.
    """
    print(f"[TTS] Loading {TTS_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    model = VitsModel.from_pretrained(TTS_MODEL_ID).to(device)
    model.eval()
    sample_rate = model.config.sampling_rate
    print(f"[TTS] Loaded. Sample rate: {sample_rate} Hz")
    return tokenizer, model, sample_rate


def synthesize(
    tokenizer,
    model,
    sample_rate: int,
    text: str,
    device: str = "cuda",
) -> tuple[int, np.ndarray]:
    """Convert Yoruba text to audio waveform.

    Args:
        tokenizer: VITS AutoTokenizer.
        model: Loaded VitsModel.
        sample_rate: Model native sample rate.
        text: Yoruba text to speak (diacritics preserved).
        device: "cuda" or "cpu".

    Returns:
        (sample_rate, float32 numpy array) for Gradio.
    """
    text = unicodedata.normalize("NFC", text)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        waveform = model(**inputs).waveform

    audio = waveform.squeeze().cpu().numpy().astype(np.float32)
    return sample_rate, audio
