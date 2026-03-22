"""tts.py — Yoruba Text-to-Speech via Meta MMS-TTS

Uses facebook/mms-tts-yor (VITS-based), which natively handles Yoruba tonal marks.
Returns (sample_rate, numpy_audio_array) suitable for Gradio's gr.Audio output.
"""

import torch
import numpy as np
from transformers import AutoProcessor, VitsModel

# Fix: system cuDNN 9.0.0.312 is broken on this machine
torch.backends.cudnn.enabled = False

TTS_MODEL_ID = "facebook/mms-tts-yor"


def load_tts(device: str = "cuda") -> tuple:
    """Load MMS-TTS processor and model.

    Returns:
        (processor, model, sample_rate) tuple.
    """
    print(f"[TTS] Loading {TTS_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(TTS_MODEL_ID)
    model = VitsModel.from_pretrained(TTS_MODEL_ID).to(device)
    model.eval()
    sample_rate = model.config.sampling_rate
    print(f"[TTS] Loaded. Sample rate: {sample_rate} Hz")
    return processor, model, sample_rate


def synthesize(
    processor,
    model,
    sample_rate: int,
    text: str,
    device: str = "cuda",
) -> tuple[int, np.ndarray]:
    """Convert Yoruba text to audio waveform.

    Args:
        processor: VITS AutoProcessor.
        model: Loaded VitsModel.
        sample_rate: Model's native sample rate (typically 16000).
        text: Yoruba text to speak (diacritics preserved).
        device: "cuda" or "cpu".

    Returns:
        (sample_rate, float32 numpy array) for Gradio.
    """
    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        waveform = model(**inputs).waveform

    audio = waveform.squeeze().cpu().numpy().astype(np.float32)
    return sample_rate, audio
