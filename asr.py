"""asr.py — Yoruba Automatic Speech Recognition

Uses faster-whisper (CTranslate2, int8) with large-v3 for:
- Yoruba + Nigerian Pidgin English code-switching
- Fast CPU inference (~30s vs timeout with HF transformers)
- No forced language — auto-detects mixed speech
"""

import numpy as np
import soundfile as sf

from faster_whisper import WhisperModel

MODEL_SIZE = "large-v3"


def load_asr(device: str = "cuda") -> WhisperModel:
    """Load faster-whisper large-v3 in int8.

    Runs on CPU regardless of device arg — keeps VRAM free for TTS.
    First call downloads the model (~1.5GB); cached after that.
    """
    print(f"[ASR] Loading faster-whisper {MODEL_SIZE} (int8, CPU)...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("[ASR] Model loaded.")
    return model


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize amplitude to avoid clipping artifacts."""
    peak = np.abs(audio).max()
    if peak > 0.0:
        audio = audio / peak * 0.85
    return audio


def transcribe(model: WhisperModel, audio_path: str) -> str:
    """Transcribe audio file — handles Yoruba/Pidgin code-switching.

    Args:
        model: Loaded WhisperModel instance.
        audio_path: Path to audio file (wav, mp3, ogg, etc).

    Returns:
        Transcribed text string with original casing.
    """
    # Load + normalize to fix clipping before passing to model
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono
    audio = _normalize(audio)

    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language=None,       # auto-detect: handles Yoruba + Pidgin
        vad_filter=True,     # skip silence, reduces hallucination
        vad_parameters={"min_silence_duration_ms": 300},
    )

    text = " ".join(s.text.strip() for s in segments)
    print(f"[ASR] Detected language: {info.language} ({info.language_probability:.0%})")
    return text
