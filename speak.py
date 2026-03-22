"""speak.py — Yoruba TTS CLI

Standalone command-line tool for text-to-speech.
Supports English input (auto-translated via NLLB-200) or direct Yoruba.

Usage:
    cd ~/yoruba_ai
    ~/ai-env/bin/python speak.py "Good morning, how are you?"
    ~/ai-env/bin/python speak.py "Báwo ni?" --lang yo
    ~/ai-env/bin/python speak.py "Hello" --engine mms -o greeting.wav
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from translate import translate_to_yoruba
from tts import synthesize
from tts_engines import load_tts, ENGINES, DEFAULT_ENGINE


def speak(
    text: str,
    lang: str = "en",
    engine: str = DEFAULT_ENGINE,
    out: str | None = None,
    device: str = "cpu",
) -> str:
    """Convert text to Yoruba speech WAV.

    Args:
        text: Input text (English or Yoruba).
        lang: "en" (auto-translate) or "yo" (already Yoruba).
        engine: TTS engine name ("mms" or "farmerline").
        out: Output WAV path. Defaults to ~/yoruba_ai/outputs/<hash>.wav.
        device: "cuda" or "cpu".

    Returns:
        Path to the generated WAV file.
    """
    if lang == "en":
        print(f"[speak] Translating: {text[:70]}")
        yoruba = translate_to_yoruba(text)
        print(f"[speak] Yoruba:      {yoruba}")
    else:
        yoruba = text

    preprocessor, model, sample_rate = load_tts(device=device, engine=engine)
    sr, audio = synthesize(preprocessor, model, sample_rate, yoruba, device=device)

    # Normalize and convert to int16 for WAV
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.95
    audio_int16 = (audio * 32767).astype(np.int16)

    if out is None:
        h = hashlib.md5(text.encode()).hexdigest()[:8]
        out = str(Path(__file__).parent / "outputs" / f"{h}.wav")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, sr, audio_int16)
    print(f"[speak] Saved: {out}  ({sr}Hz, {len(audio_int16)/sr:.1f}s)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yoruba TTS — speak text in Yoruba")
    parser.add_argument("text", help="Input text")
    parser.add_argument(
        "--lang", "-l", default="en", choices=["en", "yo"],
        help="en=English (auto-translate), yo=pass Yoruba directly",
    )
    parser.add_argument(
        "--engine", "-e", default=DEFAULT_ENGINE, choices=list(ENGINES.keys()),
        help=f"TTS engine (default: {DEFAULT_ENGINE})",
    )
    parser.add_argument("--out", "-o", default=None, help="Output WAV path")
    parser.add_argument(
        "--device", "-d", default="cpu", choices=["cpu", "cuda"],
        help="Device (default: cpu)",
    )
    args = parser.parse_args()

    path = speak(args.text, lang=args.lang, engine=args.engine, out=args.out, device=args.device)
    print(f"\nPlay: ffplay -nodisp -autoexit {path}")
