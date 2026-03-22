"""translate.py — English→Yoruba translation via NLLB-200

Uses facebook/nllb-200-distilled-600M for fast, accurate translation.
Lazy-loads the model on first call to avoid startup cost when not needed.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
YORUBA_LANG_CODE = "yor_Latn"

_model = None
_tokenizer = None


def load_translator() -> tuple:
    """Load NLLB-200 translator model and tokenizer.

    Returns:
        (model, tokenizer) tuple. Cached globally after first call.
    """
    global _model, _tokenizer
    if _model is None:
        print(f"[Translate] Loading {NLLB_MODEL_ID}...")
        _tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
        _model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID)
        _model.eval()
        print("[Translate] Model ready.")
    return _model, _tokenizer


def translate_to_yoruba(text: str) -> str:
    """Translate English text to Yoruba.

    Args:
        text: English input text.

    Returns:
        Yoruba translation string.
    """
    model, tokenizer = load_translator()
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    forced_bos = tokenizer.convert_tokens_to_ids(YORUBA_LANG_CODE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=256,
        )

    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]
