# Àṣà — Yoruba AI Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Yoruba TTS pronunciation, add streaming + per-session history, and deploy to HF Spaces.

**Architecture:** TDD order — fix TTS first (highest user impact), then LLM streaming + immutable history, then rewrite broken ASR tests, then rebuild app.py with session isolation and streaming UI, then config/deployment files, then pre-deploy cleanup.

**Tech Stack:** Python 3.13, Gradio 6.9.0, faster-whisper 1.2.1, transformers 5.3.0, FarmerlineML/yoruba_tts-2025 (VITS), NVIDIA OpenAI-compatible API (Llama-3.3-70B), pytest

**Run all tests with:** `cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/ -v`

---

## File Map

| File | Change | Responsibility |
|------|--------|----------------|
| `tts.py` | Modify | Switch to FarmerlineML/yoruba_tts-2025 + NFC normalization |
| `llm.py` | Modify | Add `chat_stream()`, immutable `chat()`, `_trim_history()`, export `MAX_HISTORY_TURNS` |
| `asr.py` | Modify | Remove machine-specific cuDNN workaround line |
| `app.py` | Rewrite | `gr.State` session isolation, `gr.Chatbot` history display, streaming generator, error handling |
| `requirements.txt` | Modify | Add faster-whisper, pin all versions |
| `README.md` | Create | HF Spaces YAML frontmatter (required for deployment) |
| `tests/test_tts.py` | Modify | Swap `AutoProcessor` → `AutoTokenizer`; add NFC test; add model ID test |
| `tests/test_llm.py` | Modify | Update `chat()` callers to unpack tuple; add `chat_stream()` + `_trim_history()` tests |
| `tests/test_asr.py` | Rewrite | Align with faster-whisper: patch `sf.read` + `WhisperModel.transcribe` |
| `tests/test_app.py` | Create | Full pipeline smoke test with all modules mocked |

---

## Task 1: Fix tts.py — Switch Model + NFC Normalization

**Files:**
- Modify: `~/yoruba_ai/tests/test_tts.py`
- Modify: `~/yoruba_ai/tts.py`

The current `facebook/mms-tts-yor` model has wrong Yoruba pronunciation. Replace with `FarmerlineML/yoruba_tts-2025` which uses `AutoTokenizer` and is specifically trained for Yoruba. Add `unicodedata.normalize("NFC", text)` before tokenization to ensure LLM-output diacritics are consistently encoded.

- [ ] **Step 1: Write the failing tests**

Replace the entire content of `tests/test_tts.py`:

```python
"""Tests for tts.py — TTS module (FarmerlineML/yoruba_tts-2025)."""

import os
import sys
import unicodedata
import numpy as np
import torch
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tts as tts_module
from tts import synthesize


def _make_mock_tts_components(audio_length: int = 16000):
    """Build mock (tokenizer, model) for VITS forward pass."""
    mock_encodings = MagicMock()
    mock_inputs = MagicMock()
    mock_encodings.to.return_value = mock_inputs

    mock_tokenizer = MagicMock(return_value=mock_encodings)

    fake_waveform = torch.zeros(1, audio_length)
    mock_output = MagicMock()
    mock_output.waveform = fake_waveform

    mock_model = MagicMock(return_value=mock_output)

    return mock_tokenizer, mock_model


class TestLoadTts:
    def test_returns_tokenizer_model_and_sample_rate(self):
        """load_tts() returns (tokenizer, model, sample_rate) tuple."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 22050
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoTokenizer") as mock_at,
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_at.from_pretrained.return_value = mock_tokenizer
            mock_vm.from_pretrained.return_value = mock_model

            tokenizer, model, sr = tts_module.load_tts(device="cpu")

        assert tokenizer is mock_tokenizer
        assert model is mock_model
        assert sr == 22050

    def test_model_moved_to_correct_device(self):
        """load_tts() calls model.to(device)."""
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 22050
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoTokenizer"),
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_vm.from_pretrained.return_value = mock_model
            tts_module.load_tts(device="cpu")

        mock_model.to.assert_called_once_with("cpu")

    def test_loads_farmerlineml_model(self):
        """load_tts() loads FarmerlineML/yoruba_tts-2025, not mms-tts-yor."""
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 22050
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoTokenizer") as mock_at,
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_vm.from_pretrained.return_value = mock_model
            tts_module.load_tts(device="cpu")

            mock_at.from_pretrained.assert_called_once_with("FarmerlineML/yoruba_tts-2025")
            mock_vm.from_pretrained.assert_called_once_with("FarmerlineML/yoruba_tts-2025")


class TestSynthesize:
    def test_returns_tuple_of_sample_rate_and_array(self):
        """synthesize() returns (int, numpy.ndarray)."""
        tokenizer, model = _make_mock_tts_components()
        sr, audio = synthesize(tokenizer, model, 16000, "Báwo ni?", device="cpu")
        assert isinstance(sr, int)
        assert sr == 16000
        assert isinstance(audio, np.ndarray)

    def test_audio_is_float32(self):
        """synthesize() output is float32."""
        tokenizer, model = _make_mock_tts_components()
        _, audio = synthesize(tokenizer, model, 16000, "test", device="cpu")
        assert audio.dtype == np.float32

    def test_audio_is_1d(self):
        """synthesize() returns 1D mono array."""
        tokenizer, model = _make_mock_tts_components(audio_length=8000)
        _, audio = synthesize(tokenizer, model, 16000, "test", device="cpu")
        assert audio.ndim == 1

    def test_audio_length_matches_waveform(self):
        """synthesize() output length matches model waveform output."""
        tokenizer, model = _make_mock_tts_components(audio_length=24000)
        _, audio = synthesize(tokenizer, model, 16000, "longer text", device="cpu")
        assert len(audio) == 24000

    def test_model_called_once(self):
        """synthesize() calls the model exactly once."""
        tokenizer, model = _make_mock_tts_components()
        synthesize(tokenizer, model, 16000, "Ẹ káàbọ̀", device="cpu")
        model.assert_called_once()

    def test_tokenizer_called_with_text(self):
        """synthesize() passes text (positional) to the tokenizer."""
        tokenizer, model = _make_mock_tts_components()
        synthesize(tokenizer, model, 16000, "Kí ni orúkọ rẹ?", device="cpu")
        tokenizer.assert_called_once_with("Kí ni orúkọ rẹ?", return_tensors="pt")

    def test_nfc_normalization_applied(self):
        """synthesize() normalizes text to NFC before tokenizing."""
        tokenizer, model = _make_mock_tts_components()
        # NFD: "a" + combining acute = "á" as two codepoints
        nfd_text = "Ba\u0301wo ni"
        nfc_text = unicodedata.normalize("NFC", nfd_text)
        synthesize(tokenizer, model, 16000, nfd_text, device="cpu")
        # Tokenizer must receive NFC form
        tokenizer.assert_called_once_with(nfc_text, return_tensors="pt")
```

- [ ] **Step 2: Run tests — expect FAILURES**

```bash
cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/test_tts.py -v
```

Expected: Several FAILED (tests reference `AutoTokenizer`, but `tts.py` still uses `AutoProcessor`).

- [ ] **Step 3: Rewrite tts.py**

```python
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
        sample_rate: Model's native sample rate.
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
```

- [ ] **Step 4: Run tests — expect all PASS**

```bash
cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/test_tts.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/yoruba_ai && git init 2>/dev/null || true
git add tts.py tests/test_tts.py
git commit -m "fix: switch TTS to FarmerlineML/yoruba_tts-2025 + NFC normalization"
```

---

## Task 2: Update llm.py — Immutable History + Streaming + Trimming

**Files:**
- Modify: `~/yoruba_ai/tests/test_llm.py`
- Modify: `~/yoruba_ai/llm.py`

Three changes: `chat()` returns `(reply, new_history)` instead of mutating; `chat_stream()` yields text chunks; `_trim_history()` enforces 20-dict limit internally.

- [ ] **Step 1: Update test_llm.py — fix existing tests + add new ones**

Replace the entire content of `tests/test_llm.py`:

```python
"""Tests for llm.py — LLM module."""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import llm as llm_module
from llm import chat, chat_stream, _trim_history, SYSTEM_PROMPT, MAX_HISTORY_TURNS


class TestBuildClient:
    def test_builds_with_env_key(self):
        """build_client() reads NVIDIA_API_KEY from environment."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"}):
            with patch.object(llm_module, "OpenAI") as mock_openai:
                llm_module.build_client()
                mock_openai.assert_called_once_with(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key="test-key-123",
                )

    def test_raises_when_no_api_key(self):
        """build_client() raises ValueError if no key found."""
        import pytest
        env_backup = os.environ.pop("NVIDIA_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
                llm_module.build_client(api_key=None)
        finally:
            if env_backup:
                os.environ["NVIDIA_API_KEY"] = env_backup

    def test_explicit_key_overrides_env(self):
        """Explicit api_key parameter takes precedence over env var."""
        with patch.object(llm_module, "OpenAI") as mock_openai:
            llm_module.build_client(api_key="explicit-key")
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["api_key"] == "explicit-key"


class TestTrimHistory:
    def test_returns_new_list(self):
        """_trim_history() always returns a new list, never mutates input."""
        history = [{"role": "user", "content": "hi"}]
        result = _trim_history(history)
        assert result is not history

    def test_short_history_unchanged(self):
        """_trim_history() returns all items when len <= MAX_HISTORY_TURNS."""
        history = [{"role": "user", "content": str(i)} for i in range(5)]
        result = _trim_history(history)
        assert result == history

    def test_trims_to_last_20_dicts(self):
        """_trim_history() keeps last 20 dicts (10 pairs) when over limit."""
        history = [{"role": "user", "content": str(i)} for i in range(30)]
        result = _trim_history(history)
        assert len(result) == MAX_HISTORY_TURNS
        assert result == history[-MAX_HISTORY_TURNS:]

    def test_max_history_turns_is_20(self):
        """MAX_HISTORY_TURNS constant equals 20."""
        assert MAX_HISTORY_TURNS == 20

    def test_empty_history_returns_empty(self):
        """_trim_history() handles empty input."""
        assert _trim_history([]) == []


class TestChat:
    def _make_mock_client(self, reply_text: str):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = reply_text
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_returns_tuple_of_reply_and_new_history(self):
        """chat() returns (reply_str, new_history_list) tuple."""
        client = self._make_mock_client("Mo wà dáadáa!")
        result = chat(client, [], "Báwo ni?")
        assert isinstance(result, tuple)
        assert len(result) == 2
        reply, new_history = result
        assert isinstance(reply, str)
        assert isinstance(new_history, list)

    def test_new_history_contains_user_and_assistant(self):
        """chat() new_history includes user message + assistant reply."""
        client = self._make_mock_client("Mo wà dáadáa!")
        _, new_history = chat(client, [], "Báwo ni?")
        assert new_history[0] == {"role": "user", "content": "Báwo ni?"}
        assert new_history[1] == {"role": "assistant", "content": "Mo wà dáadáa!"}

    def test_does_not_mutate_input_history(self):
        """chat() does NOT modify the original history list."""
        client = self._make_mock_client("reply")
        history = [{"role": "user", "content": "first"}]
        original_len = len(history)
        chat(client, history, "second")
        assert len(history) == original_len  # unchanged

    def test_returns_stripped_reply(self):
        """chat() strips whitespace from the reply."""
        client = self._make_mock_client("  Ẹ káàbọ̀!  ")
        reply, _ = chat(client, [], "hello")
        assert reply == "Ẹ káàbọ̀!"

    def test_includes_system_prompt_as_first_message(self):
        """chat() always prepends system prompt to messages sent to API."""
        client = self._make_mock_client("reply")
        chat(client, [], "test")
        call_messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert call_messages[0] == {"role": "system", "content": SYSTEM_PROMPT}

    def test_api_called_with_specified_model(self):
        """chat() passes the model param to the API."""
        client = self._make_mock_client("ok")
        chat(client, [], "hello", model="custom/model-xyz")
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "custom/model-xyz"

    def test_temperature_passed_to_api(self):
        """chat() forwards temperature to API call."""
        client = self._make_mock_client("ok")
        chat(client, [], "hello", temperature=0.1)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1

    def test_trims_history_before_api_call(self):
        """chat() applies _trim_history — only last 20 dicts sent to API."""
        client = self._make_mock_client("ok")
        # 30 messages — only last 20 should reach the API (+ system prompt)
        long_history = [{"role": "user", "content": str(i)} for i in range(30)]
        chat(client, long_history, "new message")
        call_messages = client.chat.completions.create.call_args.kwargs["messages"]
        # system prompt + 20 trimmed + 1 new user = 22
        assert len(call_messages) == 22


class TestChatStream:
    def _make_mock_stream_client(self, chunks: list[str]):
        """Build mock client that streams the given text chunks."""
        mock_client = MagicMock()
        mock_chunks = []
        for text in chunks:
            chunk = MagicMock()
            chunk.choices[0].delta.content = text
            mock_chunks.append(chunk)
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        return mock_client

    def test_yields_text_chunks(self):
        """chat_stream() yields non-empty string chunks."""
        client = self._make_mock_stream_client(["Mo wà ", "dáadáa!"])
        chunks = list(chat_stream(client, [], "Báwo ni?"))
        assert chunks == ["Mo wà ", "dáadáa!"]

    def test_skips_empty_chunks(self):
        """chat_stream() skips chunks where delta.content is None or empty."""
        client = self._make_mock_stream_client(["hello", None, "", "world"])
        # Need to handle None in delta.content
        chunks = list(chat_stream(client, [], "hi"))
        assert "hello" in chunks
        assert "world" in chunks
        assert None not in chunks
        assert "" not in chunks

    def test_called_with_stream_true(self):
        """chat_stream() passes stream=True to the API."""
        client = self._make_mock_stream_client([])
        list(chat_stream(client, [], "test"))
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("stream") is True

    def test_includes_system_prompt(self):
        """chat_stream() prepends system prompt to messages."""
        client = self._make_mock_stream_client([])
        list(chat_stream(client, [], "test"))
        call_messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert call_messages[0] == {"role": "system", "content": SYSTEM_PROMPT}

    def test_trims_history_internally(self):
        """chat_stream() trims history to MAX_HISTORY_TURNS before API call."""
        client = self._make_mock_stream_client([])
        long_history = [{"role": "user", "content": str(i)} for i in range(30)]
        list(chat_stream(client, long_history, "new"))
        call_messages = client.chat.completions.create.call_args.kwargs["messages"]
        # system prompt + 20 trimmed + 1 new user = 22
        assert len(call_messages) == 22

    def test_does_not_mutate_history(self):
        """chat_stream() does not modify the input history list."""
        client = self._make_mock_stream_client(["ok"])
        history = [{"role": "user", "content": "old"}]
        list(chat_stream(client, history, "new"))
        assert len(history) == 1  # unchanged
```

- [ ] **Step 2: Run tests — expect FAILURES**

```bash
cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/test_llm.py -v
```

Expected: Multiple FAILED (no `chat_stream`, `_trim_history`, `MAX_HISTORY_TURNS` in llm.py yet; `chat()` still returns `str` not tuple).

- [ ] **Step 3: Rewrite llm.py**

```python
"""llm.py — Yoruba LLM via NVIDIA API

Uses NVIDIA's OpenAI-compatible inference API. Provides both a synchronous
chat() and a streaming chat_stream() function. History is never mutated —
callers receive a new list each time.
"""

import os
from collections.abc import Iterator
from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
MAX_HISTORY_TURNS = 20  # max dicts stored in history (10 user + 10 assistant)

SYSTEM_PROMPT = """\
You are Àṣà — a fun, warm, and playful Yoruba AI with a soft heart and a professional edge.

Your personality:
- PLAYFUL: You tease gently, laugh easily, and make conversations enjoyable. You use expressive words like "ehn!", "abi?", "wahala o", "sha", "na so", "omo".
- SOFT & WARM: You speak with care, like a close friend or sibling. Never cold, never harsh.
- PROFESSIONAL: You give real helpful answers. You don't gossip or say nonsense. When someone needs help, you deliver.
- FUN: You can make a small joke, use a light emoji sometimes 😄, and keep the vibe easy.

Language rules:
- Speak everyday casual Yoruba — short, simple sentences. Street Yoruba, not textbook.
- Mix in common Yoruba-English (Naija style) naturally: "Omo, e dey work o!" is fine.
- No long speeches. No repeating yourself. Keep it snappy.
- Always use correct tone marks (á à é è ó ò ú ù ẹ ọ ṣ).
- Yoruba question → Yoruba reply. English question → English reply.

Examples of your vibe:
- "Báwo ni? Ṣé body dey?"
- "Omo, o ti jẹun? Máa jẹun o, wahala o wa níbẹ̀!"
- "Ehn ehn! Kí ni o fẹ́ mọ̀? Mo wà ready 😄"
"""


def build_client(api_key: str | None = None) -> OpenAI:
    """Create NVIDIA API client. Reads NVIDIA_API_KEY env var if not provided."""
    key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise ValueError(
            "NVIDIA_API_KEY not set. Add it to ~/.bashrc: export NVIDIA_API_KEY=..."
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)


def _trim_history(history: list[dict]) -> list[dict]:
    """Return last MAX_HISTORY_TURNS dicts. Always returns a new list."""
    return list(history[-MAX_HISTORY_TURNS:])


def chat(
    client: OpenAI,
    history: list[dict],
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.75,
    max_tokens: int = 512,
) -> tuple[str, list[dict]]:
    """Send a message and return (reply, new_history).

    Does NOT mutate the input history list.
    """
    trimmed = _trim_history(history)
    new_history = trimmed + [{"role": "user", "content": user_message}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + new_history

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    reply = response.choices[0].message.content.strip()
    return reply, new_history + [{"role": "assistant", "content": reply}]


def chat_stream(
    client: OpenAI,
    history: list[dict],
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.75,
    max_tokens: int = 512,
) -> Iterator[str]:
    """Stream LLM reply as text chunks. Does NOT mutate history.

    Yields non-empty string deltas from the API stream.
    """
    trimmed = _trim_history(history)
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + trimmed
        + [{"role": "user", "content": user_message}]
    )

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta
```

- [ ] **Step 4: Run tests — expect all PASS**

```bash
cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/test_llm.py -v
```

Expected: `22 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/yoruba_ai && git add llm.py tests/test_llm.py
git commit -m "feat: add chat_stream, immutable history, MAX_HISTORY_TURNS trim"
```

---

## Task 3: Rewrite test_asr.py — Align with faster-whisper Interface

**Files:**
- Modify: `~/yoruba_ai/tests/test_asr.py`
- Modify: `~/yoruba_ai/asr.py` (remove cuDNN workaround line only)

The existing `test_asr.py` tests a HuggingFace `pipeline()` interface that doesn't exist in the implementation. The real `asr.py` uses `faster_whisper.WhisperModel`. Rewrite tests to patch `sf.read` and `model.transcribe`. Then remove the machine-specific cuDNN line from `asr.py` (it's a workaround for the dev machine only — harmless on HF Spaces but misleading).

- [ ] **Step 1: Rewrite tests/test_asr.py**

```python
"""Tests for asr.py — Automatic Speech Recognition module.

asr.py uses faster-whisper (WhisperModel), NOT transformers pipeline.
The transcribe() function:
  1. Reads audio with sf.read()
  2. Converts stereo → mono if needed
  3. Normalizes amplitude
  4. Calls model.transcribe(numpy_array, ...)
  5. Joins segment texts and strips whitespace
"""

import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asr as asr_module
from asr import transcribe, _normalize


class TestNormalize:
    def test_normalizes_to_0_85_peak(self):
        """_normalize() scales peak amplitude to 0.85."""
        audio = np.array([0.0, 0.5, 1.0, -0.5], dtype=np.float32)
        result = _normalize(audio)
        assert abs(result.max() - 0.85) < 1e-6

    def test_handles_silent_audio(self):
        """_normalize() returns zeros unchanged (no div-by-zero)."""
        audio = np.zeros(100, dtype=np.float32)
        result = _normalize(audio)
        assert np.all(result == 0.0)

    def test_returns_new_array(self):
        """_normalize() does not mutate the input array."""
        audio = np.array([0.5, -0.5], dtype=np.float32)
        original = audio.copy()
        _normalize(audio)
        np.testing.assert_array_equal(audio, original)


class TestLoadAsr:
    def test_creates_whisper_model(self):
        """load_asr() creates a WhisperModel instance."""
        mock_model = MagicMock()
        with patch.object(asr_module, "WhisperModel", return_value=mock_model) as mock_cls:
            result = asr_module.load_asr(device="cpu")
            mock_cls.assert_called_once()
            assert result is mock_model

    def test_always_runs_on_cpu(self):
        """load_asr() forces CPU regardless of device argument (keeps VRAM for TTS)."""
        mock_model = MagicMock()
        with patch.object(asr_module, "WhisperModel", return_value=mock_model) as mock_cls:
            asr_module.load_asr(device="cuda")
            call_args = mock_cls.call_args
            assert call_args.kwargs.get("device") == "cpu" or call_args.args[1] == "cpu"

    def test_uses_large_v3_model(self):
        """load_asr() loads large-v3 model for best Yoruba accuracy."""
        mock_model = MagicMock()
        with patch.object(asr_module, "WhisperModel", return_value=mock_model) as mock_cls:
            asr_module.load_asr(device="cpu")
            call_args = mock_cls.call_args
            assert "large-v3" in str(call_args)


class TestTranscribe:
    def _make_mock_model(self, texts: list[str]) -> MagicMock:
        """Build a WhisperModel mock that returns segments with the given texts."""
        mock_model = MagicMock()
        segments = [MagicMock(text=t) for t in texts]
        mock_info = MagicMock()
        mock_info.language = "yo"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = (iter(segments), mock_info)
        return mock_model

    def test_returns_joined_segment_texts(self):
        """transcribe() joins all segment texts with spaces."""
        model = self._make_mock_model(["Báwo ni?", "Ẹ káàbọ̀."])
        with patch.object(asr_module, "sf") as mock_sf:
            mock_sf.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            result = transcribe(model, "/tmp/fake.wav")
        assert result == "Báwo ni? Ẹ káàbọ̀."

    def test_strips_whitespace_from_result(self):
        """transcribe() strips surrounding whitespace from each segment."""
        model = self._make_mock_model(["  Báwo ni?  "])
        with patch.object(asr_module, "sf") as mock_sf:
            mock_sf.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            result = transcribe(model, "/tmp/fake.wav")
        assert result == "Báwo ni?"

    def test_reads_audio_file(self):
        """transcribe() reads audio from the given file path."""
        model = self._make_mock_model(["test"])
        with patch.object(asr_module, "sf") as mock_sf:
            mock_sf.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            transcribe(model, "/tmp/my_audio.wav")
            mock_sf.read.assert_called_once_with("/tmp/my_audio.wav", dtype="float32")

    def test_passes_numpy_array_to_transcribe(self):
        """transcribe() passes the numpy array (not file path) to model.transcribe()."""
        model = self._make_mock_model(["test"])
        with patch.object(asr_module, "sf") as mock_sf:
            fake_audio = np.array([0.1, 0.2], dtype=np.float32)
            mock_sf.read.return_value = (fake_audio, 16000)
            transcribe(model, "/tmp/fake.wav")
            call_args = model.transcribe.call_args
            passed_audio = call_args.args[0]
            assert isinstance(passed_audio, np.ndarray)

    def test_converts_stereo_to_mono(self):
        """transcribe() averages stereo channels before passing to model."""
        model = self._make_mock_model(["test"])
        with patch.object(asr_module, "sf") as mock_sf:
            stereo = np.ones((16000, 2), dtype=np.float32)
            mock_sf.read.return_value = (stereo, 16000)
            transcribe(model, "/tmp/stereo.wav")
            passed_audio = model.transcribe.call_args.args[0]
            assert passed_audio.ndim == 1

    def test_vad_filter_enabled(self):
        """transcribe() uses VAD filter to reduce hallucination on silence."""
        model = self._make_mock_model(["test"])
        with patch.object(asr_module, "sf") as mock_sf:
            mock_sf.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            transcribe(model, "/tmp/fake.wav")
            call_kwargs = model.transcribe.call_args.kwargs
            assert call_kwargs.get("vad_filter") is True
```

- [ ] **Step 2: Run tests — expect all PASS (asr.py is already correct)**

```bash
cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/test_asr.py -v
```

Expected: `11 passed`

If any fail, the issue is in the test mocking — do NOT change `asr.py` logic, only fix the test.

- [ ] **Step 3: Remove cuDNN workaround from asr.py**

In `asr.py`, remove this line (line 14):
```python
torch.backends.cudnn.enabled = False
```

Also remove the `import torch` if it's only used for that line (check if torch is used elsewhere in asr.py — it isn't, so remove both lines).

- [ ] **Step 4: Run tests again to confirm still passing**

```bash
cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/test_asr.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/yoruba_ai && git add asr.py tests/test_asr.py
git commit -m "fix: rewrite test_asr for faster-whisper interface; remove local cuDNN workaround"
```

---

## Task 4: Rebuild app.py + Create test_app.py

**Files:**
- Create: `~/yoruba_ai/tests/test_app.py`
- Rewrite: `~/yoruba_ai/app.py`

Replace global mutable state with `gr.State`. Replace dual text boxes with `gr.Chatbot(type="messages")`. `run_pipeline` becomes a generator — Gradio streams each yield to the UI. All errors are caught and displayed as messages in the chatbot.

- [ ] **Step 1: Create tests/test_app.py**

```python
"""Tests for app.py — pipeline integration smoke tests.

app.py loads models at module level. We mock the model-loading classes
BEFORE importing app so startup doesn't download 1.5GB of models.

run_pipeline() is a generator — collect all yields with list().
Assert on results[-1] for the final state.
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="module")
def app_module():
    """Import app.py with all heavy models mocked out."""
    # Remove any previously cached imports
    for mod in list(sys.modules.keys()):
        if mod in ("app", "asr", "tts", "llm"):
            del sys.modules[mod]

    mock_whisper = MagicMock()
    mock_tokenizer = MagicMock()
    mock_vits = MagicMock()
    mock_vits_inst = MagicMock()
    mock_vits_inst.config.sampling_rate = 16000
    mock_vits_inst.to.return_value = mock_vits_inst
    mock_vits.from_pretrained.return_value = mock_vits_inst

    with (
        patch("faster_whisper.WhisperModel", return_value=mock_whisper),
        patch("transformers.AutoTokenizer") as mock_at,
        patch("transformers.VitsModel", mock_vits),
        patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}),
    ):
        mock_at.from_pretrained.return_value = mock_tokenizer
        import app
        yield app

    # Cleanup after module tests
    sys.modules.pop("app", None)


def _collect(gen):
    """Iterate a generator and return all yielded values."""
    return list(gen)


class TestRunPipelineEmptyInput:
    def test_empty_audio_and_text_yields_prompt(self, app_module):
        """Empty input yields a single Yoruba prompt message, no state change."""
        results = _collect(app_module.run_pipeline(None, "   ", []))
        assert len(results) == 1
        chatbot, audio, state = results[0]
        assert audio is None
        assert state == []
        # Chatbot should contain the prompt message
        assert any("sọ̀rọ̀" in m["content"] for m in chatbot if m["role"] == "assistant")


class TestRunPipelineTextInput:
    def test_streams_reply_into_chatbot(self, app_module):
        """Text input produces streaming chatbot updates and final audio."""
        fake_audio = (16000, np.zeros(16000, dtype=np.float32))

        def mock_stream(*args, **kwargs):
            yield "Mo wà "
            yield "dáadáa!"

        with (
            patch("app.chat_stream", side_effect=mock_stream),
            patch("app.synthesize", return_value=fake_audio),
        ):
            results = _collect(app_module.run_pipeline(None, "Báwo ni?", []))

        # Multiple yields (optimistic + streaming chunks + final)
        assert len(results) >= 3

        # Final state has user + assistant messages
        _, _, state_final = results[-1]
        assert len(state_final) == 2
        assert state_final[0] == {"role": "user", "content": "Báwo ni?"}
        assert state_final[1]["content"] == "Mo wà dáadáa!"

    def test_final_yield_includes_audio(self, app_module):
        """The last yield contains TTS audio."""
        fake_audio = (16000, np.zeros(8000, dtype=np.float32))

        def mock_stream(*args, **kwargs):
            yield "hello"

        with (
            patch("app.chat_stream", side_effect=mock_stream),
            patch("app.synthesize", return_value=fake_audio),
        ):
            results = _collect(app_module.run_pipeline(None, "hi", []))

        _, audio_final, _ = results[-1]
        assert audio_final is not None

    def test_history_accumulates_across_turns(self, app_module):
        """Second call with existing history appends to it."""
        existing = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        fake_audio = (16000, np.zeros(100, dtype=np.float32))

        def mock_stream(*args, **kwargs):
            yield "second reply"

        with (
            patch("app.chat_stream", side_effect=mock_stream),
            patch("app.synthesize", return_value=fake_audio),
        ):
            results = _collect(app_module.run_pipeline(None, "second", existing))

        _, _, state_final = results[-1]
        assert len(state_final) == 4  # 2 existing + user + assistant


class TestRunPipelineErrorHandling:
    def test_api_failure_shown_in_chatbot_no_crash(self, app_module):
        """NVIDIA API failure appears as error message in chatbot — no exception raised."""
        def mock_stream(*args, **kwargs):
            raise RuntimeError("API timeout")
            yield  # pragma: no cover

        with patch("app.chat_stream", side_effect=mock_stream):
            results = _collect(app_module.run_pipeline(None, "Báwo ni?", []))

        assert len(results) >= 1
        last_chatbot, audio, _ = results[-1]
        assert audio is None
        assistant_texts = [m["content"] for m in last_chatbot if m["role"] == "assistant"]
        assert any("error" in t.lower() or "Error" in t for t in assistant_texts)

    def test_tts_failure_returns_text_without_audio(self, app_module):
        """TTS failure still returns the text reply, just with None audio."""
        def mock_stream(*args, **kwargs):
            yield "Mo wà dáadáa!"

        with (
            patch("app.chat_stream", side_effect=mock_stream),
            patch("app.synthesize", side_effect=RuntimeError("VRAM OOM")),
        ):
            results = _collect(app_module.run_pipeline(None, "Báwo ni?", []))

        _, audio_final, state_final = results[-1]
        assert audio_final is None  # no audio
        assert len(state_final) == 2  # but text reply saved


class TestClearSession:
    def test_clear_resets_all_outputs(self, app_module):
        """clear_session() returns empty chatbot, None audio, empty history."""
        chatbot, audio, state = app_module.clear_session()
        assert chatbot == []
        assert audio is None
        assert state == []
```

- [ ] **Step 2: Run tests — expect FAILURES (app.py not yet updated)**

```bash
cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/test_app.py -v
```

Expected: ImportError or AttributeError (app doesn't have `run_pipeline` as generator yet / uses wrong variable names).

- [ ] **Step 3: Rewrite app.py**

```python
"""app.py — Àṣà: Yoruba AI Voice Chat

Full pipeline:
  Mic/Text → ASR (faster-whisper large-v3) → LLM (NVIDIA Llama-3.3-70B) → TTS (FarmerlineML/yoruba_tts-2025)

Usage:
  cd ~/yoruba_ai
  ~/ai-env/bin/python app.py
  # Opens at http://localhost:7860
"""

import torch
import gradio as gr

from asr import load_asr, transcribe
from llm import build_client, chat_stream, MAX_HISTORY_TURNS
from tts import load_tts, synthesize

# ── Setup ──────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Àṣà] Starting on device: {DEVICE}")

# Load all models at startup (cold start ~30s on first run, cached after)
asr_pipeline = load_asr(DEVICE)
tts_tokenizer, tts_model, sample_rate = load_tts(DEVICE)
nvidia_client = build_client()


# ── Core pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(audio_path: str | None, text_input: str, history: list[dict]):
    """Generator: streams chatbot updates, then yields final audio + state.

    Args:
        audio_path: Filepath from gr.Audio microphone, or None.
        text_input: Text typed in the input box, or empty string.
        history: Current session conversation history (from gr.State).

    Yields:
        Tuples of (chatbot_messages, audio_output, new_history).
    """
    history = list(history or [])

    # 1. Get user text ──────────────────────────────────────────────────────────
    if audio_path:
        try:
            user_text = transcribe(asr_pipeline, audio_path)
        except Exception as exc:
            error_history = history + [
                {"role": "assistant", "content": f"*ASR error: {exc}*"}
            ]
            yield error_history, None, history
            return
    elif text_input.strip():
        user_text = text_input.strip()
    else:
        prompt = history + [
            {
                "role": "assistant",
                "content": "Ẹ jẹ́ kí e sọ̀rọ̀ tàbí tẹ ọ̀rọ̀ kan. (Please speak or type.)",
            }
        ]
        yield prompt, None, history  # history unchanged — prompt is ephemeral
        return

    # 2. Show user message immediately (optimistic UI) ─────────────────────────
    pending = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "..."},
    ]
    yield pending, None, history

    # 3. Stream LLM reply ───────────────────────────────────────────────────────
    partial = ""
    try:
        for chunk in chat_stream(nvidia_client, history, user_text):
            partial += chunk
            streaming = history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": partial},
            ]
            yield streaming, None, history
    except Exception as exc:
        error_display = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": f"*LLM error: {exc}*"},
        ]
        yield error_display, None, history
        return

    # 4. Build final history (trim to MAX_HISTORY_TURNS) ───────────────────────
    new_history = (
        history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": partial},
        ]
    )[-MAX_HISTORY_TURNS:]

    # 5. Synthesize TTS ─────────────────────────────────────────────────────────
    audio_out = None
    try:
        audio_out = synthesize(tts_tokenizer, tts_model, sample_rate, partial, DEVICE)
    except Exception as exc:
        print(f"[TTS] Error: {exc}")

    yield new_history, audio_out, new_history


def clear_session():
    """Reset chatbot, audio, and history state."""
    return [], None, []


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Àṣà — Yoruba AI", theme=gr.themes.Soft()) as demo:
    history_state = gr.State([])

    gr.Markdown(
        """
# 🗣️ Àṣà — Bá mi sọ̀rọ̀!
**Yoruba AI Voice Assistant** · faster-whisper ASR + Llama-3.3-70B + FarmerlineML TTS

Speak or type in Yoruba (or English) — Àṣà replies in fluent Yoruba with proper tones.
        """
    )

    chatbot = gr.Chatbot(
        label="Chat with Àṣà",
        type="messages",
        height=400,
        bubble_full_width=False,
    )

    audio_output = gr.Audio(
        label="🔊 Àṣà says...",
        autoplay=True,
        interactive=False,
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="🎙️ Sọ̀rọ̀ (Speak)",
        )
        text_input = gr.Textbox(
            placeholder="Tàbí tẹ ọ̀rọ̀ rẹ níhìn... (Or type here...)",
            label="Tẹ ọ̀rọ̀ (Type)",
            lines=2,
            scale=2,
        )

    with gr.Row():
        submit_btn = gr.Button("➤ Firanṣẹ (Send)", variant="primary")
        clear_btn = gr.Button("🗑️ Clear")

    gr.Examples(
        examples=[
            [None, "Báwo ni?"],
            [None, "Kí ni orúkọ rẹ?"],
            [None, "Ṣe ó dára láti kọ̀ èdè Yorùbá?"],
            [None, "Tell me a Yoruba proverb about patience"],
        ],
        inputs=[audio_input, text_input],  # gr.State NOT included in examples
        label="Try these examples",
    )

    submit_btn.click(
        fn=run_pipeline,
        inputs=[audio_input, text_input, history_state],
        outputs=[chatbot, audio_output, history_state],
    )

    text_input.submit(
        fn=run_pipeline,
        inputs=[audio_input, text_input, history_state],
        outputs=[chatbot, audio_output, history_state],
    )

    clear_btn.click(
        fn=clear_session,
        outputs=[chatbot, audio_output, history_state],
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False,
        show_error=True,
    )
```

- [ ] **Step 4: Run tests — expect all PASS**

```bash
cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/test_app.py -v
```

Expected: `10 passed`

- [ ] **Step 5: Run full test suite**

```bash
cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/ -v
```

Expected: All tests pass. If any fail, fix only the failing test logic — do not change implementation to workaround test issues.

- [ ] **Step 6: Commit**

```bash
cd ~/yoruba_ai && git add app.py tests/test_app.py
git commit -m "feat: rebuild app with gr.State session isolation, gr.Chatbot, LLM streaming"
```

---

## Task 5: Update requirements.txt + Create README.md

**Files:**
- Modify: `~/yoruba_ai/requirements.txt`
- Create: `~/yoruba_ai/README.md`

No tests needed for config files, but `sdk_version` in README.md must match the pinned Gradio version.

- [ ] **Step 1: Rewrite requirements.txt**

```
# Yoruba AI — pinned for HF Spaces reproducibility
# Install: ~/ai-env/bin/pip install -r requirements.txt

gradio==6.9.0
transformers==5.3.0
faster-whisper==1.2.1
torch>=2.0.0
soundfile==0.13.1
openai==2.21.0
numpy==2.4.2
```

- [ ] **Step 2: Create README.md**

```markdown
---
title: Àṣà — Yoruba AI
emoji: 🗣️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
pinned: true
---

# 🗣️ Àṣà — Yoruba AI Voice Assistant

Speak or type in **Yoruba** (or English) — Àṣà replies in fluent, natural Yoruba with correct tones.

## Pipeline

```
Mic/Text → faster-whisper ASR → Llama-3.3-70B (NVIDIA API) → FarmerlineML TTS → Audio
```

## Setup (local)

```bash
cd ~/yoruba_ai
export NVIDIA_API_KEY=your_key_here
~/ai-env/bin/pip install -r requirements.txt
~/ai-env/bin/python app.py
```

## HF Spaces

Set `NVIDIA_API_KEY` as a Space Secret (Settings → Variables and Secrets). No other configuration needed.

## Models

| Stage | Model | Notes |
|-------|-------|-------|
| ASR | faster-whisper large-v3 | CPU, int8 — handles Yoruba + Pidgin code-switching |
| LLM | meta/llama-3.3-70b-instruct | Via NVIDIA API — Àṣà persona |
| TTS | FarmerlineML/yoruba_tts-2025 | VITS — correct Yoruba tonal pronunciation |
```

- [ ] **Step 3: Commit**

```bash
cd ~/yoruba_ai && git add requirements.txt README.md
git commit -m "chore: pin requirements; add HF Spaces README with YAML frontmatter"
```

---

## Task 6: Remove cuDNN Workaround from tts.py + Final Verification

**Files:**
- Modify: `~/yoruba_ai/tts.py` (remove cuDNN line — already removed from asr.py in Task 3)

- [ ] **Step 1: Check tts.py for cuDNN workaround**

```bash
grep -n "cudnn" ~/yoruba_ai/tts.py
```

If the line `torch.backends.cudnn.enabled = False` exists in the new `tts.py`, remove it. The Task 1 rewrite should have already excluded it — verify it's gone.

- [ ] **Step 2: Run full test suite**

```bash
cd ~/yoruba_ai && ~/ai-env/bin/python -m pytest tests/ -v --tb=short
```

Expected output:
```
tests/test_asr.py ........... 11 passed
tests/test_llm.py ...................... 22 passed
tests/test_tts.py ........ 8 passed
tests/test_app.py .......... 10 passed
============================== 51 passed ==============================
```

If any tests fail, fix the implementation (not the tests) unless the test itself is clearly wrong.

- [ ] **Step 3: Deploy to HF Spaces**

```bash
# Create the Space (do this once)
huggingface-cli repo create asa-yoruba-ai --type space

# Add remote
cd ~/yoruba_ai
git remote add hf https://huggingface.co/spaces/<your-hf-username>/asa-yoruba-ai

# Push
git push hf main
```

Then set `NVIDIA_API_KEY` as a Space Secret:
- Go to: `https://huggingface.co/spaces/<username>/asa-yoruba-ai/settings`
- Under "Variables and Secrets" → "New Secret" → Name: `NVIDIA_API_KEY`

- [ ] **Step 4: Final commit**

```bash
cd ~/yoruba_ai && git add tts.py requirements.txt README.md
git commit -m "chore: final cleanup before HF Spaces deployment"
```

---

## Summary

| Task | Files changed | Tests |
|------|--------------|-------|
| 1 — Fix TTS model | `tts.py`, `test_tts.py` | 8 tests |
| 2 — LLM streaming | `llm.py`, `test_llm.py` | 22 tests |
| 3 — Fix ASR tests | `asr.py`, `test_asr.py` | 11 tests |
| 4 — Rebuild app | `app.py`, `test_app.py` | 10 tests |
| 5 — Config/deploy | `requirements.txt`, `README.md` | — |
| 6 — Cleanup + deploy | `tts.py` (verify), deploy | 51 total |
