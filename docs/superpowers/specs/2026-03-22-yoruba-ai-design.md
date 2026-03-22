# Àṣà — Yoruba AI: Completion & HF Spaces Deployment

**Date:** 2026-03-22
**Status:** Approved
**Goal:** Complete, fix, and deploy the Yoruba AI voice assistant to Hugging Face Spaces as a shareable public app.

---

## 1. Context & Current State

`~/yoruba_ai/` is a Gradio voice chat app with a 3-stage pipeline:
- **ASR:** faster-whisper large-v3 (CPU, int8) — transcribes Yoruba / Pidgin code-switching
- **LLM:** NVIDIA API (Llama-3.3-70B) — Àṣà persona with Yoruba tone marks
- **TTS:** Meta MMS-TTS-yor (VITS) — GPU inference, **known issue: wrong Yoruba pronunciation / tone marks not honoured**

### Known Issues
| Issue | Severity |
|-------|----------|
| `test_asr.py` tests `pipeline()` / `LyngualLabs/whisper-small-yoruba` — wrong interface | Critical |
| Global `conversation_history` — shared across all users on deployment | Critical |
| `faster-whisper` missing from `requirements.txt` | High |
| No LLM history trimming — context grows unbounded | High |
| No error handling in UI (API failures crash silently) | High |
| No `gr.Chatbot` — no visible history panel | Medium |
| No streaming — blank wait during LLM generation | Medium |
| No HF Spaces `README.md` | Deployment blocker |
| MMS-TTS-yor wrong pronunciation / tone marks | High — replace with `FarmerlineML/yoruba_tts-2025` + NFC normalization |

---

## 2. Architecture

```
~/yoruba_ai/
├── app.py             # Gradio UI — gr.Chatbot + gr.State (per-user session)
├── asr.py             # faster-whisper large-v3, CPU int8 (unchanged logic, aligned API)
├── llm.py             # NVIDIA API — streaming generator + history trimming
├── tts.py             # Meta MMS-TTS-yor VITS (no changes)
├── requirements.txt   # + faster-whisper, pinned versions
├── README.md          # HF Spaces YAML frontmatter
└── tests/
    ├── test_asr.py    # Rewritten to match faster-whisper interface
    ├── test_llm.py    # Extended: streaming + trimming tests
    ├── test_tts.py    # Unchanged
    └── test_app.py    # New: full pipeline smoke test (all modules mocked)
```

### Data Flow

```
User mic/text
      │
      ▼
[ASR] faster-whisper transcribe()
      │  user_text: str
      ▼
[gr.State] history: list[dict]  ──── trim to last 20 turns ────
      │                                                         │
      ▼                                                         │
[LLM] NVIDIA API stream()  ◄──── system prompt + history ──────┘
      │  reply chunks → gr.Chatbot (live token stream)
      │  full_reply: str (after stream complete)
      ▼
[TTS] MMS-TTS-yor synthesize()
      │  (sample_rate, np.ndarray)
      ▼
gr.Audio (autoplay)
```

---

## 3. Component Designs

### 3.1 `asr.py` — No Logic Changes, Interface Alignment

The current implementation is correct (faster-whisper, CPU, int8, VAD filter). Only `test_asr.py` needs rewriting to test the actual `WhisperModel.transcribe()` interface.

**Public API (unchanged):**
```python
load_asr(device: str) -> WhisperModel
transcribe(model: WhisperModel, audio_path: str) -> str
```

**Test rewrite guidance:** `asr.py` reads audio with `sf.read()`, normalizes the numpy array, then passes the array directly to `model.transcribe()` — it does NOT pass a file path. New `test_asr.py` must:
- Patch `sf.read` to return a fake numpy array + sample rate
- Patch `WhisperModel.transcribe` (or mock the whole `WhisperModel` instance) to return `(segments_iter, info)`
- Assert that `transcribe()` returns joined, stripped segment text

**Note:** The `torch.backends.cudnn.enabled = False` line in `asr.py` and `tts.py` is a machine-specific workaround for a broken local cuDNN (v9.0.0.312). Remove it before pushing to HF Spaces — it is harmless but misleading in a public deployment.

### 3.2 `llm.py` — Streaming + History Trimming

Add two new capabilities:
- `chat_stream()` — yields reply chunks as a generator for Gradio streaming; calls `_trim_history` internally before building messages
- History trimming: `MAX_HISTORY_TURNS = 20` means **20 total dicts** (10 user + 10 assistant pairs); implemented as `history[-20:]`

**Mutation rule:** `chat()` and `chat_stream()` must NOT mutate the history list in-place. They return `(reply, new_history)` where `new_history` is a new list. The caller (`app.py`) stores it back into `gr.State`.

**History trimming ownership:** `_trim_history` is called internally by `chat_stream()` (and `chat()`) before building the messages list. `app.py` never calls it directly.

**Public API:**
```python
build_client(api_key: str | None) -> OpenAI
chat(client, history, user_message, ...) -> tuple[str, list[dict]]         # returns (reply, new_history)
chat_stream(client, history, user_message, ...) -> Iterator[tuple[str, list[dict]]]  # yields (partial_reply, new_history)
```

**Trim boundary:** `MAX_HISTORY_TURNS = 20` (20 dicts = 10 pairs). Slice: `history[-MAX_HISTORY_TURNS:]` applied before prepending the system prompt. The system prompt is never stored in history.

### 3.3 `tts.py` — Replace MMS with FarmerlineML + NFC Normalization

Switch TTS model from `facebook/mms-tts-yor` to `FarmerlineML/yoruba_tts-2025`. This model uses `AutoTokenizer` (not `AutoProcessor`) and is specifically trained for Yoruba pronunciation. Already used in `~/yoruba-tts/speak.py`.

Add `unicodedata.normalize("NFC", text)` before tokenization to ensure consistent tone mark encoding regardless of how the LLM encodes diacritics.

**Public API (unchanged signatures):**
```python
load_tts(device: str) -> tuple[AutoTokenizer, VitsModel, int]
synthesize(tokenizer, model, sample_rate, text, device) -> tuple[int, np.ndarray]
```

**Key changes inside `synthesize()`:**
```python
import unicodedata
text = unicodedata.normalize("NFC", text)   # fix diacritic encoding
inputs = tokenizer(text, return_tensors="pt").to(device)  # tokenizer, not processor
```

### 3.4 `app.py` — Session Isolation + Streaming UI

Replace global `conversation_history` with `gr.State`. Replace dual text boxes with `gr.Chatbot`. Wire up streaming via a generator function.

**Key changes:**
- `run_pipeline(audio, text, history)` becomes a generator that yields `(chatbot_pairs, audio, history)` incrementally — the final yield contains the complete audio
- History passed in/out via `gr.State` — never shared across users; `gr.State` is the third input AND an output of `run_pipeline`
- All errors caught and displayed as italic messages in the chatbot panel

**gr.Examples wiring:** `gr.Examples` inputs stay as `[audio_input, text_input]` only — `gr.State` is never included in examples. Clicking an example populates only the two visible inputs; state carries over from the current session.

### 3.4 `README.md` — HF Spaces Config

Required YAML frontmatter for HF Spaces:
```yaml
---
title: Àṣà — Yoruba AI
emoji: 🗣️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: true
---
```

**Note:** `sdk_version` must match the Gradio version pinned in `requirements.txt`. If `requirements.txt` pins `gradio==5.29.0`, the YAML must say `sdk_version: "5.29.0"`. Keep these in sync — a mismatch causes HF Spaces build failure.

`NVIDIA_API_KEY` set as a Space Secret (never in code).

---

## 4. UI Layout

```
┌─────────────────────────────────────────────────────┐
│  🗣️ Àṣà — Bá mi sọ̀rọ̀!                            │
│  Yoruba AI · ASR + Llama-3.3-70B + MMS-TTS          │
├─────────────────────────────────────────────────────┤
│  ┌─ gr.Chatbot (scrollable, bubble style) ─────────┐ │
│  │  You: Báwo ni?                                   │ │
│  │  Àṣà: Mo wà dáadáa! Ṣé body dey? 😄            │ │
│  └──────────────────────────────────────────────────┘ │
│  ┌─ Audio output (autoplay) ──────────────────────┐  │
│  │  🔊 [audio player]                              │  │
│  └─────────────────────────────────────────────────┘ │
│  ┌─ Input ──────────────────────────────────────────┐ │
│  │  🎙️ [Mic recorder]  📝 [Type here...]            │ │
│  │         [➤ Send]      [🗑️ Clear]                 │ │
│  └──────────────────────────────────────────────────┘ │
│  Examples: Báwo ni? | Kí ni orúkọ rẹ? | ...          │
└─────────────────────────────────────────────────────┘
```

---

## 5. Error Handling

| Failure Point | Behaviour |
|---------------|-----------|
| Empty mic recording / transcription | Display `"Ẹ jẹ́ kí e sọ̀rọ̀ tàbí tẹ ọ̀rọ̀ kan."` in chat |
| NVIDIA API failure | Display error message in chat (red italic), skip TTS, return `None` audio |
| TTS synthesis failure | Display text reply in chat, return `None` audio with warning |
| No NVIDIA_API_KEY at startup | Raise `ValueError` immediately with clear message (fail-fast) |
| History > 20 turns | Silently trim oldest pairs before each LLM call |

---

## 6. Testing Plan

| File | Coverage Target | Key Tests |
|------|----------------|-----------|
| `test_asr.py` | Rewritten | `load_asr()` returns `WhisperModel`; `transcribe()` calls `model.transcribe()` with VAD; normalizes audio; returns stripped string |
| `test_llm.py` | Extended | Existing 7 tests + `chat_stream()` yields chunks + `_trim_history()` at 20 turns |
| `test_tts.py` | Updated | Swap `AutoProcessor` → `AutoTokenizer`, `VitsModel` mocks updated; add NFC normalization test |
| `test_app.py` | New | Full pipeline with mocked ASR/LLM/TTS; error path for empty input; error path for API failure. **Note:** `run_pipeline` is a generator — tests must iterate it: `results = list(run_pipeline(...))` and assert on `results[-1]` for the final state. |

Target: **80%+ coverage** across all modules.

---

## 7. Deployment

**Pre-deploy checklist:**
- [ ] Remove `torch.backends.cudnn.enabled = False` from `asr.py` and `tts.py` (machine-specific workaround)
- [ ] Verify `sdk_version` in `README.md` matches pinned Gradio in `requirements.txt`
- [ ] All tests pass: `~/ai-env/bin/python -m pytest tests/ -v`

**HF Spaces free tier runs CPU-only.** `torch.cuda.is_available()` returns `False`, so `DEVICE = "cpu"`. TTS inference on CPU takes ~5–15 seconds per utterance — acceptable for a demo. ASR already runs on CPU.

**Steps:**
1. Create the Space: `huggingface-cli repo create asa-yoruba-ai --type space`
   (SDK is inferred from `README.md` frontmatter — no `--sdk` flag needed)
2. Set `NVIDIA_API_KEY` as a Space Secret via HF web UI (Settings → Variables and Secrets)
3. Push code: `git remote add hf https://huggingface.co/spaces/<username>/asa-yoruba-ai && git push hf main`
4. Space auto-builds and serves at `https://huggingface.co/spaces/<username>/asa-yoruba-ai`

No Docker, no VPS, no ongoing cost.

---

## 8. Out of Scope

- Multi-language support beyond Yoruba/English/Pidgin
- Conversation export / download
- User authentication
- Analytics / usage tracking
- Voice cloning / custom TTS voice
