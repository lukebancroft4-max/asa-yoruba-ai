"""app.py — Àṣà: Yoruba AI Voice Chat

Full pipeline:
  Mic/Text → ASR (faster-whisper) → LLM (NVIDIA API Llama-3.3-70B) → TTS (dual engine)

Modes:
  Chat   — Talk to Àṣà (ASR → LLM → TTS)
  Speak  — Type English/Yoruba → hear it in Yoruba (translate → TTS, no LLM)

Usage:
  cd ~/yoruba_ai
  ~/ai-env/bin/python app.py
  # Opens at http://localhost:7860
"""

import torch

import gradio as gr

from asr import load_asr, transcribe
from llm import build_client, chat
from translate import translate_to_yoruba
from tts import synthesize
from tts_engines import load_tts, ENGINES, DEFAULT_ENGINE

# ── Setup ──────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Àṣà] Starting on device: {DEVICE}")

# Load default TTS + ASR + LLM at startup
asr_pipeline = load_asr(DEVICE)
tts_preprocessor, tts_model, tts_sample_rate = load_tts(DEVICE, engine=DEFAULT_ENGINE)
nvidia_client = build_client()

# Track current engine so we only reload when it changes
_current_engine = DEFAULT_ENGINE

# Per-session conversation history (resets on "Clear")
conversation_history: list[dict] = []


# ── Engine hot-swap ───────────────────────────────────────────────────────────
def _ensure_engine(engine: str):
    """Reload TTS model only if engine changed."""
    global _current_engine, tts_preprocessor, tts_model, tts_sample_rate
    if engine != _current_engine:
        tts_preprocessor, tts_model, tts_sample_rate = load_tts(DEVICE, engine=engine)
        _current_engine = engine


# ── Chat pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(audio_path: str | None, text_input: str, engine: str):
    """End-to-end voice/text → voice/text pipeline (ASR → LLM → TTS)."""
    _ensure_engine(engine)

    if audio_path:
        user_text = transcribe(asr_pipeline, audio_path)
    elif text_input.strip():
        user_text = text_input.strip()
    else:
        return None, "", "Ẹ jẹ́ kí e sọ̀rọ̀ tàbí tẹ ọ̀rọ̀ kan. (Please speak or type.)"

    reply_text = chat(nvidia_client, conversation_history, user_text)
    audio_out = synthesize(tts_preprocessor, tts_model, tts_sample_rate, reply_text, DEVICE)

    return audio_out, user_text, reply_text


# ── Speak pipeline (translate → TTS, no LLM) ─────────────────────────────────
def run_speak(text_input: str, lang: str, engine: str):
    """Direct text-to-speech: optionally translate English → Yoruba, then TTS."""
    _ensure_engine(engine)

    if not text_input.strip():
        return None, ""

    text = text_input.strip()

    if lang == "English (auto-translate)":
        yoruba = translate_to_yoruba(text)
        display = f"{text}\n→ {yoruba}"
    else:
        yoruba = text
        display = yoruba

    audio_out = synthesize(tts_preprocessor, tts_model, tts_sample_rate, yoruba, DEVICE)
    return audio_out, display


def clear_session():
    conversation_history.clear()
    return None, "", ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Àṣà — Yoruba AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# Àṣà — Bá mi sọ̀rọ̀!
**Yoruba AI Voice Assistant** · ASR + Llama-3.3-70B + Dual TTS + NLLB Translation
        """
    )

    engine_selector = gr.Dropdown(
        choices=list(ENGINES.keys()),
        value=DEFAULT_ENGINE,
        label="TTS Engine",
    )

    with gr.Tabs():
        # ── Tab 1: Chat ──
        with gr.Tab("Chat with Àṣà"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Sọ̀rọ̀ (Speak)",
                    )
                    chat_text_input = gr.Textbox(
                        placeholder="Tàbí tẹ ọ̀rọ̀ rẹ níhìn... (Or type here...)",
                        label="Tẹ ọ̀rọ̀ (Type)",
                        lines=2,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("Firanṣẹ (Send)", variant="primary")
                        clear_btn = gr.Button("Clear")

                with gr.Column(scale=1):
                    chat_audio_out = gr.Audio(
                        label="Àṣà says...",
                        autoplay=True,
                        interactive=False,
                    )
                    user_text_out = gr.Textbox(
                        label="You said:", interactive=False, lines=2
                    )
                    reply_text_out = gr.Textbox(
                        label="Àṣà replied:", interactive=False, lines=4
                    )

            gr.Examples(
                examples=[
                    [None, "Báwo ni?"],
                    [None, "Kí ni orúkọ rẹ?"],
                    [None, "Ṣe ó dára láti kọ̀ èdè Yorùbá?"],
                    [None, "Tell me a Yoruba proverb about patience"],
                ],
                inputs=[audio_input, chat_text_input],
                label="Try these examples",
            )

            submit_btn.click(
                fn=run_pipeline,
                inputs=[audio_input, chat_text_input, engine_selector],
                outputs=[chat_audio_out, user_text_out, reply_text_out],
            )
            chat_text_input.submit(
                fn=run_pipeline,
                inputs=[audio_input, chat_text_input, engine_selector],
                outputs=[chat_audio_out, user_text_out, reply_text_out],
            )
            clear_btn.click(
                fn=clear_session,
                outputs=[chat_audio_out, user_text_out, reply_text_out],
            )

        # ── Tab 2: Speak ──
        with gr.Tab("Speak (TTS only)"):
            gr.Markdown("Type English or Yoruba text and hear it spoken in Yoruba.")
            with gr.Row():
                with gr.Column(scale=1):
                    speak_text = gr.Textbox(
                        placeholder="Type text to speak...",
                        label="Text",
                        lines=3,
                    )
                    speak_lang = gr.Radio(
                        choices=["English (auto-translate)", "Yoruba (direct)"],
                        value="English (auto-translate)",
                        label="Input language",
                    )
                    speak_btn = gr.Button("Sọ (Speak)", variant="primary")

                with gr.Column(scale=1):
                    speak_audio_out = gr.Audio(
                        label="Audio output",
                        autoplay=True,
                        interactive=False,
                    )
                    speak_display = gr.Textbox(
                        label="Translation", interactive=False, lines=3
                    )

            gr.Examples(
                examples=[
                    ["Good morning, how are you?"],
                    ["The market opens at 8 o'clock"],
                    ["I love learning Yoruba"],
                ],
                inputs=[speak_text],
                label="English examples (auto-translated)",
            )

            speak_btn.click(
                fn=run_speak,
                inputs=[speak_text, speak_lang, engine_selector],
                outputs=[speak_audio_out, speak_display],
            )
            speak_text.submit(
                fn=run_speak,
                inputs=[speak_text, speak_lang, engine_selector],
                outputs=[speak_audio_out, speak_display],
            )

if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False,
        show_error=True,
    )
