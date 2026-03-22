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

asr_pipeline = load_asr(DEVICE)
tts_preprocessor, tts_model, sample_rate = load_tts(DEVICE)
nvidia_client = build_client()


# ── Core pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(audio_path: str | None, text_input: str, history: list[dict]):
    """Generator: streams chatbot updates, then yields final audio + state.

    Yields:
        Tuples of (chatbot_messages, audio_output, new_history).
    """
    history = list(history or [])

    # 1. Get user text
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
        yield prompt, None, history
        return

    # 2. Show user message immediately
    pending = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "..."},
    ]
    yield pending, None, history

    # 3. Stream LLM reply
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

    # 4. Build final history (trimmed to even boundary so we never split a user+assistant pair)
    _trim_count = MAX_HISTORY_TURNS - (MAX_HISTORY_TURNS % 2)
    new_history = (
        history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": partial},
        ]
    )[-_trim_count:]

    # 5. Synthesize TTS
    audio_out = None
    try:
        audio_out = synthesize(tts_preprocessor, tts_model, sample_rate, partial, DEVICE)
    except Exception as exc:
        print(f"[TTS] Error: {exc}")

    yield new_history, audio_out, new_history


def clear_session():
    """Reset chatbot, audio, and history state."""
    return [], None, []


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Àṣà — Yoruba AI") as demo:
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
        height=400,
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
        inputs=[audio_input, text_input],
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
        theme=gr.themes.Soft(),
    )
