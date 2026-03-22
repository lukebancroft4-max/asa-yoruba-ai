"""app.py — Àṣà: Yoruba AI Voice Chat

Full pipeline:
  Mic/Text → ASR (NCAIR1/Yoruba-ASR) → LLM (NVIDIA API Llama-3.3-70B) → TTS (MMS-yor)

Usage:
  cd ~/yoruba_ai
  ~/ai-env/bin/python app.py
  # Opens at http://localhost:7860
"""

import torch

import gradio as gr

from asr import load_asr, transcribe
from llm import build_client, chat
from tts import load_tts, synthesize

# ── Setup ──────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Àṣà] Starting on device: {DEVICE}")

# Load all models at startup (cold start ~30s, then instant)
asr_pipeline = load_asr(DEVICE)
tts_processor, tts_model, sample_rate = load_tts(DEVICE)
nvidia_client = build_client()

# Per-session conversation history (resets on "Clear")
conversation_history: list[dict] = []


# ── Core pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(audio_path: str | None, text_input: str):
    """End-to-end voice/text → voice/text pipeline.

    Args:
        audio_path: Filepath from gr.Audio microphone recording, or None.
        text_input: Text typed in the input box, or empty string.

    Returns:
        Tuple: (audio_output, user_text_display, assistant_text_display)
    """
    # 1. Get user text
    if audio_path:
        user_text = transcribe(asr_pipeline, audio_path)
    elif text_input.strip():
        user_text = text_input.strip()
    else:
        return None, "", "Ẹ jẹ́ kí e sọ̀rọ̀ tàbí tẹ ọ̀rọ̀ kan. (Please speak or type.)"

    # 2. LLM reply
    reply_text = chat(nvidia_client, conversation_history, user_text)

    # 3. Synthesize reply
    audio_out = synthesize(tts_processor, tts_model, sample_rate, reply_text, DEVICE)

    return audio_out, user_text, reply_text


def clear_session():
    conversation_history.clear()
    return None, "", ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Àṣà — Yoruba AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🗣️ Àṣà — Bá mi sọ̀rọ̀!
**Yoruba AI Voice Assistant** · Powered by NCAIR1 ASR + Llama-3.3-70B + Meta MMS-TTS

Speak or type in Yoruba (or English) — Àṣà replies in fluent Yoruba with proper tones.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎙️ Sọ̀rọ̀ (Speak)",
            )
            text_input = gr.Textbox(
                placeholder="Tàbí tẹ ọ̀rọ̀ rẹ níhìn... (Or type here...)",
                label="Tẹ ọ̀rọ̀ (Type)",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("➤ Firanṣẹ (Send)", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="🔊 Àṣà says...",
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
        inputs=[audio_input, text_input],
        label="Try these examples",
    )

    submit_btn.click(
        fn=run_pipeline,
        inputs=[audio_input, text_input],
        outputs=[audio_output, user_text_out, reply_text_out],
    )

    # Also trigger on Enter in text box
    text_input.submit(
        fn=run_pipeline,
        inputs=[audio_input, text_input],
        outputs=[audio_output, user_text_out, reply_text_out],
    )

    clear_btn.click(
        fn=clear_session,
        outputs=[audio_output, user_text_out, reply_text_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False,
        show_error=True,
    )
