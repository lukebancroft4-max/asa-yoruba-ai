"""llm.py — Yoruba LLM via NVIDIA API

Uses NVIDIA's OpenAI-compatible inference API so we don't burn local VRAM
on a full 8B model. The default model is Llama-3.3-70B for quality;
swap to llama-3.1-8b-instruct for faster/cheaper responses.
"""

import os
from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

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

Yoruba pronunciation coaching (important):
- When a user asks how to pronounce a Yoruba word/sentence, switch to coach mode.
- Teach the basics first: Yoruba is tonal.
  - High tone: acute mark (á é í ó ú)
  - Mid tone: usually unmarked (a e i o u)
  - Low tone: grave mark (à è ì ò ù)
- Explain key letters clearly:
  - ẹ = open "e" sound (like "e" in "bet")
  - ọ = open "o" sound (like "o" in "thought")
  - ṣ = "sh" sound
  - gb = pronounced together in one beat (not separate "guh-buh")
- Break words into syllables with hyphens when teaching pronunciation.
- Keep explanations practical, short, and beginner-friendly. Give 1-3 examples max.
- If helpful, provide a simple English approximation, but prioritize correct Yoruba tone marks.
- Correct wrong tone marks gently and show the corrected form.

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


def chat(
    client: OpenAI,
    history: list[dict],
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.75,
    max_tokens: int = 512,
) -> str:
    """Send a message and return the assistant reply.

    Args:
        client: NVIDIA OpenAI client.
        history: List of {"role": ..., "content": ...} dicts (mutated in-place).
        user_message: The user's text input.
        model: NVIDIA-hosted model ID.
        temperature: Sampling temperature.
        max_tokens: Max reply tokens.

    Returns:
        Assistant reply text.
    """
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    reply = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply})
    return reply
