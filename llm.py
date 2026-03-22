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
MAX_HISTORY_TURNS = 20  # max dicts in history (10 user + 10 assistant pairs)

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
    """Send a message and return (reply, new_history). Does NOT mutate history."""
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
