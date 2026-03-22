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
