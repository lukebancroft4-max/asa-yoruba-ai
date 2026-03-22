"""Tests for app.py — pipeline integration smoke tests.

app.py loads models at module level. We mock model-loading classes
BEFORE importing app so startup doesn't download gigabytes of models.

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
    for mod in list(sys.modules.keys()):
        if mod in ("app", "asr", "tts", "llm"):
            del sys.modules[mod]

    mock_whisper_inst = MagicMock()
    mock_tts_preprocessor = MagicMock()
    mock_vits_inst = MagicMock()
    mock_vits_inst.config.sampling_rate = 16000
    mock_vits_inst.to.return_value = mock_vits_inst

    with (
        patch("faster_whisper.WhisperModel", return_value=mock_whisper_inst),
        patch("transformers.AutoTokenizer") as mock_at,
        patch("transformers.AutoProcessor") as mock_ap,
        patch("transformers.VitsModel") as mock_vm,
        patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}),
    ):
        mock_at.from_pretrained.return_value = mock_tts_preprocessor
        mock_ap.from_pretrained.return_value = mock_tts_preprocessor
        mock_vm.from_pretrained.return_value = mock_vits_inst
        import app
        yield app

    sys.modules.pop("app", None)


def _collect(gen):
    """Iterate a generator and return all yielded values."""
    return list(gen)


class TestRunPipelineEmptyInput:
    def test_empty_audio_and_text_yields_prompt(self, app_module):
        """Empty input yields a Yoruba prompt message, history unchanged."""
        results = _collect(app_module.run_pipeline(None, "   ", []))
        assert len(results) == 1
        chatbot, audio, state = results[0]
        assert audio is None
        assert state == []
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

        assert len(results) >= 2

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
        assert len(state_final) == 4


class TestRunPipelineErrorHandling:
    def test_api_failure_shown_in_chatbot_no_crash(self, app_module):
        """NVIDIA API failure appears as error message — no exception raised."""
        def mock_stream(*args, **kwargs):
            raise RuntimeError("API timeout")
            yield  # make it a generator

        with patch("app.chat_stream", side_effect=mock_stream):
            results = _collect(app_module.run_pipeline(None, "Báwo ni?", []))

        assert len(results) >= 1
        last_chatbot, audio, _ = results[-1]
        assert audio is None
        assistant_texts = [m["content"] for m in last_chatbot if m["role"] == "assistant"]
        assert any("error" in t.lower() or "Error" in t for t in assistant_texts)

    def test_tts_failure_returns_text_without_audio(self, app_module):
        """TTS failure still returns text reply, just with None audio."""
        def mock_stream(*args, **kwargs):
            yield "Mo wà dáadáa!"

        with (
            patch("app.chat_stream", side_effect=mock_stream),
            patch("app.synthesize", side_effect=RuntimeError("VRAM OOM")),
        ):
            results = _collect(app_module.run_pipeline(None, "Báwo ni?", []))

        _, audio_final, state_final = results[-1]
        assert audio_final is None
        assert len(state_final) == 2


class TestClearSession:
    def test_clear_resets_all_outputs(self, app_module):
        """clear_session() returns empty chatbot, None audio, empty history."""
        chatbot, audio, state = app_module.clear_session()
        assert chatbot == []
        assert audio is None
        assert state == []
