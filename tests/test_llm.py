"""Tests for llm.py — LLM module."""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import llm as llm_module
from llm import chat, SYSTEM_PROMPT


class TestSystemPrompt:
    def test_includes_yoruba_pronunciation_basics(self):
        """SYSTEM_PROMPT includes core pronunciation-coaching guidance."""
        assert "Yoruba pronunciation coaching (important):" in SYSTEM_PROMPT
        assert "Yoruba is tonal." in SYSTEM_PROMPT
        assert "ẹ = open \"e\" sound" in SYSTEM_PROMPT
        assert "ọ = open \"o\" sound" in SYSTEM_PROMPT
        assert "ṣ = \"sh\" sound" in SYSTEM_PROMPT
        assert "gb = pronounced together in one beat" in SYSTEM_PROMPT


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
        """build_client() raises ValueError if no key is found anywhere."""
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
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["api_key"] == "explicit-key"


class TestChat:
    def _make_mock_client(self, reply_text: str):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = reply_text
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_appends_user_message_to_history(self):
        """chat() adds user message to history before calling API."""
        client = self._make_mock_client("Ẹ káàbọ̀!")
        history = []
        chat(client, history, "Báwo ni?")
        assert history[0] == {"role": "user", "content": "Báwo ni?"}

    def test_appends_assistant_reply_to_history(self):
        """chat() adds assistant reply to history after API call."""
        client = self._make_mock_client("Mo wà dáadáa!")
        history = []
        chat(client, history, "Báwo ni?")
        assert history[-1] == {"role": "assistant", "content": "Mo wà dáadáa!"}

    def test_returns_stripped_reply(self):
        """chat() strips whitespace from the reply."""
        client = self._make_mock_client("  Ẹ káàbọ̀!  ")
        history = []
        result = chat(client, history, "hello")
        assert result == "Ẹ káàbọ̀!"

    def test_includes_system_prompt_as_first_message(self):
        """chat() always prepends system prompt to the messages sent to API."""
        client = self._make_mock_client("reply")
        history = []
        chat(client, history, "test")
        call_messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert call_messages[0] == {"role": "system", "content": SYSTEM_PROMPT}

    def test_multi_turn_history_grows(self):
        """chat() accumulates user+assistant pairs across turns."""
        client = self._make_mock_client("reply")
        history = []
        chat(client, history, "first")
        chat(client, history, "second")
        assert len(history) == 4  # user + assistant, twice

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
