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
        assert len(history) == original_len

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
        long_history = [{"role": "user", "content": str(i)} for i in range(30)]
        chat(client, long_history, "new message")
        call_messages = client.chat.completions.create.call_args.kwargs["messages"]
        # system prompt + 20 trimmed + 1 new user = 22
        assert len(call_messages) == 22


class TestChatStream:
    def _make_mock_stream_client(self, chunks: list):
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

    def test_skips_empty_and_none_chunks(self):
        """chat_stream() skips chunks where delta.content is None or empty."""
        client = self._make_mock_stream_client(["hello", None, "", "world"])
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
        assert len(history) == 1
