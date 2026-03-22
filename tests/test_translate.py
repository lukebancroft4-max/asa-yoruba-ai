"""Tests for translate.py — NLLB-200 translation module."""

import os
import sys
import torch
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import translate as translate_module
from translate import translate_to_yoruba, load_translator


class TestLoadTranslator:
    def test_returns_model_and_tokenizer(self):
        """load_translator() returns (model, tokenizer) tuple."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Reset global state
        translate_module._model = None
        translate_module._tokenizer = None

        with (
            patch.object(translate_module, "AutoModelForSeq2SeqLM") as mock_seq2seq,
            patch.object(translate_module, "AutoTokenizer") as mock_at,
        ):
            mock_seq2seq.from_pretrained.return_value = mock_model
            mock_at.from_pretrained.return_value = mock_tokenizer

            model, tokenizer = load_translator()

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        mock_model.eval.assert_called_once()

        # Cleanup
        translate_module._model = None
        translate_module._tokenizer = None

    def test_caches_after_first_load(self):
        """load_translator() only loads once (global cache)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        translate_module._model = None
        translate_module._tokenizer = None

        with (
            patch.object(translate_module, "AutoModelForSeq2SeqLM") as mock_seq2seq,
            patch.object(translate_module, "AutoTokenizer") as mock_at,
        ):
            mock_seq2seq.from_pretrained.return_value = mock_model
            mock_at.from_pretrained.return_value = mock_tokenizer

            load_translator()
            load_translator()  # second call

            # Only loaded once
            assert mock_seq2seq.from_pretrained.call_count == 1

        translate_module._model = None
        translate_module._tokenizer = None

    def test_loads_correct_model_id(self):
        """load_translator() loads facebook/nllb-200-distilled-600M."""
        translate_module._model = None
        translate_module._tokenizer = None

        with (
            patch.object(translate_module, "AutoModelForSeq2SeqLM") as mock_seq2seq,
            patch.object(translate_module, "AutoTokenizer") as mock_at,
        ):
            mock_seq2seq.from_pretrained.return_value = MagicMock()
            mock_at.from_pretrained.return_value = MagicMock()

            load_translator()

            mock_seq2seq.from_pretrained.assert_called_once_with(
                "facebook/nllb-200-distilled-600M"
            )

        translate_module._model = None
        translate_module._tokenizer = None


class TestTranslateToYoruba:
    def test_returns_string(self):
        """translate_to_yoruba() returns a string."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.convert_tokens_to_ids.return_value = 42
        mock_tokenizer.batch_decode.return_value = ["Báwo ni?"]
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

        translate_module._model = mock_model
        translate_module._tokenizer = mock_tokenizer

        result = translate_to_yoruba("How are you?")
        assert isinstance(result, str)
        assert result == "Báwo ni?"

        translate_module._model = None
        translate_module._tokenizer = None

    def test_uses_yoruba_forced_bos(self):
        """translate_to_yoruba() forces yor_Latn BOS token."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1]])}
        mock_tokenizer.convert_tokens_to_ids.return_value = 99
        mock_tokenizer.batch_decode.return_value = ["test"]
        mock_model.generate.return_value = torch.tensor([[1]])

        translate_module._model = mock_model
        translate_module._tokenizer = mock_tokenizer

        translate_to_yoruba("hello")

        mock_tokenizer.convert_tokens_to_ids.assert_called_once_with("yor_Latn")
        call_kwargs = mock_model.generate.call_args.kwargs
        assert call_kwargs["forced_bos_token_id"] == 99

        translate_module._model = None
        translate_module._tokenizer = None

    def test_max_length_set(self):
        """translate_to_yoruba() sets max_length=256."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1]])}
        mock_tokenizer.convert_tokens_to_ids.return_value = 1
        mock_tokenizer.batch_decode.return_value = ["x"]
        mock_model.generate.return_value = torch.tensor([[1]])

        translate_module._model = mock_model
        translate_module._tokenizer = mock_tokenizer

        translate_to_yoruba("test")

        call_kwargs = mock_model.generate.call_args.kwargs
        assert call_kwargs["max_length"] == 256

        translate_module._model = None
        translate_module._tokenizer = None
