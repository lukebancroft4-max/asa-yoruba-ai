"""Tests for tts.py — TTS module (FarmerlineML/yoruba_tts-2025)."""

import os
import sys
import unicodedata
import numpy as np
import torch
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tts as tts_module
from tts import synthesize


def _make_mock_tts_components(audio_length: int = 16000):
    """Build mock (tokenizer, model) for VITS forward pass."""
    mock_encodings = MagicMock()
    mock_inputs = MagicMock()
    mock_encodings.to.return_value = mock_inputs

    mock_tokenizer = MagicMock(return_value=mock_encodings)

    fake_waveform = torch.zeros(1, audio_length)
    mock_output = MagicMock()
    mock_output.waveform = fake_waveform

    mock_model = MagicMock(return_value=mock_output)

    return mock_tokenizer, mock_model


class TestLoadTts:
    def test_returns_tokenizer_model_and_sample_rate(self):
        """load_tts() returns (tokenizer, model, sample_rate) tuple."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 22050
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoTokenizer") as mock_at,
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_at.from_pretrained.return_value = mock_tokenizer
            mock_vm.from_pretrained.return_value = mock_model

            tokenizer, model, sr = tts_module.load_tts(device="cpu")

        assert tokenizer is mock_tokenizer
        assert model is mock_model
        assert sr == 22050

    def test_model_moved_to_correct_device(self):
        """load_tts() calls model.to(device)."""
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 22050
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoTokenizer"),
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_vm.from_pretrained.return_value = mock_model
            tts_module.load_tts(device="cpu")

        mock_model.to.assert_called_once_with("cpu")

    def test_loads_farmerlineml_model(self):
        """load_tts() loads FarmerlineML/yoruba_tts-2025, not mms-tts-yor."""
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 22050
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoTokenizer") as mock_at,
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_vm.from_pretrained.return_value = mock_model
            tts_module.load_tts(device="cpu")

            mock_at.from_pretrained.assert_called_once_with("FarmerlineML/yoruba_tts-2025")
            mock_vm.from_pretrained.assert_called_once_with("FarmerlineML/yoruba_tts-2025")


class TestSynthesize:
    def test_returns_tuple_of_sample_rate_and_array(self):
        """synthesize() returns (int, numpy.ndarray)."""
        tokenizer, model = _make_mock_tts_components()
        sr, audio = synthesize(tokenizer, model, 16000, "Báwo ni?", device="cpu")
        assert isinstance(sr, int)
        assert sr == 16000
        assert isinstance(audio, np.ndarray)

    def test_audio_is_float32(self):
        """synthesize() output is float32."""
        tokenizer, model = _make_mock_tts_components()
        _, audio = synthesize(tokenizer, model, 16000, "test", device="cpu")
        assert audio.dtype == np.float32

    def test_audio_is_1d(self):
        """synthesize() returns 1D mono array."""
        tokenizer, model = _make_mock_tts_components(audio_length=8000)
        _, audio = synthesize(tokenizer, model, 16000, "test", device="cpu")
        assert audio.ndim == 1

    def test_audio_length_matches_waveform(self):
        """synthesize() output length matches model waveform output."""
        tokenizer, model = _make_mock_tts_components(audio_length=24000)
        _, audio = synthesize(tokenizer, model, 16000, "longer text", device="cpu")
        assert len(audio) == 24000

    def test_model_called_once(self):
        """synthesize() calls the model exactly once."""
        tokenizer, model = _make_mock_tts_components()
        synthesize(tokenizer, model, 16000, "Ẹ káàbọ̀", device="cpu")
        model.assert_called_once()

    def test_tokenizer_called_with_text(self):
        """synthesize() passes text (positional) to the tokenizer."""
        tokenizer, model = _make_mock_tts_components()
        synthesize(tokenizer, model, 16000, "Kí ni orúkọ rẹ?", device="cpu")
        tokenizer.assert_called_once_with("Kí ni orúkọ rẹ?", return_tensors="pt")

    def test_nfc_normalization_applied(self):
        """synthesize() normalizes text to NFC before tokenizing."""
        tokenizer, model = _make_mock_tts_components()
        # NFD: "a" + combining acute = "á" as two codepoints
        nfd_text = "Ba\u0301wo ni"
        nfc_text = unicodedata.normalize("NFC", nfd_text)
        synthesize(tokenizer, model, 16000, nfd_text, device="cpu")
        # Tokenizer must receive NFC form
        tokenizer.assert_called_once_with(nfc_text, return_tensors="pt")
