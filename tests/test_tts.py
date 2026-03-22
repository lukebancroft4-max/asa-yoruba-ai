"""Tests for tts.py — dual-engine TTS module."""

import os
import sys
import unicodedata
import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tts as tts_module
from tts import synthesize, load_tts, ENGINES, DEFAULT_ENGINE


def _make_mock_tts_components(audio_length: int = 16000):
    """Build mock (preprocessor, model) for VITS forward pass."""
    mock_encodings = MagicMock()
    mock_inputs = MagicMock()
    mock_encodings.to.return_value = mock_inputs

    mock_preprocessor = MagicMock(return_value=mock_encodings)

    fake_waveform = torch.zeros(1, audio_length)
    mock_output = MagicMock()
    mock_output.waveform = fake_waveform

    mock_model = MagicMock(return_value=mock_output)

    return mock_preprocessor, mock_model


class TestEngineConfig:
    def test_engines_dict_has_mms_and_farmerline(self):
        """ENGINES contains both 'mms' and 'farmerline' entries."""
        assert "mms" in ENGINES
        assert "farmerline" in ENGINES

    def test_default_engine_is_farmerline(self):
        """Default engine is farmerline."""
        assert DEFAULT_ENGINE == "farmerline"


class TestLoadTts:
    def test_returns_preprocessor_model_and_sample_rate(self):
        """load_tts() returns (preprocessor, model, sample_rate) tuple."""
        mock_preprocessor = MagicMock()
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 22050
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoTokenizer") as mock_at,
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_at.from_pretrained.return_value = mock_preprocessor
            mock_vm.from_pretrained.return_value = mock_model

            preprocessor, model, sr = load_tts(device="cpu", engine="farmerline")

        assert preprocessor is mock_preprocessor
        assert model is mock_model
        assert sr == 22050

    def test_farmerline_uses_auto_tokenizer(self):
        """Farmerline engine uses AutoTokenizer."""
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 22050
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoTokenizer") as mock_at,
            patch.object(tts_module, "AutoProcessor") as mock_ap,
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_at.from_pretrained.return_value = MagicMock()
            mock_vm.from_pretrained.return_value = mock_model

            load_tts(device="cpu", engine="farmerline")

            mock_at.from_pretrained.assert_called_once_with("FarmerlineML/yoruba_tts-2025")
            mock_ap.from_pretrained.assert_not_called()

    def test_mms_uses_auto_processor(self):
        """MMS engine uses AutoProcessor."""
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 16000
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoTokenizer") as mock_at,
            patch.object(tts_module, "AutoProcessor") as mock_ap,
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_ap.from_pretrained.return_value = MagicMock()
            mock_vm.from_pretrained.return_value = mock_model

            load_tts(device="cpu", engine="mms")

            mock_ap.from_pretrained.assert_called_once_with("facebook/mms-tts-yor")
            mock_at.from_pretrained.assert_not_called()

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
            load_tts(device="cpu", engine="farmerline")

        mock_model.to.assert_called_once_with("cpu")

    def test_invalid_engine_raises_value_error(self):
        """load_tts() raises ValueError for unknown engine."""
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            load_tts(device="cpu", engine="nonexistent")


class TestSynthesize:
    def test_returns_tuple_of_sample_rate_and_array(self):
        """synthesize() returns (int, numpy.ndarray)."""
        preprocessor, model = _make_mock_tts_components()
        sr, audio = synthesize(preprocessor, model, 16000, "Báwo ni?", device="cpu")
        assert isinstance(sr, int)
        assert sr == 16000
        assert isinstance(audio, np.ndarray)

    def test_audio_is_float32(self):
        """synthesize() output is float32."""
        preprocessor, model = _make_mock_tts_components()
        _, audio = synthesize(preprocessor, model, 16000, "test", device="cpu")
        assert audio.dtype == np.float32

    def test_audio_is_1d(self):
        """synthesize() returns 1D mono array."""
        preprocessor, model = _make_mock_tts_components(audio_length=8000)
        _, audio = synthesize(preprocessor, model, 16000, "test", device="cpu")
        assert audio.ndim == 1

    def test_audio_length_matches_waveform(self):
        """synthesize() output length matches model waveform output."""
        preprocessor, model = _make_mock_tts_components(audio_length=24000)
        _, audio = synthesize(preprocessor, model, 16000, "longer text", device="cpu")
        assert len(audio) == 24000

    def test_model_called_once(self):
        """synthesize() calls the model exactly once."""
        preprocessor, model = _make_mock_tts_components()
        synthesize(preprocessor, model, 16000, "Ẹ káàbọ̀", device="cpu")
        model.assert_called_once()

    def test_preprocessor_called_with_text_kwarg(self):
        """synthesize() passes text= keyword to preprocessor (works for both engines)."""
        preprocessor, model = _make_mock_tts_components()
        synthesize(preprocessor, model, 16000, "Kí ni orúkọ rẹ?", device="cpu")
        preprocessor.assert_called_once_with(
            text="Kí ni orúkọ rẹ?", return_tensors="pt"
        )

    def test_nfc_normalization_applied(self):
        """synthesize() normalizes text to NFC before tokenizing."""
        preprocessor, model = _make_mock_tts_components()
        nfd_text = "Ba\u0301wo ni"
        nfc_text = unicodedata.normalize("NFC", nfd_text)
        synthesize(preprocessor, model, 16000, nfd_text, device="cpu")
        preprocessor.assert_called_once_with(text=nfc_text, return_tensors="pt")
