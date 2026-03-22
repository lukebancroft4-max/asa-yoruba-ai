"""Tests for tts.py and tts_engines.py — single + dual engine TTS."""

import os
import sys
import unicodedata
import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tts as tts_module
import tts_engines as engines_module
from tts import synthesize
from tts_engines import load_tts, ENGINES, DEFAULT_ENGINE


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
    def test_farmerline_delegates_to_base_tts(self):
        """load_tts(engine='farmerline') delegates to base tts.load_tts."""
        mock_result = (MagicMock(), MagicMock(), 22050)
        with patch.object(engines_module, "_load_farmerline", return_value=mock_result) as mock_fn:
            result = load_tts(device="cpu", engine="farmerline")
            mock_fn.assert_called_once_with(device="cpu")
            assert result == mock_result

    def test_mms_uses_auto_processor(self):
        """MMS engine uses AutoProcessor."""
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 16000
        mock_model.to.return_value = mock_model

        with (
            patch.object(engines_module, "AutoProcessor") as mock_ap,
            patch.object(engines_module, "VitsModel") as mock_vm,
        ):
            mock_ap.from_pretrained.return_value = MagicMock()
            mock_vm.from_pretrained.return_value = mock_model

            load_tts(device="cpu", engine="mms")

            mock_ap.from_pretrained.assert_called_once_with("facebook/mms-tts-yor")

    def test_model_moved_to_correct_device(self):
        """load_tts() with MMS engine calls model.to(device)."""
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 16000
        mock_model.to.return_value = mock_model

        with (
            patch.object(engines_module, "AutoProcessor"),
            patch.object(engines_module, "VitsModel") as mock_vm,
        ):
            mock_vm.from_pretrained.return_value = mock_model
            load_tts(device="cpu", engine="mms")

        mock_model.to.assert_called_once_with("cpu")

    def test_invalid_engine_raises_value_error(self):
        """load_tts() raises ValueError for unknown engine."""
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            load_tts(device="cpu", engine="nonexistent")


class TestBaseTtsLoadTts:
    def test_returns_tokenizer_model_and_sample_rate(self):
        """Base load_tts() returns (tokenizer, model, sample_rate) tuple."""
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

    def test_loads_farmerlineml_model(self):
        """Base load_tts() loads FarmerlineML model."""
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

    def test_nfc_normalization_applied(self):
        """synthesize() normalizes text to NFC before tokenizing."""
        preprocessor, model = _make_mock_tts_components()
        nfd_text = "Ba\u0301wo ni"
        nfc_text = unicodedata.normalize("NFC", nfd_text)
        synthesize(preprocessor, model, 16000, nfd_text, device="cpu")
        # Check NFC form was passed (either as text= kwarg or positional)
        call_args = preprocessor.call_args
        passed_text = call_args.kwargs.get("text") or call_args.args[0]
        assert passed_text == nfc_text
