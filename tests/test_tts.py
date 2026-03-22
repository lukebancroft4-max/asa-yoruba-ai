"""Tests for tts.py — TTS module."""

import os
import sys
import numpy as np
import torch
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tts as tts_module
from tts import synthesize


def _make_mock_tts_components(audio_length: int = 16000):
    """Build mock (processor, model) that simulate VITS forward pass.

    The processor mock returns a MagicMock whose .to() also returns a MagicMock
    (matching the real HuggingFace BatchEncoding.to() API).
    """
    # processor(text=..., return_tensors="pt") → encodings → .to(device) → inputs
    mock_encodings = MagicMock()
    mock_inputs = MagicMock()
    mock_encodings.to.return_value = mock_inputs

    mock_processor = MagicMock(return_value=mock_encodings)

    # model(**inputs).waveform → fake 1D tensor
    fake_waveform = torch.zeros(1, audio_length)
    mock_output = MagicMock()
    mock_output.waveform = fake_waveform

    mock_model = MagicMock(return_value=mock_output)

    return mock_processor, mock_model


class TestLoadTts:
    def test_returns_processor_model_and_sample_rate(self):
        """load_tts() returns (processor, model, sample_rate) tuple."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 16000
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoProcessor") as mock_ap,
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model

            processor, model, sr = tts_module.load_tts(device="cpu")

        assert processor is mock_processor
        assert model is mock_model
        assert sr == 16000

    def test_model_moved_to_correct_device(self):
        """load_tts() calls model.to(device)."""
        mock_model = MagicMock()
        mock_model.config.sampling_rate = 16000
        mock_model.to.return_value = mock_model

        with (
            patch.object(tts_module, "AutoProcessor"),
            patch.object(tts_module, "VitsModel") as mock_vm,
        ):
            mock_vm.from_pretrained.return_value = mock_model
            tts_module.load_tts(device="cpu")

        mock_model.to.assert_called_once_with("cpu")


class TestSynthesize:
    def test_returns_tuple_of_sample_rate_and_array(self):
        """synthesize() returns (int, numpy.ndarray)."""
        processor, model = _make_mock_tts_components()
        sr, audio = synthesize(processor, model, 16000, "Báwo ni?", device="cpu")
        assert isinstance(sr, int)
        assert sr == 16000
        assert isinstance(audio, np.ndarray)

    def test_audio_is_float32(self):
        """synthesize() output is float32 (compatible with Gradio gr.Audio)."""
        processor, model = _make_mock_tts_components()
        _, audio = synthesize(processor, model, 16000, "test", device="cpu")
        assert audio.dtype == np.float32

    def test_audio_is_1d(self):
        """synthesize() returns 1D mono array."""
        processor, model = _make_mock_tts_components(audio_length=8000)
        _, audio = synthesize(processor, model, 16000, "test", device="cpu")
        assert audio.ndim == 1

    def test_audio_length_matches_waveform(self):
        """synthesize() output length matches the model's waveform output."""
        processor, model = _make_mock_tts_components(audio_length=24000)
        _, audio = synthesize(processor, model, 16000, "longer text", device="cpu")
        assert len(audio) == 24000

    def test_model_called_once(self):
        """synthesize() calls the model exactly once per invocation."""
        processor, model = _make_mock_tts_components()
        synthesize(processor, model, 16000, "Ẹ káàbọ̀", device="cpu")
        model.assert_called_once()

    def test_processor_called_with_text(self):
        """synthesize() passes the text through to the processor."""
        processor, model = _make_mock_tts_components()
        synthesize(processor, model, 16000, "Kí ni orúkọ rẹ?", device="cpu")
        processor.assert_called_once_with(
            text="Kí ni orúkọ rẹ?", return_tensors="pt"
        )
