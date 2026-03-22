"""Tests for asr.py — faster-whisper ASR module."""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asr as asr_module
from asr import transcribe, load_asr, _normalize


class TestLoadAsr:
    def test_creates_whisper_model(self):
        """load_asr() creates a WhisperModel with correct params."""
        with patch.object(asr_module, "WhisperModel") as mock_wm:
            mock_wm.return_value = MagicMock()
            load_asr(device="cuda")

            mock_wm.assert_called_once_with("large-v3", device="cpu", compute_type="int8")

    def test_always_runs_on_cpu(self):
        """load_asr() uses CPU regardless of device arg."""
        with patch.object(asr_module, "WhisperModel") as mock_wm:
            mock_wm.return_value = MagicMock()
            load_asr(device="cuda")

            call_kwargs = mock_wm.call_args
            assert call_kwargs.kwargs["device"] == "cpu"

    def test_returns_whisper_model_instance(self):
        """load_asr() returns the WhisperModel instance."""
        mock_model = MagicMock()
        with patch.object(asr_module, "WhisperModel", return_value=mock_model):
            result = load_asr()
            assert result is mock_model


class TestNormalize:
    def test_normalizes_loud_audio(self):
        """_normalize() scales peak to 0.85."""
        audio = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        result = _normalize(audio)
        assert np.isclose(np.abs(result).max(), 0.85)

    def test_silent_audio_unchanged(self):
        """_normalize() handles all-zero audio without division by zero."""
        audio = np.zeros(100, dtype=np.float32)
        result = _normalize(audio)
        assert np.all(result == 0.0)


class TestTranscribe:
    def _make_mock_model(self, segments_text: list[str], lang: str = "yo", prob: float = 0.95):
        """Build a mock WhisperModel that returns given segments."""
        mock_model = MagicMock()

        mock_segments = []
        for text in segments_text:
            seg = MagicMock()
            seg.text = text
            mock_segments.append(seg)

        mock_info = MagicMock()
        mock_info.language = lang
        mock_info.language_probability = prob

        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
        return mock_model

    def _make_temp_wav(self, duration_s: float = 1.0, sr: int = 16000) -> str:
        """Create a temporary WAV file with silence."""
        audio = np.zeros(int(sr * duration_s), dtype=np.float32)
        path = tempfile.mktemp(suffix=".wav")
        sf.write(path, audio, sr)
        return path

    def test_returns_transcribed_text(self):
        """transcribe() joins segment texts."""
        model = self._make_mock_model(["Báwo", "ni?"])
        path = self._make_temp_wav()
        try:
            result = transcribe(model, path)
            assert result == "Báwo ni?"
        finally:
            os.unlink(path)

    def test_strips_segment_whitespace(self):
        """transcribe() strips whitespace from each segment."""
        model = self._make_mock_model(["  Ẹ káàbọ̀  "])
        path = self._make_temp_wav()
        try:
            result = transcribe(model, path)
            assert result == "Ẹ káàbọ̀"
        finally:
            os.unlink(path)

    def test_calls_model_transcribe_with_vad(self):
        """transcribe() passes vad_filter=True to model."""
        model = self._make_mock_model(["test"])
        path = self._make_temp_wav()
        try:
            transcribe(model, path)
            call_kwargs = model.transcribe.call_args.kwargs
            assert call_kwargs["vad_filter"] is True
            assert call_kwargs["beam_size"] == 5
            assert call_kwargs["language"] is None
        finally:
            os.unlink(path)

    def test_handles_stereo_audio(self):
        """transcribe() converts stereo to mono."""
        model = self._make_mock_model(["ok"])
        # Write stereo WAV
        stereo = np.zeros((16000, 2), dtype=np.float32)
        path = tempfile.mktemp(suffix=".wav")
        sf.write(path, stereo, 16000)
        try:
            result = transcribe(model, path)
            assert isinstance(result, str)
        finally:
            os.unlink(path)

    def test_empty_segments_returns_empty_string(self):
        """transcribe() returns empty string when no segments detected."""
        model = self._make_mock_model([])
        path = self._make_temp_wav()
        try:
            result = transcribe(model, path)
            assert result == ""
        finally:
            os.unlink(path)
