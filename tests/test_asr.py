"""Tests for asr.py — Automatic Speech Recognition module.

asr.py uses faster-whisper (WhisperModel), NOT transformers pipeline.
transcribe() reads audio with sf.read(), normalizes, then passes the
numpy array (not a file path) to model.transcribe().
"""

import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asr as asr_module
from asr import transcribe, _normalize


class TestNormalize:
    def test_normalizes_to_0_85_peak(self):
        """_normalize() scales peak amplitude to 0.85."""
        audio = np.array([0.0, 0.5, 1.0, -0.5], dtype=np.float32)
        result = _normalize(audio)
        assert abs(result.max() - 0.85) < 1e-6

    def test_handles_silent_audio(self):
        """_normalize() returns zeros unchanged (no div-by-zero)."""
        audio = np.zeros(100, dtype=np.float32)
        result = _normalize(audio)
        assert np.all(result == 0.0)

    def test_returns_new_array(self):
        """_normalize() does not mutate the input array."""
        audio = np.array([0.5, -0.5], dtype=np.float32)
        original = audio.copy()
        _normalize(audio)
        np.testing.assert_array_equal(audio, original)


class TestLoadAsr:
    def test_creates_whisper_model(self):
        """load_asr() creates a WhisperModel instance."""
        mock_model = MagicMock()
        with patch.object(asr_module, "WhisperModel", return_value=mock_model) as mock_cls:
            result = asr_module.load_asr(device="cpu")
            mock_cls.assert_called_once()
            assert result is mock_model

    def test_always_runs_on_cpu(self):
        """load_asr() forces CPU regardless of device argument (keeps VRAM for TTS)."""
        mock_model = MagicMock()
        with patch.object(asr_module, "WhisperModel", return_value=mock_model) as mock_cls:
            asr_module.load_asr(device="cuda")
            call_args = mock_cls.call_args
            # device arg is second positional or keyword
            device_used = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("device")
            assert device_used == "cpu"

    def test_uses_large_v3_model(self):
        """load_asr() loads large-v3 model for best Yoruba accuracy."""
        mock_model = MagicMock()
        with patch.object(asr_module, "WhisperModel", return_value=mock_model) as mock_cls:
            asr_module.load_asr(device="cpu")
            call_args = mock_cls.call_args
            model_size = call_args.args[0] if call_args.args else call_args.kwargs.get("model_size_or_path")
            assert model_size == "large-v3"


class TestTranscribe:
    def _make_mock_model(self, texts: list[str]) -> MagicMock:
        """Build a WhisperModel mock that returns segments with the given texts."""
        mock_model = MagicMock()
        segments = [MagicMock(text=t) for t in texts]
        mock_info = MagicMock()
        mock_info.language = "yo"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = (iter(segments), mock_info)
        return mock_model

    def test_returns_joined_segment_texts(self):
        """transcribe() joins all segment texts with spaces."""
        model = self._make_mock_model(["Báwo ni?", "Ẹ káàbọ̀."])
        with patch.object(asr_module, "sf") as mock_sf:
            mock_sf.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            result = transcribe(model, "/tmp/fake.wav")
        assert result == "Báwo ni? Ẹ káàbọ̀."

    def test_strips_whitespace_from_result(self):
        """transcribe() strips surrounding whitespace from each segment."""
        model = self._make_mock_model(["  Báwo ni?  "])
        with patch.object(asr_module, "sf") as mock_sf:
            mock_sf.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            result = transcribe(model, "/tmp/fake.wav")
        assert result == "Báwo ni?"

    def test_reads_audio_file(self):
        """transcribe() reads audio from the given file path."""
        model = self._make_mock_model(["test"])
        with patch.object(asr_module, "sf") as mock_sf:
            mock_sf.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            transcribe(model, "/tmp/my_audio.wav")
            mock_sf.read.assert_called_once_with("/tmp/my_audio.wav", dtype="float32")

    def test_passes_numpy_array_to_transcribe(self):
        """transcribe() passes the numpy array (not file path) to model.transcribe()."""
        model = self._make_mock_model(["test"])
        with patch.object(asr_module, "sf") as mock_sf:
            fake_audio = np.array([0.1, 0.2], dtype=np.float32)
            mock_sf.read.return_value = (fake_audio, 16000)
            transcribe(model, "/tmp/fake.wav")
            call_args = model.transcribe.call_args
            passed_audio = call_args.args[0]
            assert isinstance(passed_audio, np.ndarray)

    def test_converts_stereo_to_mono(self):
        """transcribe() averages stereo channels before passing to model."""
        model = self._make_mock_model(["test"])
        with patch.object(asr_module, "sf") as mock_sf:
            stereo = np.ones((16000, 2), dtype=np.float32)
            mock_sf.read.return_value = (stereo, 16000)
            transcribe(model, "/tmp/stereo.wav")
            passed_audio = model.transcribe.call_args.args[0]
            assert passed_audio.ndim == 1

    def test_vad_filter_enabled(self):
        """transcribe() uses VAD filter to reduce hallucination on silence."""
        model = self._make_mock_model(["test"])
        with patch.object(asr_module, "sf") as mock_sf:
            mock_sf.read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            transcribe(model, "/tmp/fake.wav")
            call_kwargs = model.transcribe.call_args.kwargs
            assert call_kwargs.get("vad_filter") is True
