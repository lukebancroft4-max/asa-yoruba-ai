"""Tests for asr.py — ASR module."""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asr as asr_module
from asr import transcribe


class TestLoadAsr:
    def test_primary_model_loads(self):
        """load_asr() tries LyngualLabs/whisper-small-yoruba first."""
        mock_pipe = MagicMock()
        with patch.object(asr_module, "pipeline", return_value=mock_pipe) as mock_fn:
            result = asr_module.load_asr(device="cpu")
            assert mock_fn.call_count == 1
            first_call_args = mock_fn.call_args_list[0]
            assert "LyngualLabs/whisper-small-yoruba" in str(first_call_args)
            assert result is mock_pipe

    def test_falls_back_to_whisper_on_failure(self):
        """load_asr() falls back to openai/whisper-small if primary raises."""
        call_count = 0
        mock_asr = MagicMock()

        def mock_pipeline_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Model not found")
            return mock_asr

        with patch.object(asr_module, "pipeline", side_effect=mock_pipeline_factory):
            result = asr_module.load_asr(device="cpu")

        assert call_count == 2
        assert result is mock_asr

    def test_always_runs_on_cpu(self):
        """load_asr() always uses CPU to preserve VRAM for TTS."""
        mock_pipe = MagicMock()
        with patch.object(asr_module, "pipeline", return_value=mock_pipe) as mock_fn:
            asr_module.load_asr(device="cuda")  # even if cuda passed
            call_kwargs = mock_fn.call_args.kwargs
            assert call_kwargs.get("device") == "cpu"


class TestTranscribe:
    def test_returns_stripped_text(self):
        """transcribe() strips surrounding whitespace."""
        mock_pipeline = MagicMock(return_value={"text": "  Báwo ni?  "})
        result = transcribe(mock_pipeline, "/tmp/fake.wav")
        assert result == "Báwo ni?"

    def test_calls_pipeline_with_audio_path(self):
        """transcribe() passes the audio path to the pipeline."""
        mock_pipeline = MagicMock(return_value={"text": "Ẹ káàbọ̀"})
        transcribe(mock_pipeline, "/tmp/audio.wav")
        mock_pipeline.assert_called_once_with("/tmp/audio.wav", return_timestamps=False)

    def test_handles_real_dummy_audio_file(self):
        """transcribe() accepts a real wav file path."""
        audio_data = np.zeros(16000, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, 16000)
            tmp_path = f.name
        try:
            mock_pipeline = MagicMock(return_value={"text": "test"})
            result = transcribe(mock_pipeline, tmp_path)
            assert isinstance(result, str)
        finally:
            os.unlink(tmp_path)
