"""Tests for speak.py — CLI TTS entry point."""

import os
import sys
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import speak as speak_module
from speak import speak


class TestSpeak:
    def _mock_tts_pipeline(self):
        """Set up mocks for translate + TTS pipeline."""
        mock_preprocessor = MagicMock()
        mock_model = MagicMock()

        # synthesize returns (sample_rate, float32 audio)
        fake_audio = np.random.randn(16000).astype(np.float32) * 0.5
        tts_patch = patch.object(
            speak_module, "synthesize",
            return_value=(16000, fake_audio),
        )
        load_patch = patch.object(
            speak_module, "load_tts",
            return_value=(mock_preprocessor, mock_model, 16000),
        )
        translate_patch = patch.object(
            speak_module, "translate_to_yoruba",
            return_value="Báwo ni?",
        )
        return tts_patch, load_patch, translate_patch

    def test_returns_wav_path(self):
        """speak() returns a path to a .wav file."""
        tts_p, load_p, trans_p = self._mock_tts_pipeline()
        with tts_p, load_p, trans_p, tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test.wav")
            result = speak("Hello", lang="yo", out=out)
            assert result == out
            assert os.path.exists(result)

    def test_english_triggers_translation(self):
        """speak() with lang='en' calls translate_to_yoruba."""
        tts_p, load_p, trans_p = self._mock_tts_pipeline()
        with tts_p, load_p, trans_p as mock_translate, tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test.wav")
            speak("Good morning", lang="en", out=out)
            mock_translate.assert_called_once_with("Good morning")

    def test_yoruba_skips_translation(self):
        """speak() with lang='yo' does NOT call translate."""
        tts_p, load_p, trans_p = self._mock_tts_pipeline()
        with tts_p, load_p, trans_p as mock_translate, tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test.wav")
            speak("Báwo ni?", lang="yo", out=out)
            mock_translate.assert_not_called()

    def test_engine_passed_to_load_tts(self):
        """speak() passes engine parameter to load_tts()."""
        tts_p, load_p, trans_p = self._mock_tts_pipeline()
        with tts_p, load_p as mock_load, trans_p, tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test.wav")
            speak("test", lang="yo", engine="mms", out=out)
            mock_load.assert_called_once_with(device="cpu", engine="mms")

    def test_output_is_valid_wav(self):
        """speak() writes a valid WAV file."""
        import scipy.io.wavfile as wavfile

        tts_p, load_p, trans_p = self._mock_tts_pipeline()
        with tts_p, load_p, trans_p, tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test.wav")
            speak("Ẹ káàbọ̀", lang="yo", out=out)

            sr, data = wavfile.read(out)
            assert sr == 16000
            assert data.dtype == np.int16

    def test_default_output_path(self):
        """speak() generates a default path in outputs/ when out=None."""
        tts_p, load_p, trans_p = self._mock_tts_pipeline()
        with tts_p, load_p, trans_p:
            result = speak("test default", lang="yo")
            assert "outputs" in result
            assert result.endswith(".wav")
            # Cleanup
            if os.path.exists(result):
                os.unlink(result)
