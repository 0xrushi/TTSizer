import pytest
import os
import sys
import torch
import ffmpeg
import tempfile
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ttsizer.asr_backends.base import ASRResult, ASRSegmentTimes
from ttsizer.asr_backends.parakeet_v2 import ParakeetV2Backend
from ttsizer.asr_backends.gemini_asr import GeminiASRBackend
from ttsizer.asr_backends.whisper_hinglish import WhisperHinglishBackend
from ttsizer.core.asr_process import ASRProcessor

# --- Backend Tests ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping GPU-heavy Parakeet test")
def test_parakeet_v2_backend(english_sample_wav):
    """Test ParakeetV2Backend with English sample."""
    backend = ParakeetV2Backend(model_name="nvidia/parakeet-tdt-0.6b-v2", device="cuda")
    results = backend.transcribe_batch([english_sample_wav])
    
    assert isinstance(results, list)
    assert len(results) == 1
    # non empty text and contains some transcription text
    assert len(results[0].text) > 10
    assert "food" in results[0].text.lower() or "favorite" in results[0].text.lower()
    print(f"Parakeet Result: {results[0].text}")


@pytest.mark.skipif("GEMINI_API_KEY" not in os.environ, reason="GEMINI_API_KEY not found in environment")
def test_gemini_backend(english_sample_wav):
    """Test GeminiASRBackend with English sample."""
    api_key = os.environ["GEMINI_API_KEY"]
    backend = GeminiASRBackend(model_name="gemini-2.0-flash-lite", api_key=api_key)
    
    results = backend.transcribe_batch([english_sample_wav])
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert len(results[0].text) > 0
    print(f"Gemini Result: {results[0].text}")


def test_whisper_hinglish_backend(hinglish_sample_wav):
    """Test WhisperHinglishBackend with Hinglish sample."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = WhisperHinglishBackend(model_name="Oriserve/Whisper-Hindi2Hinglish-Prime", device=device)

    results = backend.transcribe_batch([hinglish_sample_wav])
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert len(results[0].text) > 0
    print(f"Whisper Hinglish Result: {results[0].text}")


# --- Processor Loading Tests ---

def test_asr_processor_load_parakeet():
    """Test ASRProcessor loads Parakeet backend correctly from config."""
    global_config = {"project_setup": {"target_speaker_labels": None}}
    asr_config = {
        "batch_size": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "backend": "parakeet_v2",
        "model_name": "nvidia/parakeet-tdt-0.6b-v2",
        "timestamp_deviation_threshold_sec": 0.5,
        "padding_sec": 0.1,
        "flagged_output_folder": "flagged",
    }
    
    proc = ASRProcessor(global_config, asr_config)
    assert isinstance(proc.backend, ParakeetV2Backend)


def test_asr_processor_load_whisper_hinglish():
    """Test ASRProcessor loads Whisper Hinglish backend correctly from config."""
    global_config = {"project_setup": {"target_speaker_labels": None}}
    asr_config = {
        "batch_size": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "backend": "whisper_hinglish",
        "whisper_hinglish_model_name": "Oriserve/Whisper-Hindi2Hinglish-Prime",
        "timestamp_deviation_threshold_sec": 0.5,
        "padding_sec": 0.1,
        "flagged_output_folder": "flagged",
    }
    
    proc = ASRProcessor(global_config, asr_config)
    assert isinstance(proc.backend, WhisperHinglishBackend)