import pytest
from pathlib import Path
import shutil
import os
import json
from ttsizer.core.llm_diarize import LLMDiarizer

@pytest.mark.skipif("GEMINI_API_KEY" not in os.environ, reason="GEMINI_API_KEY not found")
def test_llm_diarizer_gemini(tmp_path, english_sample_wav):
    """Test LLMDiarizer with Gemini backend."""
    # Setup inputs
    input_dir = tmp_path / "vocals_norm"
    input_dir.mkdir()
    output_dir = tmp_path / "transcripts"
    
    # Copy sample
    shutil.copy(english_sample_wav, input_dir / "test_ep.wav")
    
    # Config
    global_config = {
        "project_setup": {
            "series_name": "TestSeries",
            "target_speaker_labels": None
        }
    }
    diarizer_config = {
        "backend": "gemini",
        "model_name": "gemini-2.0-flash-lite",
        "temperature": 0.0,
        "top_p": 0.8,
        "prompt_template_file": "configs/prompt_template.txt",
        "skip_if_output_exists": False,
        "save_raw_llm_output": True,
        "compact_output": True,
        "continue_on_truncation": False,
        "max_output_tokens": 1024,
        "reuse_uploaded_files": False # Force upload for test
    }
    
    # Check template
    if not Path(diarizer_config["prompt_template_file"]).exists():
        pytest.skip("Prompt template not found")

    diarizer = LLMDiarizer(global_config, diarizer_config)
    
    # Run
    diarizer.process_directory(input_dir, output_dir)
    
    # Verify
    out_json = output_dir / "test_ep.json"
    assert out_json.exists()
    
    with open(out_json, 'r') as f:
        data = json.load(f)
        assert isinstance(data, list)
        # Should have at least one segment
        assert len(data) > 0
        assert "start" in data[0]
        assert "end" in data[0]
        assert "speaker" in data[0]
        # Transcript might be there if Gemini did it
        assert "transcript" in data[0]

@pytest.mark.skipif("HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ, 
                    reason="HF_TOKEN not found for Pyannote")
def test_llm_diarizer_pyannote(tmp_path, english_sample_wav):
    """Test LLMDiarizer with Pyannote backend."""
    # Setup inputs
    input_dir = tmp_path / "vocals_norm_py"
    input_dir.mkdir()
    output_dir = tmp_path / "transcripts_py"
    
    shutil.copy(english_sample_wav, input_dir / "test_ep.wav")
    
    global_config = {"project_setup": {"series_name": "TestSeries"}}
    diarizer_config = {
        "backend": "pyannote_audio",
        "pyannote_model_name": "pyannote/speaker-diarization-3.1",
        "hf_token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu", # Pyannote might need GPU? Works on CPU but slow.
        "skip_if_output_exists": False,
        "min_speakers": 1,
        "max_speakers": 2
    }

    try:
        diarizer = LLMDiarizer(global_config, diarizer_config)
        diarizer.process_directory(input_dir, output_dir)
    except Exception as e:
        pytest.fail(f"Pyannote diarization failed: {e}")

    # Verify
    out_json = output_dir / "test_ep.json"
    assert out_json.exists()
    
    with open(out_json, 'r') as f:
        data = json.load(f)
        assert isinstance(data, list)
        if len(data) > 0:
            assert data[0]["transcript"] is None # Pyannote leaves it null
