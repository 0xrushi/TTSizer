import pytest
from pathlib import Path
import shutil
from ttsizer.core.audio_extract import AudioExtractor

def test_audio_extractor(tmp_path):
    """Test AudioExtractor on the sample directory."""
    # Setup inputs
    # We use the existing 'sample' directory which contains .mp4 files.
    input_dir = Path("sample")
    if not input_dir.exists():
        pytest.skip("Sample directory 'sample' not found.")
    
    # Setup output
    output_dir = tmp_path / "audio"
    
    # Config
    config = {
        "output_folder": "audio",
        "preferred_lang_codes": ["eng", "en"],
        "max_workers": 2,
        "output_sample_rate": 16000, # Use 16k for speed/size
        "output_codec": "flac",
        "resolution_threshold_for_aac_step": 480,
        "intermediate_aac_bitrate": "32k"
    }
    
    extractor = AudioExtractor(config)
    
    # Run
    extractor.process_directory(input_dir, output_dir)
    
    # Verify
    assert output_dir.exists()
    files = list(output_dir.rglob("*.flac"))
    assert len(files) > 0, "No FLAC files were extracted."
    
    # Check if files have non-zero size
    for f in files:
        assert f.stat().st_size > 0
