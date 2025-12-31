import pytest
from pathlib import Path
import shutil
import torch
from ttsizer.core.vocals_extract import VocalsExtractor

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping GPU-heavy VocalsExtractor test")
def test_vocals_extractor(tmp_path, english_sample_wav):
    """Test VocalsExtractor on a sample audio file."""
    # Check dependencies
    weights_path = Path("weights/kimmel_unwa_ft2_bleedless.ckpt")
    config_path = Path("configs/config_kimmel_unwa_ft.yaml")
    
    if not weights_path.exists() or not config_path.exists():
        pytest.skip(f"Weights or config not found for VocalsExtractor: {weights_path}, {config_path}")

    # Setup directories
    input_dir = tmp_path / "input_audio"
    input_dir.mkdir()
    output_dir = tmp_path / "vocals"
    
    # Copy sample audio to input dir
    # VocalsExtractor looks for files in input_dir
    src_path = Path(english_sample_wav)
    # Ensure it has a name
    shutil.copy(src_path, input_dir / "test_audio.wav")
    
    # Config
    config = {
        "output_folder": "vocals",
        "model_type": "mel_band_roformer",
        "use_gpu": True,
        "gpu_ids": [0],
        "output_format": "flac",
        "output_pcm_type": "PCM_24",
        "skip_if_output_exists": False,
        "model_path": str(weights_path),
        "model_config_path": str(config_path)
    }
    
    extractor = VocalsExtractor(config)
    
    # Run
    extractor.process_directory(input_dir, output_dir)
    
    # Verify
    assert output_dir.exists()
    out_file = output_dir / "test_audio_vocals.flac"
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    
    # Maybe check duration matches?
    # import soundfile as sf
    # info_in = sf.info(input_dir / "test_audio.wav")
    # info_out = sf.info(out_file)
    # assert abs(info_in.duration - info_out.duration) < 0.1
