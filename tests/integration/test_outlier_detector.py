import pytest
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf
from ttsizer.core.outlier_detect import OutlierDetector

def create_dummy_wav(path, duration=1.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sr)

@pytest.mark.skipif(not Path("weights/wespeaker-voxceleb-resnet293-LM").exists(), 
                    reason="Embedding model weights not found")
def test_outlier_detector(tmp_path):
    """Test OutlierDetector on dummy data."""
    # Setup inputs
    input_dir = tmp_path / "aligned_clips"
    output_dir = tmp_path / "final"
    
    # Create speakers
    spk_a = input_dir / "Speaker_A"
    spk_a.mkdir(parents=True)
    spk_b = input_dir / "Speaker_B"
    spk_b.mkdir(parents=True)
    
    # Create clips
    # Speaker A: 5 clips (need min_segments_for_master_profile? default is 10 in config, but I can lower it)
    for i in range(5):
        create_dummy_wav(spk_a / f"clip_{i}.wav", duration=1.0)
        
    # Speaker B: 5 clips
    for i in range(5):
        create_dummy_wav(spk_b / f"clip_{i}.wav", duration=1.0)
        
    # Config
    global_config = {
        "project_setup": {
            "series_name": "TestSeries",
            "target_speaker_labels": None
        }
    }
    
    detector_config = {
        "output_folder": "final",
        "target_sample_rate": 16000,
        "use_gpu": False, # Use CPU for test
        "min_clip_duration_seconds": 0.5,
        "min_segments_for_master_profile": 3, # Lower threshold for test
        "centroid_refinement_percentile": 50,
        "min_segments_for_refinement": 3,
        "outlier_threshold_definite": 0.70,
        "outlier_threshold_uncertain": 0.60,
        "embedding_model_path": "weights/wespeaker-voxceleb-resnet293-LM"
    }
    
    try:
        detector = OutlierDetector(global_config, detector_config)
    except Exception as e:
        pytest.fail(f"Failed to init OutlierDetector: {e}")
        
    # Run
    detector.process_directory(input_dir, output_dir)
    
    # Verify
    # Expect output structure: output_dir / Speaker_A / definite / ...
    # Since these are identical sine waves, they should match perfectly (distance ~0) and be 'definite'.
    
    out_spk_a = output_dir / "Speaker_A" / "definite"
    assert out_spk_a.exists()
    assert len(list(out_spk_a.glob("*.wav"))) >= 3
    
    out_spk_b = output_dir / "Speaker_B" / "definite"
    assert out_spk_b.exists()
    assert len(list(out_spk_b.glob("*.wav"))) >= 3
