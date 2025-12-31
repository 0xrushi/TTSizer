import pytest
import numpy as np
import soundfile as sf
import tempfile
import os
import ffmpeg
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if it exists
env_path = Path(__file__).parent.parent.parent / ".env"
print(f"env_path: {env_path}")
if env_path.exists():
    load_dotenv(env_path)
else:
    raise FileNotFoundError(f"Environment file not found at {env_path}")

@pytest.fixture(scope="session")
def sample_audio_path():
    """Generates a temporary 1-second 16kHz sine wave audio file."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # A simple 440Hz sine wave (A4 note)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        sf.write(tf.name, audio, sr)
        path = tf.name
    
    yield path
    
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture(scope="session")
def sample_silence_path():
    """Generates a temporary 1-second 16kHz silence audio file."""
    sr = 16000
    duration = 1.0
    audio = np.zeros(int(sr * duration))
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        sf.write(tf.name, audio, sr)
        path = tf.name
    
    yield path
    
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture(scope="session")
def english_sample_wav():
    """Extracts audio from sample/dua_lipa_fav_food.mp4 to a temporary WAV."""
    video_path = Path("sample/dua_lipa_fav_food.mp4").resolve()
    if not video_path.exists():
        pytest.skip(f"Sample file {video_path} not found")
        
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        out_path = tf.name
    
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(out_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:
        if os.path.exists(out_path): os.remove(out_path)
        pytest.fail(f"Failed to extract audio from {video_path}: {e}")
        
    yield out_path
    if os.path.exists(out_path): os.remove(out_path)

@pytest.fixture(scope="session")
def hinglish_sample_wav():
    """Extracts audio from sample/amitabh_deepika_ranveer_vid.mp4 to a temporary WAV."""
    video_path = Path("sample/amitabh_deepika_ranveer_vid.mp4").resolve()
    if not video_path.exists():
        pytest.skip(f"Sample file {video_path} not found")
        
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        out_path = tf.name
    
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(out_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:
        if os.path.exists(out_path): os.remove(out_path)
        pytest.fail(f"Failed to extract audio from {video_path}: {e}")
        
    yield out_path
    if os.path.exists(out_path): os.remove(out_path)