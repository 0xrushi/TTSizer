import json
import soundfile as sf
import numpy as np
import tempfile
import os
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, Any, List
from ttsizer.utils.logger import get_logger
from ttsizer.asr_backends.parakeet_v2 import ParakeetV2Backend

logger = get_logger("segment_transcriber")

class SegmentTranscriber:
    """
    Transcribes audio segments defined in a JSON file using a local ASR model (Parakeet).
    Used to fill in transcripts when diarization (e.g., Pyannote) only provides timestamps.
    """
    def __init__(self, global_config: Dict[str, Any], asr_config: Dict[str, Any]):
        self.device = asr_config.get("device", "cuda")
        self.model_name = asr_config.get("model_name", "nvidia/parakeet-tdt-0.6b-v2")
        # Use a larger batch size for segments as they are short
        self.batch_size = asr_config.get("batch_size", 16) 
        
        logger.info(f"Initializing SegmentTranscriber with {self.model_name} on {self.device}")
        self.backend = ParakeetV2Backend(model_name=self.model_name, device=self.device)

    def _time_to_sec(self, time_str: str) -> float:
        parts = time_str.split(':')
        if len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
        elif len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        return float(parts[0])

    def process_directory(self, json_dir: Path, audio_dir: Path):
        json_files = sorted([f for f in json_dir.glob("*.json") if not f.name.endswith(".uploaded.json")])
        
        if not json_files:
            logger.warning(f"No JSON files found in {json_dir}")
            return

        logger.info(f"Found {len(json_files)} episodes to transcribe.")

        for json_path in tqdm(json_files, desc="Episodes", unit="ep"):
            self._process_episode(json_path, audio_dir)

    def _process_episode(self, json_path: Path, audio_dir: Path):
        # Find audio
        audio_path = None
        for fmt in ['.flac', '.wav']:
            path = audio_dir / json_path.with_suffix(fmt).name
            if path.exists():
                audio_path = path
                break
        
        if not audio_path:
            logger.warning(f"Audio not found for {json_path.name}, skipping.")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)

        # Identify segments needing transcription
        to_transcribe_indices = []
        for i, seg in enumerate(segments):
            if not seg.get("transcript"):
                to_transcribe_indices.append(i)
        
        if not to_transcribe_indices:
            logger.info(f"No missing transcripts in {json_path.name}, skipping.")
            return

        logger.info(f"Transcribing {len(to_transcribe_indices)} segments in {json_path.name}...")

        # Load audio once
        try:
            audio, sr = sf.read(str(audio_path), dtype='float32')
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        except Exception as e:
            logger.error(f"Failed to read audio {audio_path}: {e}")
            return

        # Prepare batches
        temp_files = []
        try:
            # We process in chunks to avoid creating thousands of temp files at once if possible,
            # but batch_transcribe needs paths.
            
            # 1. Create temp files for all segments
            # (A optimized approach would be to yield batches, but let's be simple first)
            
            batch_indices = []
            current_batch_paths = []
            
            for idx in to_transcribe_indices:
                seg = segments[idx]
                start = self._time_to_sec(seg["start"])
                end = self._time_to_sec(seg["end"])
                
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                
                if end_sample <= start_sample or start_sample >= len(audio):
                    continue # Skip invalid audio
                
                chunk = audio[start_sample:end_sample]
                
                tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tf.name, chunk, sr)
                tf.close()
                temp_files.append(tf.name)
                
                current_batch_paths.append(tf.name)
                batch_indices.append(idx)
                
                if len(current_batch_paths) >= self.batch_size:
                    self._transcribe_batch(current_batch_paths, batch_indices, segments)
                    current_batch_paths = []
                    batch_indices = []

            # Process remaining
            if current_batch_paths:
                self._transcribe_batch(current_batch_paths, batch_indices, segments)

            # Save updated JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=4, ensure_ascii=False)
            
        finally:
            # Cleanup temp files
            for tf in temp_files:
                if os.path.exists(tf):
                    os.unlink(tf)

    def _transcribe_batch(self, paths: List[str], indices: List[int], segments: List[Dict]):
        try:
            results = self.backend.transcribe_batch(paths)
            for i, res in enumerate(results):
                seg_idx = indices[i]
                if res and res.text:
                    segments[seg_idx]["transcript"] = res.text.strip()
                else:
                    segments[seg_idx]["transcript"] = "" # Empty string if failed/silence
        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
