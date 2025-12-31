import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
import torch
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

import shutil
import yaml
from typing import Dict, Any
from ttsizer.utils.logger import get_logger
from ttsizer.asr_backends.parakeet_v2 import ParakeetV2Backend
from ttsizer.asr_backends.gemini_asr import GeminiASRBackend
from ttsizer.asr_backends.whisper_hinglish import WhisperHinglishBackend
from ttsizer.asr_backends.base import ASRBackend, ASRResult

logger = get_logger("asr_processor")

class ASRProcessor:
    """Processes audio files using an ASR backend for transcription and flagging.

    This class transcribes audio files, extracts word-level timestamps, and flags segments
    where the detected speech boundaries deviate significantly from the file boundaries.
    Flagged segments are saved with padding for further review.
    """

    def __init__(self, global_config: Dict[str, Any], asr_config: Dict[str, Any]):
        """Initializes the ASRProcessor with global and ASR-specific configurations.

        Args:
            global_config: Dictionary containing global project setup information.
            asr_config: Dictionary containing configuration specific to the ASR processor.
        """
        self.target_speakers = global_config['project_setup'].get('target_speaker_labels')
        
        self.batch_size = asr_config["batch_size"]
        self.device = asr_config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        self.backend_name = str(asr_config.get("backend", "parakeet_v2"))
        
        self.time_thresh = asr_config["timestamp_deviation_threshold_sec"]
        self.padding = asr_config["padding_sec"]

        self.flagged_dir = asr_config["flagged_output_folder"]
        self.def_dir = "definite"

        self.backend: ASRBackend = self._load_backend(asr_config)
        logger.info(f"ASRProcessor initialized with backend: {self.backend.name}")

    def _load_backend(self, asr_config: Dict[str, Any]) -> ASRBackend:
        name = self.backend_name.lower()
        if name in {"parakeet", "parakeet_v2", "nemo_parakeet"}:
            model_name = asr_config.get("model_name", "nvidia/parakeet-tdt-0.6b-v2")
            return ParakeetV2Backend(model_name=model_name, device=self.device)
        if name in {"gemini"}:
            model_name = asr_config.get("gemini_model_name", "gemini-2.0-flash-lite")
            return GeminiASRBackend(model_name=model_name, api_key=asr_config.get("gemini_api_key"))
        if name in {"whisper_hinglish"}:
            model_name = asr_config.get("whisper_hinglish_model_name", "Oriserve/Whisper-Hindi2Hinglish-Prime")
            return WhisperHinglishBackend(model_name=model_name, device=self.device)
        raise ValueError(f"Unknown ASR backend: {self.backend_name!r}")

    def _maybe_flag(
        self,
        *,
        audio: np.ndarray,
        sr: int,
        dur: float,
        result: ASRResult,
        out_audio_path: Path,
        out_txt_path: Path,
    ) -> bool:
        if result.times is None:
            return False

        start = float(result.times.start)
        end = float(result.times.end)
        start_dev = abs(start - 0.0)
        end_dev = abs(end - dur)
        if start_dev <= self.time_thresh and end_dev <= self.time_thresh:
            return False

        pad_start = max(0.0, start - self.padding)
        pad_end = min(dur, end + self.padding)
        if pad_end <= pad_start:
            if end > start:
                pad_start, pad_end = start, end
            else:
                pad_start, pad_end = 0.0, dur

        start_frame = int(pad_start * sr)
        end_frame = int(pad_end * sr)
        crop_audio = audio
        if dur > 0 and end_frame > start_frame and start_frame >= 0 and end_frame <= len(audio):
            crop_audio = audio[start_frame:end_frame]

        if crop_audio.size == 0:
            if audio.size > 0:
                crop_audio = audio
            else:
                return False

        sf.write(str(out_audio_path), crop_audio, sr, subtype='PCM_24')
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write(result.text)
        return True

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Processes all .wav files in speaker-specific subdirectories for ASR.

        Transcribes files, checks for timestamp deviations, and saves flagged files
        (with their transcripts) to a separate 'flagged' output directory.

        Args:
            input_dir: The base input directory containing speaker subdirectories
                       (e.g., .../outlier_detector_output/SPEAKER_NAME/definite/).
            output_dir: The base output directory where ASR results, including a
                        'flagged' subdirectory for each speaker, will be saved.
        """
        speakers_to_process = self.target_speakers
        if not speakers_to_process:
            speakers_to_process = [d.name.replace("_", " ") for d in input_dir.iterdir() if d.is_dir()]
            logger.info(f"Auto-discovered {len(speakers_to_process)} speakers from {input_dir}")

        for spkr in speakers_to_process:
            spkr_dir = spkr.replace(" ", "_")
            
            in_dir = input_dir / spkr_dir / self.def_dir
            flagged_dir = output_dir / spkr_dir / self.flagged_dir
            if flagged_dir.exists(): 
                shutil.rmtree(flagged_dir)
            flagged_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Input: {in_dir.resolve()}")
            logger.info(f"Flagged output: {flagged_dir.resolve()}")

            files = sorted(list(in_dir.rglob("*.wav")))
            if not files:
                logger.warning(f"No '.wav' files found in {in_dir}")
                return

            logger.info(f"Found {len(files)} files")
            flagged = 0

            for i in tqdm(range(0, len(files), self.batch_size), desc="Processing", unit="batch"):
                batch = [str(p) for p in files[i:i + self.batch_size]]
                batch_files = files[i:i + self.batch_size]

                try:
                    results = self.backend.transcribe_batch(batch)
                except Exception as e:
                    logger.error(f"Error in batch: {e}")
                    continue

                if not isinstance(results, list) or len(results) != len(batch_files):
                    continue

                for path, result in zip(batch_files, results):
                    try:
                        audio, sr = sf.read(str(path), dtype='float32')
                        if audio.ndim > 1:
                            audio = np.mean(audio, axis=1)
                        dur = len(audio) / sr if sr > 0 else 0.0

                        flagged_audio = flagged_dir / path.name
                        flagged_txt = flagged_dir / path.with_suffix(".txt").name
                        if self._maybe_flag(
                            audio=audio,
                            sr=sr,
                            dur=dur,
                            result=result,
                            out_audio_path=flagged_audio,
                            out_txt_path=flagged_txt,
                        ):
                            flagged += 1
                    
                    except Exception as e:
                        logger.error(f"Error processing {path.name}: {e}")

            logger.info(f"\nProcessing complete for {spkr}")
            logger.info(f"Total files: {len(files)}")
            logger.info(f"Flagged: {flagged}")

if __name__ == "__main__":
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    base_dir = Path(cfg["project_setup"]["output_base_dir"]) / Path(cfg["project_setup"]["series_name"])
    in_dir = base_dir / Path(cfg["outlier_detector"]["output_folder"])
    out_dir = base_dir / Path(cfg["asr_processor"]["output_folder"])
    
    print(f"Processing ASR for {in_dir}")
    processor = ASRProcessor(cfg, cfg["asr_processor"])
    processor.process_directory(in_dir, out_dir)
