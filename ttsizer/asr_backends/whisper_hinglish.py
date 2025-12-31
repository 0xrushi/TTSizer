import torch
import numpy as np
import ffmpeg
from typing import List, Dict, Tuple
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutomaticSpeechRecognitionPipeline
from silero_vad import get_speech_timestamps, load_silero_vad
from ttsizer.asr_backends.base import ASRResult, ASRSegmentTimes
from ttsizer.utils.logger import get_logger

logger = get_logger("whisper_hinglish")

class WhisperHinglishBackend:
    """ASR Backend using a fine-tuned Whisper model for Hinglish (Hindi+English) transcription."""
    
    name = "whisper_hinglish"

    def __init__(self, model_name: str, device: str = "cuda"):
        """Initializes the Whisper Hinglish backend.

        Args:
            model_name: The Hugging Face model ID (e.g., "Oriserve/Whisper-Hindi2Hinglish-Prime").
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.model_id = model_name
        self.device = device
        
        logger.info(f"Loading Whisper Hinglish model: {self.model_id} on {self.device}")
        
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)

        # The model seems to use 'en' token for Hinglish based on the original script
        self.model.generation_config.language = "en" 
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.no_timestamps_token_id = 50364 

        self.processor = WhisperProcessor.from_pretrained(self.model_id)

        self.pipe = AutomaticSpeechRecognitionPipeline(
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            # Pipeline expects int device id or -1 for cpu
            device=0 if self.device == "cuda" else -1, 
            generate_kwargs={
                "task": "transcribe",
                "language": "en",
                "generation_config": self.model.generation_config
            }
        )
        
        self.vad_model = load_silero_vad()

    def _load_audio(self, audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Loads audio from any format using ffmpeg and converts it to a NumPy array.

        Args:
            audio_path: Path to the audio file.
            target_sr: Target sampling rate (default 16000 Hz).

        Returns:
            A tuple containing the audio data as a float32 NumPy array and the sampling rate.
        
        Raises:
            RuntimeError: If ffmpeg fails to load the audio.
        """
        try:
            out, _ = (
                ffmpeg
                .input(audio_path)
                .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=target_sr)
                .run(capture_stdout=True, capture_stderr=True)
            )
            audio = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
            return audio, target_sr
        except Exception as e:
            logger.error(f"ffmpeg failed to load '{audio_path}': {e}")
            raise RuntimeError(f"ffmpeg failed to load '{audio_path}'") from e

    def _merge_vad_segments(self, segments: List[Dict[str, int]], max_pause: float = 0.5, sr: int = 16000) -> List[Dict[str, int]]:
        """Merges VAD segments that are separated by short pauses.

        Args:
            segments: List of speech segments (dictionaries with 'start' and 'end' sample indices).
            max_pause: Maximum pause duration (in seconds) allowed within a merged segment.
            sr: Sampling rate.

        Returns:
            A list of merged speech segments.
        """
        if not segments:
            return []
        
        merged: List[Dict[str, int]] = []
        current = segments[0].copy()
        for seg in segments[1:]:
            pause = (seg['start'] - current['end']) / sr
            if pause <= max_pause:
                current['end'] = seg['end']
            else:
                merged.append(current)
                current = seg.copy()

        merged.append(current)
        return merged

    def transcribe_batch(self, paths: List[str]) -> List[ASRResult]:
        """Transcribes a batch of audio files.

        Applies VAD to identify speech segments and then transcribes them using the Whisper pipeline.

        Args:
            paths: List of file paths to transcribe.

        Returns:
            A list of ASRResult objects corresponding to the input paths.
        """
        results: List[ASRResult] = []
        for path in paths:
            if not isinstance(path, str):
                logger.warning(f"Invalid path type in batch: {type(path)}. Skipping.")
                results.append(ASRResult(text="", times=None))
                continue

            try:
                # 1. Load Audio
                audio, sr = self._load_audio(path)
                
                # 2. VAD
                speech_segments = get_speech_timestamps(torch.from_numpy(audio), self.vad_model, sampling_rate=sr)
                merged_segments = self._merge_vad_segments(speech_segments, max_pause=0.5, sr=sr)
                
                full_text = ""
                min_start = float('inf')
                max_end = float('-inf')
                
                if not merged_segments:
                    # If no speech detected by VAD, return empty result (silence)
                    results.append(ASRResult(text="", times=None))
                    continue

                # 3. Transcribe Segments
                for seg in merged_segments:
                    start_sample = seg['start']
                    end_sample = seg['end']
                    segment_audio = audio[start_sample:end_sample]
                    
                    # Skip extremely short segments (< 0.01s)
                    if len(segment_audio) < 160: 
                         continue

                    out = self.pipe(segment_audio, return_timestamps='segment')
                    seg_text = out['text']
                    full_text += seg_text + " "
                    
                    start_sec = start_sample / sr
                    end_sec = end_sample / sr
                    
                    if start_sec < min_start: min_start = start_sec
                    if end_sec > max_end: max_end = end_sec

                # 4. Result
                if min_start == float('inf'):
                     results.append(ASRResult(text="", times=None))
                else:
                    results.append(ASRResult(
                        text=full_text.strip(),
                        times=ASRSegmentTimes(start=min_start, end=max_end)
                    ))

            except Exception as e:
                logger.error(f"Error transcribing {path}: {e}")
                results.append(ASRResult(text="", times=None)) # Return empty on failure
        
        return results