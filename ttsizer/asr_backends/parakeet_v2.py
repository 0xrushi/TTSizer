import warnings

warnings.filterwarnings("ignore")

from typing import Any

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
import torch

from ttsizer.asr_backends.base import ASRBackend, ASRResult, ASRSegmentTimes
from ttsizer.utils.logger import get_logger

logger = get_logger("asr_parakeet_v2")


# Patch for NumPy 2.0 compatibility (required by some NeMo deps).
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }


class ParakeetV2Backend(ASRBackend):
    name = "parakeet_v2"

    def __init__(self, *, model_name: str, device: str):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load()

    def _load(self) -> None:
        logger.info(f"Loading ASR model: {self.model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name,
            map_location=self.device,
        )
        self.model.eval()

    def _get_times(self, result: Any, dur: float) -> ASRSegmentTimes | None:
        if hasattr(result, "timestamp") and isinstance(result.timestamp, dict) and "word" in result.timestamp:
            words = result.timestamp["word"]
            valid = [s for s in words if isinstance(s, dict) and "start" in s and "end" in s]
            if valid:
                start = max(0.0, min(float(s["start"]) for s in valid))
                end = min(dur, max(float(s["end"]) for s in valid))
                if end >= start:
                    return ASRSegmentTimes(start=start, end=end)
        return None

    def transcribe_batch(self, paths: list[str]) -> list[ASRResult]:
        if not paths:
            return []
        if self.model is None:
            raise RuntimeError("Parakeet model is not loaded.")

        results = self.model.transcribe(
            paths,
            batch_size=len(paths),
            timestamps=True,
            verbose=False,
        )
        if isinstance(results, tuple):
            if len(results) == 1 and isinstance(results[0], list):
                results = results[0]
            else:
                list_candidate = None
                for item in results:
                    if isinstance(item, list) and (len(item) == len(paths)):
                        list_candidate = item
                        break
                if list_candidate is not None:
                    results = list_candidate
        if not isinstance(results, list):
            raise TypeError(f"Unexpected NeMo transcribe return type: {type(results).__name__}")
        if len(results) != len(paths):
            raise ValueError(f"ASR returned {len(results)} results for {len(paths)} inputs.")

        out: list[ASRResult] = []
        for path, result in zip(paths, results):
            try:
                info = sf.info(path)
                dur = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
            except Exception:
                dur = 0.0

            if isinstance(result, str):
                out.append(ASRResult(text=result, times=None, raw=result))
                continue
            if hasattr(result, "text"):
                text = getattr(result, "text", "") or ""
                times = self._get_times(result, dur) if dur else None
                out.append(ASRResult(text=text, times=times, raw=result))
                continue
            out.append(ASRResult(text="", times=None, raw=result))
        return out
