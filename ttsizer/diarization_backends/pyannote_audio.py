import warnings

warnings.filterwarnings("ignore")

import json
from pathlib import Path

import soundfile as sf
import torch
from tqdm.auto import tqdm

from ttsizer.utils.logger import get_logger

logger = get_logger("pyannote_diarizer")


def _seconds_to_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:06.3f}"


def _safe_device(want: str) -> torch.device:
    want = (want or "").lower()
    if want.startswith("cuda") or want == "gpu":
        try:
            if torch.cuda.device_count() > 0:
                torch.empty(1, device="cuda")
                return torch.device("cuda:0")
        except Exception as e:
            logger.warning(f"CUDA requested but unavailable; falling back to CPU ({type(e).__name__}: {e})")
    return torch.device("cpu")


def _iter_diarization_tracks(diarization):
    if hasattr(diarization, "itertracks"):
        return diarization.itertracks(yield_label=True)
    if isinstance(diarization, dict):
        candidate = (
            diarization.get("speaker_diarization")
            or diarization.get("diarization")
            or diarization.get("annotation")
        )
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate.itertracks(yield_label=True)
    for attr in ("speaker_diarization", "diarization", "annotation", "exclusive_speaker_diarization"):
        candidate = getattr(diarization, attr, None)
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate.itertracks(yield_label=True)
    raise AttributeError("Diarization output does not expose itertracks()")


class PyannoteAudioDiarizer:
    """
    Speaker diarization using `pyannote.audio`.

    Produces segments with `speaker` labels like `SPEAKER_00`, `SPEAKER_01`, ...
    Transcript is left as `null` because pyannote is diarization-only.
    """

    name = "pyannote_audio"

    def __init__(
        self,
        *,
        model_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: str | None = None,
        device: str = "cuda",
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        skip_if_output_exists: bool = True,
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = _safe_device(device)
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.skip_if_output_exists = bool(skip_if_output_exists)

        try:
            from pyannote.audio import Pipeline  # type: ignore
        except Exception as e:
            raise ImportError(
                "pyannote.audio is not installed. Install it (and its deps) to use the pyannote diarizer."
            ) from e

        kwargs = {}
        if self.hf_token:
            kwargs["use_auth_token"] = self.hf_token
        # Try-catch for the kwarg name change in recent libraries
        logger.info(f"Loading pyannote pipeline: {self.model_name}")
        try:
            self.pipeline = Pipeline.from_pretrained(self.model_name, **kwargs)
        except TypeError as e:
            if "use_auth_token" in str(e):
                logger.info("Retrying Pipeline.from_pretrained with 'token' instead of 'use_auth_token'...")
                if "use_auth_token" in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                self.pipeline = Pipeline.from_pretrained(self.model_name, **kwargs)
            else:
                raise e
        
        try:
            self.pipeline.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to move pyannote pipeline to {self.device}: {type(e).__name__}: {e}")

    def _process_file(self, audio_path: Path, out_path: Path) -> None:
        incomplete_marker = out_path.with_suffix(out_path.suffix + ".incomplete")
        if self.skip_if_output_exists and out_path.exists() and not incomplete_marker.exists():
            return

        try:
            info = sf.info(str(audio_path))
            duration_seconds = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
        except Exception:
            duration_seconds = 0.0

        try:
            diarization = self.pipeline(
                {"audio": str(audio_path)},
                num_speakers=self.num_speakers,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
        except Exception as e:
            logger.error(f"pyannote diarization failed for {audio_path.name}: {type(e).__name__}: {e}")
            incomplete_marker.write_text(
                json.dumps({"error": str(e)}, indent=2),
                encoding="utf-8",
            )
            return

        segments: list[dict] = []
        for segment, _, label in _iter_diarization_tracks(diarization):
            start = float(segment.start)
            end = float(segment.end)
            if duration_seconds:
                start = max(0.0, min(start, duration_seconds))
                end = max(0.0, min(end, duration_seconds))
            if end <= start:
                continue
            segments.append(
                {
                    "start": _seconds_to_ts(start),
                    "end": _seconds_to_ts(end),
                    "speaker": str(label),
                    "transcript": None,
                }
            )

        out_path.write_text(json.dumps(segments, indent=4, ensure_ascii=False), encoding="utf-8")
        if incomplete_marker.exists():
            incomplete_marker.unlink()

    def process_file(self, audio_path: Path, out_path: Path) -> None:
        self._process_file(audio_path, out_path)

    def process_directory(self, norm_dir: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(list(norm_dir.glob("*.flac")))
        if not files:
            files = sorted(list(norm_dir.glob("*.wav")))
        if not files:
            logger.warning(f"No audio files found in {norm_dir}")
            return

        for audio_path in tqdm(files, desc="pyannote diarization", unit="file"):
            out_path = out_dir / audio_path.with_suffix(".json").name
            self._process_file(audio_path, out_path)
