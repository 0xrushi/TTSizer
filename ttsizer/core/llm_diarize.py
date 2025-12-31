import warnings
warnings.filterwarnings("ignore")

import os
import json
from pathlib import Path
from typing import List, Dict, Any, TYPE_CHECKING
from tqdm.auto import tqdm
from dotenv import load_dotenv
import yaml
import re
import soundfile as sf
from google import genai
from google.genai import types as genai_types
from ttsizer.utils.logger import get_logger

if TYPE_CHECKING:
    from ttsizer.diarization_backends.pyannote_audio import PyannoteAudioDiarizer

load_dotenv()

logger = get_logger("llm_diarizer")

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _skip_ws(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _parse_json_array_lenient(text: str) -> tuple[list[Any], bool]:
    """
    Parse a JSON array from model output.

    If the JSON is truncated, returns the longest prefix of complete elements.
    Returns (elements, is_complete).
    """
    cleaned = _strip_code_fences(text)

    # Fast path: valid JSON as-is.
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data, True
    except json.JSONDecodeError:
        pass

    # Locate first array start.
    start = cleaned.find("[")
    if start == -1:
        raise json.JSONDecodeError("No JSON array found", cleaned, 0)

    decoder = json.JSONDecoder()
    idx = start + 1
    items: list[Any] = []

    while True:
        idx = _skip_ws(cleaned, idx)
        if idx >= len(cleaned):
            return items, False
        if cleaned[idx] == "]":
            return items, True

        try:
            item, next_idx = decoder.raw_decode(cleaned, idx)
        except json.JSONDecodeError:
            return items, False
        items.append(item)
        idx = _skip_ws(cleaned, next_idx)
        if idx >= len(cleaned):
            return items, False
        if cleaned[idx] == ",":
            idx += 1
            continue
        if cleaned[idx] == "]":
            return items, True

        # Unexpected token -> treat as truncated/invalid tail; keep what we have.
        return items, False


def _ts_to_seconds(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    raise ValueError(f"Invalid timestamp format: {ts!r}")

def _safe_ts_to_seconds(ts: Any) -> float | None:
    if not isinstance(ts, str):
        return None
    try:
        return _ts_to_seconds(ts)
    except Exception:
        return None


def _seconds_to_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:06.3f}"

def _max_end_seconds(segments: list[Dict[str, Any]]) -> float | None:
    max_end: float | None = None
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        end = seg.get("end")
        end_sec = _safe_ts_to_seconds(end)
        if end_sec is None:
            continue
        if max_end is None or end_sec > max_end:
            max_end = end_sec
    return max_end


def _next_speaker_number(segments: list[Dict[str, Any]]) -> int:
    max_n = 0
    for seg in segments:
        spk = str(seg.get("speaker", ""))
        if spk.startswith("Speaker "):
            try:
                n = int(spk.split("Speaker ", 1)[1])
                max_n = max(max_n, n)
            except ValueError:
                continue
    return max_n + 1


class LLMDiarizer:
    """Handles diarization of audio files using a large language model (LLM).
    
    Uses a specified Gemini model to analyze audio and generate speaker diarization data.
    It formats prompts based on a template and project-specific details, processes audio
    files by uploading them to the LLM, and saves the resulting JSON output.
    """
    def __init__(self, global_config: Dict[str, Any], diarizer_config: Dict[str, Any]):
        """Initializes the LLMDiarizer with global and diarizer-specific configurations.

        Args:
            global_config: Dictionary containing global project setup information.
            diarizer_config: Dictionary containing configuration specific to the diarizer.

        Raises:
            FileNotFoundError: If the prompt template file specified in config is not found.
        """
        self.proj = global_config["project_setup"]

        self.backend = str(diarizer_config.get("backend", "gemini")).lower()
        self._pyannote_backend: "PyannoteAudioDiarizer | None" = None

        self.model_name = diarizer_config.get("model_name", "gemini-2.0-flash-lite")
        self.temperature = diarizer_config.get("temperature", 0.1)
        self.top_p = diarizer_config.get("top_p", 0.8)
        self.max_output_tokens = diarizer_config.get("max_output_tokens")
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.skip = diarizer_config.get("skip_if_output_exists", True)
        self.save_raw = diarizer_config.get("save_raw_llm_output", True)
        self.compact_output = bool(diarizer_config.get("compact_output", True))
        self.continue_on_truncation = bool(diarizer_config.get("continue_on_truncation", False))
        self.max_continuations = int(diarizer_config.get("max_continuations", 3))
        self.reuse_uploaded_files = bool(diarizer_config.get("reuse_uploaded_files", True))
        self.continuation_window_seconds = diarizer_config.get("continuation_window_seconds", 600)
        self.max_no_progress_retries = int(diarizer_config.get("max_no_progress_retries", 2))
        self.no_progress_window_seconds = diarizer_config.get("no_progress_window_seconds", 120)

        if self.backend in {"pyannote", "pyannote_audio"}:
            from ttsizer.diarization_backends.pyannote_audio import PyannoteAudioDiarizer

            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN not found in environment variables")

            self._pyannote_backend = PyannoteAudioDiarizer(
                model_name=diarizer_config.get("pyannote_model_name", "pyannote/speaker-diarization-3.1"),
                hf_token=hf_token,
                device=diarizer_config.get("device", "cuda"),
                num_speakers=diarizer_config.get("num_speakers"),
                min_speakers=diarizer_config.get("min_speakers"),
                max_speakers=diarizer_config.get("max_speakers"),
                skip_if_output_exists=self.skip,
            )
            self.tmpl_file = None
            self.prompt = ""
        else:
            self.tmpl_file = Path(diarizer_config["prompt_template_file"])
            if not self.tmpl_file.exists():
                raise FileNotFoundError(f"Template not found: {self.tmpl_file}")
            labels = self.proj.get("target_speaker_labels") or []
            self.prompt = self._format_prompt(self.proj["series_name"], labels)

    def _build_prompt(
        self,
        *,
        resume_from: str | None = None,
        window_end: str | None = None,
        audio_duration: str | None = None,
        known_speakers: list[str] | None = None,
        next_speaker_number: int | None = None,
        strict_window: bool = False,
    ) -> str:
        prompt = self.prompt
        if audio_duration:
            prompt += (
                "\n\n## Audio Metadata\n"
                f"- Actual audio duration: `{audio_duration}` (MM:SS.mmm)\n"
                "- Process the full duration; do not stop early due to 'typical episode length'.\n"
            )
        if self.compact_output:
            prompt += (
                "\n\n## Output Constraints (Important)\n"
                "- Prefer *no* `SOUND` segments except OP/ED or long non-speech blocks (>10s).\n"
                "- If you include `SOUND`, merge consecutive `SOUND` into one and never emit back-to-back `SOUND`.\n"
                "- Do NOT try to cover every second of silence with `SOUND` filler.\n"
                "- Avoid extremely short segments; prefer fewer, longer segments (aim for <= 400 total segments).\n"
                "- Merge adjacent segments by the same speaker when reasonable.\n"
                "- Output MUST be a single valid JSON array, no markdown fences.\n"
            )
        if resume_from:
            prompt += (
                "\n\n## Continuation\n"
                f"You already processed the episode up to `{resume_from}`.\n"
                f"Continue diarization strictly from `{resume_from}`.\n"
                f"Do NOT repeat or revise earlier segments; the first segment you output must start at or after `{resume_from}`.\n"
            )
            if window_end:
                prompt += (
                    f"Only output segments in the time window `{resume_from}` to `{window_end}`.\n"
                    f"Do NOT output any segments before `{resume_from}`.\n"
                )
                if strict_window:
                    prompt += (
                        f"Do NOT output any segments after `{window_end}`; "
                        f"we will request the next window later.\n"
                    )
            if next_speaker_number is not None:
                prompt += (
                    f"If you need a new generic speaker label, continue numbering from `Speaker {next_speaker_number}`.\n"
                )
            if known_speakers:
                prompt += "\nKnown speaker labels to reuse exactly:\n" + "\n".join(
                    f"- {s}" for s in known_speakers
                )
                prompt += "\n"
        return prompt

    def _get_or_upload(
        self, client: genai.Client, *, norm_path: Path, cache_path: Path
    ):
        sig = {"size": norm_path.stat().st_size, "mtime_ns": norm_path.stat().st_mtime_ns}

        if self.reuse_uploaded_files and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("sig") == sig and "uri" in cached and "mime_type" in cached:
                    return cached["uri"], cached["mime_type"]
            except Exception:
                pass

        logger.info(f"Uploading {norm_path.name}...")
        uploaded = client.files.upload(file=str(norm_path))
        logger.info(f"Uploaded: {uploaded.uri}")

        if self.reuse_uploaded_files:
            try:
                cache_path.write_text(
                    json.dumps(
                        {"sig": sig, "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        return uploaded.uri, uploaded.mime_type

    def _log_response_meta(self, response: Any, *, norm_path: Path, part: int):
        finish_reason = None
        token_counts: Dict[str, Any] = {}
        try:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                cand0 = candidates[0]
                finish_reason = getattr(cand0, "finish_reason", None) or getattr(cand0, "finishReason", None)
        except Exception:
            finish_reason = None

        usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
        if usage is not None:
            for k in [
                "prompt_token_count",
                "candidates_token_count",
                "total_token_count",
                "promptTokenCount",
                "candidatesTokenCount",
                "totalTokenCount",
            ]:
                v = getattr(usage, k, None)
                if v is not None:
                    token_counts[k] = v

        if finish_reason or token_counts:
            logger.info(
                f"LLM response meta for {norm_path.name} part {part}: "
                f"finish_reason={finish_reason!s} tokens={token_counts if token_counts else 'n/a'}"
            )
    
    def _format_prompt(self, series_name: str, target_speaker_labels: List[str] | None) -> str:
        """Formats the prompt string using a template and provided speaker labels.

        Args:
            series_name: The name of the series or project.
            target_speaker_labels: A list of target speaker labels for diarization, or None.

        Returns:
            The formatted prompt string.
        """
        template_text = self.tmpl_file.read_text(encoding='utf-8')
        
        if target_speaker_labels:
            template_chars = "[" + ", ".join(target_speaker_labels) + "]"
            char_1 = target_speaker_labels[0]
        else:
            template_chars = "All distinct speakers found in the audio"
            char_1 = "Speaker 1"

        prompt = (
            template_text
            .replace("{SERIES_NAME}", series_name)
            .replace("{TARGET_SPEAKER_LABELS}", template_chars)
            .replace("{CHARACTER_1}", char_1)
        )
        logger.info(
            f"Loaded diarization prompt template from {self.tmpl_file} ({len(prompt)} chars)"
        )
        logger.debug(prompt)
        return prompt
    
    def _process_file(self, norm_path: Path, out_path: Path):
        """Processes a single normalized audio file to generate diarization data.

        Uploads the audio file, sends it to the LLM with the formatted prompt,
        and saves the JSON response. Handles potential JSON decoding errors.

        Args:
            norm_path: Path to the normalized input audio file.
            out_path: Path to save the output JSON diarization data.
        """
        if self.backend in {"pyannote", "pyannote_audio"}:
            assert self._pyannote_backend is not None
            self._pyannote_backend.process_file(norm_path, out_path)
            return

        if not self.api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY; set it in your environment (or in .env) to run llm_diarizer."
            )

        incomplete_marker = out_path.with_suffix(out_path.suffix + ".incomplete")
        existing_segments: list[Dict[str, Any]] = []
        if out_path.exists() and incomplete_marker.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    existing_segments = existing
            except Exception:
                existing_segments = []

        info = sf.info(str(norm_path))
        duration_seconds = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
        duration_ts = _seconds_to_ts(duration_seconds) if duration_seconds else None

        client = genai.Client(api_key=self.api_key)
        upload_cache_path = out_path.with_suffix(".uploaded.json")
        file_uri, mime_type = self._get_or_upload(client, norm_path=norm_path, cache_path=upload_cache_path)

        config_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "safety_settings": [
                genai_types.SafetySetting(category=c, threshold="BLOCK_NONE")
                for c in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                ]
            ],
            "response_mime_type": "application/json",
            "top_p": self.top_p,
        }
        if self.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = int(self.max_output_tokens)
        config = genai_types.GenerateContentConfig(**config_kwargs)

        segments_accum: list[Dict[str, Any]] = list(existing_segments)
        no_progress_retries = 0
        for attempt in range(self.max_continuations + 1):
            known_speakers = sorted({str(s.get("speaker")) for s in segments_accum if s.get("speaker")})
            next_speaker_number = _next_speaker_number(segments_accum) if segments_accum else None
            resume_from_ts = None
            resume_from_sec = None
            if segments_accum:
                resume_from_sec = _max_end_seconds(segments_accum)
                resume_from_ts = _seconds_to_ts(resume_from_sec) if resume_from_sec is not None else None

            prev_last_end_sec = resume_from_sec
            window_end_ts = None
            if resume_from_sec is not None and duration_seconds:
                try:
                    cont_window = float(
                        self.no_progress_window_seconds
                        if no_progress_retries > 0
                        else self.continuation_window_seconds
                    )
                except Exception:
                    cont_window = 0.0
                if cont_window > 0.0:
                    window_end_ts = _seconds_to_ts(min(duration_seconds, resume_from_sec + cont_window))
            prompt_text = (
                self._build_prompt(
                    resume_from=resume_from_ts,
                    window_end=window_end_ts,
                    strict_window=bool(window_end_ts),
                    audio_duration=duration_ts,
                    known_speakers=known_speakers,
                    next_speaker_number=next_speaker_number,
                )
                if resume_from_ts
                else self._build_prompt(audio_duration=duration_ts)
            )
            contents = [
                genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part.from_uri(
                            file_uri=file_uri,
                            mime_type=mime_type,
                        ),
                        genai_types.Part.from_text(text=prompt_text),
                    ],
                )
            ]

            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            self._log_response_meta(response, norm_path=norm_path, part=attempt + 1)
            
            json_text = ""
            try:
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    json_text = response.candidates[0].content.parts[0].text or ""
            except Exception:
                pass

            if not json_text:
                logger.warning(f"Empty or blocked response for {norm_path.name} (part {attempt + 1}). Checking finish reason...")
                break

            if self.save_raw:
                raw_name = f"{out_path.stem}.raw_llm_output.part{attempt + 1}.txt"
                (out_path.parent / raw_name).write_text(json_text, encoding="utf-8")

            try:
                new_segments_any, is_complete = _parse_json_array_lenient(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Could not parse LLM output for {norm_path.name} (part {attempt + 1}): {e}")
                break

            new_segments: list[Dict[str, Any]] = [
                s for s in new_segments_any if isinstance(s, dict)
            ]

            if resume_from_sec is not None:
                filtered: list[Dict[str, Any]] = []
                for seg in new_segments:
                    start = seg.get("start")
                    end = seg.get("end")
                    if not isinstance(start, str):
                        continue
                    try:
                        start_sec = _ts_to_seconds(start)
                        end_sec = _ts_to_seconds(end) if isinstance(end, str) else None
                    except Exception:
                        continue
                    # Be tolerant of slight overlap on continuation responses to avoid stalling
                    # when the model repeats a small amount of context.
                    if start_sec + 0.001 >= resume_from_sec - 2.0 or (
                        end_sec is not None and end_sec >= resume_from_sec + 0.001
                    ):
                        filtered.append(seg)
                new_segments = filtered

            existing_keys = {
                (
                    s.get("start"),
                    s.get("end"),
                    s.get("speaker"),
                    s.get("transcript"),
                )
                for s in segments_accum
                if isinstance(s, dict)
            }
            for seg in new_segments:
                key = (
                    seg.get("start"),
                    seg.get("end"),
                    seg.get("speaker"),
                    seg.get("transcript"),
                )
                if key in existing_keys:
                    continue
                segments_accum.append(seg)
                existing_keys.add(key)

            def _sort_key(seg: Any) -> float:
                if not isinstance(seg, dict):
                    return float("inf")
                start_sec = _safe_ts_to_seconds(seg.get("start"))
                return start_sec if start_sec is not None else float("inf")

            segments_accum.sort(key=_sort_key)
            out_path.write_text(json.dumps(segments_accum, indent=4, ensure_ascii=False), encoding="utf-8")

            last_end_sec = _max_end_seconds(segments_accum)

            if (
                prev_last_end_sec is not None
                and last_end_sec is not None
                and last_end_sec <= prev_last_end_sec + 0.05
            ):
                if self.continue_on_truncation and no_progress_retries < self.max_no_progress_retries:
                    logger.warning(
                        f"No progress extending {norm_path.name} beyond {resume_from_ts}; "
                        f"retrying with a smaller continuation window."
                    )
                    no_progress_retries += 1
                    continue
                logger.warning(
                    f"No progress extending {norm_path.name} beyond {resume_from_ts}; stopping continuation."
                )
                break
            no_progress_retries = 0

            is_done = bool(last_end_sec is not None and duration_seconds and last_end_sec >= duration_seconds - 0.5)
            if is_done:
                if incomplete_marker.exists():
                    incomplete_marker.unlink()
                return

            if not is_complete:
                logger.warning(
                    f"LLM output for {norm_path.name} appears truncated; accumulated {len(segments_accum)} segments "
                    f"up to {segments_accum[-1].get('end') if segments_accum else 'unknown'}."
                )

            incomplete_marker.write_text(
                json.dumps(
                    {
                        "duration_seconds": duration_seconds,
                        "last_end": _seconds_to_ts(last_end_sec) if last_end_sec is not None else None,
                        "next_speaker_number": _next_speaker_number(segments_accum),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            if not self.continue_on_truncation:
                break

            if not segments_accum:
                break

        logger.warning(
            f"Stopped diarization early for {norm_path.name}. Partial JSON saved to {out_path.name} "
            f"(marker: {incomplete_marker.name})."
        )

    def process_directory(self, norm_dir: Path, out_dir: Path):
        """Processes all audio files (FLAC or WAV) in a given directory for diarization.

        Iterates through audio files, calls _process_file for each, and saves outputs.

        Args:
            norm_dir: Path to the directory containing normalized audio files.
            out_dir: Path to the directory where output JSON files will be saved.
        """
        if self.backend in {"pyannote", "pyannote_audio"}:
            assert self._pyannote_backend is not None
            self._pyannote_backend.process_directory(norm_dir, out_dir)
            return

        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(norm_dir.glob("*.flac")))
        if not files:
            files = sorted(list(norm_dir.glob("*.wav")))
        
        if not files:
            logger.warning(f"No files found in {norm_dir}")
            return
        
        logger.info(f"Processing {len(files)} files in {norm_dir}")

        for norm_path in tqdm(files, desc="Processing files", unit="file"):
            out_path = out_dir / norm_path.with_suffix(".json").name
            incomplete_marker = out_path.with_suffix(out_path.suffix + ".incomplete")
            if self.skip and out_path.exists() and not incomplete_marker.exists():
                continue
            raw_path = out_path.with_suffix(".raw_llm_output.txt")
            if raw_path.exists() and not out_path.exists():
                try:
                    recovered, is_complete = _parse_json_array_lenient(raw_path.read_text(encoding="utf-8"))
                    try:
                        out_path.write_text(
                            json.dumps(recovered, indent=4, ensure_ascii=False), encoding="utf-8"
                        )
                    except PermissionError as e:
                        logger.error(f"Permission error writing {out_path}: {e}")
                        continue
                    if not is_complete:
                        logger.warning(
                            f"Recovered partial JSON from {raw_path.name} ({len(recovered)} segments)."
                        )
                        incomplete_marker.write_text(
                            json.dumps({"last_end": recovered[-1].get("end") if recovered else None}, indent=2),
                            encoding="utf-8",
                        )
                        if self.continue_on_truncation:
                            self._process_file(norm_path, out_path)
                    continue
                except json.JSONDecodeError:
                    pass
            self._process_file(norm_path, out_path)
        

if __name__ == '__main__':
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    base = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["series_name"]
    in_dir = base / cfg["vocals_normalizer"]["output_folder"]
    out_dir = base / cfg["llm_diarizer"]["output_folder"]
    
    print(f"Generating transcriptions for {in_dir}")
    diarizer = LLMDiarizer(global_config=cfg, diarizer_config=cfg["llm_diarizer"])
    diarizer.process_directory(in_dir, out_dir) 
