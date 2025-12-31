import argparse
import json
from pathlib import Path
from typing import Any

import soundfile as sf


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


def _seconds_to_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:06.3f}"


def _max_end_seconds(segments: list[dict[str, Any]]) -> float | None:
    max_end: float | None = None
    for seg in segments:
        end = seg.get("end")
        if not isinstance(end, str):
            continue
        try:
            end_sec = _ts_to_seconds(end)
        except Exception:
            continue
        if max_end is None or end_sec > max_end:
            max_end = end_sec
    return max_end


def _find_audio_files(audio_dir: Path) -> list[Path]:
    files = sorted(audio_dir.glob("*.flac"))
    if files:
        return files
    files = sorted(audio_dir.glob("*.wav"))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate llm_diarizer JSON coverage and incomplete markers.")
    parser.add_argument(
        "--series-dir",
        type=Path,
        required=True,
        help="Series output directory (contains vocals_normalized/ and transcriptions/).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Override audio dir (default: <series-dir>/vocals_normalized).",
    )
    parser.add_argument(
        "--transcriptions-dir",
        type=Path,
        default=None,
        help="Override transcriptions dir (default: <series-dir>/transcriptions).",
    )
    args = parser.parse_args()

    series_dir: Path = args.series_dir
    audio_dir: Path = args.audio_dir or (series_dir / "vocals_normalized")
    trans_dir: Path = args.transcriptions_dir or (series_dir / "transcriptions")

    if not audio_dir.exists():
        raise SystemExit(f"Audio dir not found: {audio_dir}")
    if not trans_dir.exists():
        raise SystemExit(f"Transcriptions dir not found: {trans_dir}")

    audio_files = _find_audio_files(audio_dir)
    if not audio_files:
        raise SystemExit(f"No audio files found in: {audio_dir}")

    rows: list[tuple[str, int | None, str | None, str | None, float | None, bool, str]] = []
    has_issues = False

    for audio_path in audio_files:
        ep_stem = audio_path.stem
        json_path = trans_dir / f"{ep_stem}.json"
        incomplete_path = json_path.with_suffix(json_path.suffix + ".incomplete")

        duration_seconds = 0.0
        try:
            info = sf.info(str(audio_path))
            duration_seconds = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
        except Exception:
            duration_seconds = 0.0

        seg_count = None
        last_end_ts = None
        coverage = None
        note = ""

        if not json_path.exists():
            has_issues = True
            note = "missing_json"
        else:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    seg_count = len(data)
                    last_end_sec = _max_end_seconds([s for s in data if isinstance(s, dict)])
                    if last_end_sec is not None:
                        last_end_ts = _seconds_to_ts(last_end_sec)
                        if duration_seconds > 0:
                            coverage = max(0.0, min(1.0, last_end_sec / duration_seconds))
            except Exception as e:
                has_issues = True
                note = f"bad_json:{type(e).__name__}"

        is_incomplete = incomplete_path.exists()
        if is_incomplete:
            has_issues = True

        duration_ts = _seconds_to_ts(duration_seconds) if duration_seconds else None
        rows.append(
            (
                ep_stem,
                seg_count,
                last_end_ts,
                duration_ts,
                coverage,
                is_incomplete,
                note,
            )
        )

    name_w = min(60, max(len(r[0]) for r in rows))
    print(f"{'episode'.ljust(name_w)}  segs  last_end   duration   cov    incomplete  note")
    for ep, segs, last_end, dur, cov, inc, note in rows:
        segs_s = f"{segs}" if segs is not None else "-"
        last_end_s = last_end or "-"
        dur_s = dur or "-"
        cov_s = f"{cov*100:5.1f}%" if cov is not None else "  -  "
        inc_s = "YES" if inc else "no"
        print(
            f"{ep.ljust(name_w)}  {segs_s.rjust(4)}  {last_end_s.rjust(8)}  {dur_s.rjust(8)}  {cov_s.rjust(6)}  {inc_s.rjust(10)}  {note}"
        )

    if has_issues:
        print("\nIssues found. Re-run `python -m ttsizer.main` with `llm_diarizer` enabled to resume incomplete files.")
        return 1

    print("\nAll diarization JSONs look complete (no `.incomplete` markers).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

