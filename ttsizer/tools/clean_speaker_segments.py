import argparse
import csv
import math
import shutil
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.fetching import LocalStrategy, fetch


def _list_wavs(root: Path) -> list[Path]:
    return sorted(root.glob("*.wav"))


def _list_speaker_dirs(final_dir: Path) -> list[Path]:
    return sorted([p for p in final_dir.iterdir() if p.is_dir() and p.name.startswith("SPEAKER_")])


def _list_wavs_recursive(root: Path) -> list[Path]:
    return sorted(root.rglob("*.wav"))


def _load_audio(path: Path, target_sr: int, resamplers: dict[int, torchaudio.transforms.Resample]) -> torch.Tensor:
    # Use soundfile directly as torchaudio.load caused crashes on some systems
    data, sr = sf.read(str(path), dtype='float32')
    wav = torch.from_numpy(data)
    
    # Ensure (channels, time) format
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.t()
        
    if wav.ndim > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = resamplers.get(sr)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            resamplers[sr] = resampler
        wav = resampler(wav)
    return wav.squeeze(0)


def _rms_db(wav: torch.Tensor) -> float:
    rms = torch.sqrt(torch.mean(wav**2) + 1e-12)
    return float(20.0 * torch.log10(rms + 1e-12))


def _active_ratio(wav: torch.Tensor, sr: int, min_rms_db: float, margin_db: float) -> float:
    frame_len = max(1, int(sr * 0.025))
    hop_len = max(1, int(sr * 0.010))
    if wav.numel() < frame_len:
        return 0.0
    frames = wav.unfold(0, frame_len, hop_len)
    frame_rms = torch.sqrt(torch.mean(frames**2, dim=1) + 1e-12)
    frame_db = 20.0 * torch.log10(frame_rms + 1e-12)
    base_db = float(torch.quantile(frame_db, 0.1))
    thresh_db = max(min_rms_db, base_db + margin_db)
    return float(torch.mean((frame_db >= thresh_db).float()))


def _measure_signal(
    wav: torch.Tensor,
    sr: int,
    min_rms_db: float,
) -> tuple[float, float, float]:
    duration = wav.numel() / float(sr)
    rms_db = _rms_db(wav)
    active_ratio = _active_ratio(wav, sr, min_rms_db, margin_db=6.0)
    return duration, rms_db, active_ratio


def _embed(
    wav: torch.Tensor,
    classifier: EncoderClassifier,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        wav = wav.to(device).unsqueeze(0)
        emb = classifier.encode_batch(wav).squeeze(0)
        if emb.ndim > 1:
            emb = emb.mean(dim=0)
        emb = torch.nn.functional.normalize(emb, dim=0)
    return emb.cpu()


def _copy_pair(wav_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wav_path, output_dir / wav_path.name)
    txt_path = wav_path.with_suffix(".txt")
    if txt_path.exists():
        shutil.copy2(txt_path, output_dir / txt_path.name)


def _write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "speaker_id",
        "relative_path",
        "similarity",
        "decision",
        "reason",
        "duration_sec",
        "rms_db",
        "active_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_classifier(
    model_source: str,
    device: torch.device,
    pymodule_file: str | None,
) -> EncoderClassifier:
    if pymodule_file:
        return EncoderClassifier.from_hparams(
            source=model_source,
            run_opts={"device": str(device)},
            pymodule_file=pymodule_file,
        )

    # Work around SpeechBrain fetching custom.py by default.
    hparams_local_path = fetch(
        filename="hyperparams.yaml",
        source=model_source,
        savedir=None,
        save_filename=None,
        local_strategy=LocalStrategy.SYMLINK,
    )
    with open(hparams_local_path, encoding="utf-8") as handle:
        hparams = load_hyperpyyaml(handle, {})

    pretrainer = hparams.get("pretrainer", None)
    if pretrainer is not None:
        pretrainer.set_collect_in(None)
        run_on_main(
            pretrainer.collect_files,
            kwargs={
                "default_source": model_source,
                "local_strategy": LocalStrategy.SYMLINK,
            },
        )
        pretrainer.load_collected()

    return EncoderClassifier(hparams["modules"], hparams, run_opts={"device": str(device)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean diarized speaker clips using speaker verification.")
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=None,
        help="Folder containing SPEAKER_* subfolders (scans all speakers).",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=None,
        help="Folder containing gold_samples/SPEAKER_* subfolders.",
    )
    parser.add_argument(
        "--brute-force-speaker",
        type=str,
        default=None,
        help="Gold speaker ID (e.g., SPEAKER_00) to compare against all speakers in final-dir. "
             "Enables brute force mode: compares one gold speaker with all candidate speakers.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="Single speaker folder with diarized segments (wav + txt).",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=None,
        help="Single speaker folder with verified reference samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write cleaned clips (default: <final>/cleaned/<speaker_id>).",
    )
    parser.add_argument(
        "--model",
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="SpeechBrain speaker embedding model.",
    )
    parser.add_argument(
        "--pymodule-file",
        default=None,
        help="Optional SpeechBrain custom module file (default: none).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for accepting a segment.",
    )
    parser.add_argument(
        "--min-sec",
        type=float,
        default=2.0,
        help="Skip segments shorter than this duration (seconds).",
    )
    parser.add_argument(
        "--min-rms-db",
        type=float,
        default=-35.0,
        help="Reject segments with RMS dB below this.",
    )
    parser.add_argument(
        "--min-active-ratio",
        type=float,
        default=0.25,
        help="Reject segments with low voiced-frame ratio.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for embeddings.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Write a CSV report with per-file decisions.",
    )
    args = parser.parse_args()

    if args.final_dir is None and args.candidate_dir is None:
        raise SystemExit("Provide --final-dir for multi-speaker mode or --candidate-dir for single-speaker mode.")
    if args.final_dir is not None and args.gold_root is None:
        raise SystemExit("Multi-speaker mode requires --gold-root.")
    if args.candidate_dir is not None and args.gold_dir is None:
        raise SystemExit("Single-speaker mode requires --gold-dir.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = _load_classifier(args.model, device, args.pymodule_file)

    def process_speaker(
        candidate_dir: Path,
        gold_dir: Path,
        output_dir: Path,
        speaker_id: str,
        report_rows: list[dict[str, str]],
    ) -> tuple[int, int, int]:
        resamplers: dict[int, torchaudio.transforms.Resample] = {}
        gold_files = _list_wavs(gold_dir)
        if not gold_files:
            print(f"[WARN] No gold wavs found in: {gold_dir}")
            return (0, 0, 0)

        gold_embs: list[torch.Tensor] = []
        for wav_path in gold_files:
            print(f"DEBUG: Processing gold file {wav_path}", flush=True)
            try:
                wav = _load_audio(wav_path, args.sample_rate, resamplers)
                duration, rms_db, active_ratio = _measure_signal(wav, args.sample_rate, args.min_rms_db)
                if duration < args.min_sec or rms_db < args.min_rms_db or active_ratio < args.min_active_ratio:
                    continue
                gold_embs.append(_embed(wav, classifier, device))
            except Exception as e:
                print(f"ERROR processing {wav_path}: {e}", flush=True)
                continue

        if not gold_embs:
            print(f"[WARN] No usable gold embeddings after filtering in: {gold_dir}")
            return (0, 0, 0)

        gold_mean = torch.stack(gold_embs).mean(dim=0)
        gold_mean = torch.nn.functional.normalize(gold_mean, dim=0)

        cand_files = _list_wavs_recursive(candidate_dir)
        if not cand_files:
            print(f"[WARN] No candidate wavs found in: {candidate_dir}")
            return (0, 0, 0)

        kept = 0
        skipped = 0
        for wav_path in cand_files:
            wav = _load_audio(wav_path, args.sample_rate, resamplers)
            duration, rms_db, active_ratio = _measure_signal(wav, args.sample_rate, args.min_rms_db)
            if duration < args.min_sec or rms_db < args.min_rms_db or active_ratio < args.min_active_ratio:
                rel_path = wav_path.relative_to(candidate_dir).as_posix()
                report_rows.append(
                    {
                        "speaker_id": speaker_id,
                        "relative_path": rel_path,
                        "similarity": "",
                        "decision": "filtered",
                        "reason": "silence_or_short",
                        "duration_sec": f"{duration:.3f}",
                        "rms_db": f"{rms_db:.2f}",
                        "active_ratio": f"{active_ratio:.3f}",
                    }
                )
                skipped += 1
                continue
            emb = _embed(wav, classifier, device)
            similarity = float(torch.dot(emb, gold_mean))
            if similarity >= args.threshold:
                rel_dir = wav_path.parent.relative_to(candidate_dir)
                _copy_pair(wav_path, output_dir / rel_dir)
                kept += 1
                decision = "kept"
                reason = ""
            else:
                skipped += 1
                decision = "rejected"
                reason = "below_threshold"
            rel_path = wav_path.relative_to(candidate_dir).as_posix()
            report_rows.append(
                {
                    "speaker_id": speaker_id,
                    "relative_path": rel_path,
                    "similarity": f"{similarity:.4f}",
                    "decision": decision,
                    "reason": reason,
                    "duration_sec": f"{duration:.3f}",
                    "rms_db": f"{rms_db:.2f}",
                    "active_ratio": f"{active_ratio:.3f}",
                }
            )
        total = len(cand_files)
        print(f"{candidate_dir.name}: processed {total}, kept {kept}, skipped {skipped}")
        return (total, kept, skipped)

    if args.brute_force_speaker is not None:
        final_dir = args.final_dir
        gold_root = args.gold_root
        if final_dir is None:
            raise SystemExit("Brute force mode requires --final-dir.")
        if gold_root is None:
            raise SystemExit("Brute force mode requires --gold-root.")

        gold_speaker_id = args.brute_force_speaker
        gold_dir = gold_root / gold_speaker_id

        if not final_dir.exists():
            raise SystemExit(f"Final dir not found: {final_dir}")
        if not gold_root.exists():
            raise SystemExit(f"Gold root not found: {gold_root}")
        if not gold_dir.exists():
            raise SystemExit(f"Gold speaker dir not found: {gold_dir}")

        speakers = _list_speaker_dirs(final_dir)
        if not speakers:
            raise SystemExit(f"No SPEAKER_* folders found in: {final_dir}")

        output_dir = args.output_dir or (final_dir / "cleaned" / gold_speaker_id)
        report_csv = args.report_csv or (final_dir / "cleaned" / "cleaning_report.csv")
        report_rows: list[dict[str, str]] = []

        # Collect all candidate files from all speakers
        resamplers: dict[int, torchaudio.transforms.Resample] = {}
        gold_files = _list_wavs(gold_dir)
        if not gold_files:
            raise SystemExit(f"No gold wavs found in: {gold_dir}")

        gold_embs: list[torch.Tensor] = []
        for wav_path in gold_files:
            print(f"DEBUG: Processing gold file {wav_path}", flush=True)
            try:
                wav = _load_audio(wav_path, args.sample_rate, resamplers)
                duration, rms_db, active_ratio = _measure_signal(wav, args.sample_rate, args.min_rms_db)
                if duration < args.min_sec or rms_db < args.min_rms_db or active_ratio < args.min_active_ratio:
                    continue
                gold_embs.append(_embed(wav, classifier, device))
            except Exception as e:
                print(f"ERROR processing {wav_path}: {e}", flush=True)
                continue

        if not gold_embs:
            raise SystemExit(f"No usable gold embeddings after filtering in: {gold_dir}")

        gold_mean = torch.stack(gold_embs).mean(dim=0)
        gold_mean = torch.nn.functional.normalize(gold_mean, dim=0)

        kept = 0
        skipped = 0
        total = 0

        # Process all candidate speakers
        for speaker_dir in speakers:
            cand_files = _list_wavs_recursive(speaker_dir)
            if not cand_files:
                print(f"[WARN] No candidate wavs found in: {speaker_dir}")
                continue

            for wav_path in cand_files:
                print(f"DEBUG: Processing candidate file {wav_path}", flush=True)
                try:
                    total += 1
                    wav = _load_audio(wav_path, args.sample_rate, resamplers)
                    duration, rms_db, active_ratio = _measure_signal(wav, args.sample_rate, args.min_rms_db)
                    if duration < args.min_sec or rms_db < args.min_rms_db or active_ratio < args.min_active_ratio:
                        rel_path = wav_path.relative_to(final_dir).as_posix()
                        report_rows.append(
                            {
                                "speaker_id": gold_speaker_id,
                                "relative_path": rel_path,
                                "similarity": "",
                                "decision": "filtered",
                                "reason": "silence_or_short",
                                "duration_sec": f"{duration:.3f}",
                                "rms_db": f"{rms_db:.2f}",
                                "active_ratio": f"{active_ratio:.3f}",
                            }
                        )
                        skipped += 1
                        continue
                    emb = _embed(wav, classifier, device)
                    similarity = float(torch.dot(emb, gold_mean))
                    if similarity >= args.threshold:
                        _copy_pair(wav_path, output_dir)
                        kept += 1
                        decision = "kept"
                        reason = ""
                    else:
                        skipped += 1
                        decision = "rejected"
                        reason = "below_threshold"
                    rel_path = wav_path.relative_to(final_dir).as_posix()
                    report_rows.append(
                        {
                            "speaker_id": gold_speaker_id,
                            "relative_path": rel_path,
                            "similarity": f"{similarity:.4f}",
                            "decision": decision,
                            "reason": reason,
                            "duration_sec": f"{duration:.3f}",
                            "rms_db": f"{rms_db:.2f}",
                            "active_ratio": f"{active_ratio:.3f}",
                        }
                    )
                except Exception as e:
                    print(f"ERROR processing candidate {wav_path}: {e}", flush=True)
                    continue

        print(f"Brute force mode: processed {total} files from all speakers, kept {kept}, skipped {skipped}")
        _write_report(report_csv, report_rows)
        print(f"Output dir: {output_dir}")
        print(f"Report CSV: {report_csv}")
        return 0

    if args.candidate_dir is not None:
        candidate_dir = args.candidate_dir
        gold_dir = args.gold_dir
        if not candidate_dir.exists():
            raise SystemExit(f"Candidate dir not found: {candidate_dir}")
        if not gold_dir.exists():
            raise SystemExit(f"Gold dir not found: {gold_dir}")
        output_dir = args.output_dir
        if output_dir is None:
            speaker_id = candidate_dir.parent.name
            output_dir = candidate_dir.parents[1] / "cleaned" / speaker_id
        report_csv = args.report_csv
        if report_csv is None:
            report_csv = output_dir.parent / "cleaning_report.csv"
        report_rows: list[dict[str, str]] = []
        process_speaker(candidate_dir, gold_dir, output_dir, output_dir.name, report_rows)
        _write_report(report_csv, report_rows)
        print(f"Output dir: {output_dir}")
        print(f"Report CSV: {report_csv}")
        return 0

    final_dir = args.final_dir
    gold_root = args.gold_root
    if final_dir is None or gold_root is None:
        raise SystemExit("Multi-speaker mode requires --final-dir and --gold-root.")
    if not final_dir.exists():
        raise SystemExit(f"Final dir not found: {final_dir}")
    if not gold_root.exists():
        raise SystemExit(f"Gold root not found: {gold_root}")

    speakers = _list_speaker_dirs(final_dir)
    if not speakers:
        raise SystemExit(f"No SPEAKER_* folders found in: {final_dir}")

    total_all = kept_all = skipped_all = 0
    report_rows: list[dict[str, str]] = []
    for speaker_dir in speakers:
        speaker_id = speaker_dir.name
        gold_dir = gold_root / speaker_id
        if not gold_dir.exists():
            print(f"[WARN] Gold dir missing for {speaker_id}: {gold_dir}")
            continue
        output_dir = args.output_dir or (final_dir / "cleaned" / speaker_id)
        total, kept, skipped = process_speaker(speaker_dir, gold_dir, output_dir, speaker_id, report_rows)
        total_all += total
        kept_all += kept
        skipped_all += skipped

    print(f"All speakers: processed {total_all}, kept {kept_all}, skipped {skipped_all}")
    report_csv = args.report_csv or (final_dir / "cleaned" / "cleaning_report.csv")
    _write_report(report_csv, report_rows)
    print(f"Report CSV: {report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
