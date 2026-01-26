"""
Script: build_tts_hf_dataset.py

Purpose:
    Build a Hugging Face TTS dataset from paired audio (.wav) and text (.txt) files.

Description:
    This script recursively scans a directory for paired .wav and .txt files,
    validates the pairs, builds a Hugging Face Dataset in Parquet format following
    the same conventions as the MrDragonFox/Elise dataset, and optionally
    uploads it to the Hugging Face Hub.

Usage:
    python build_tts_hf_dataset.py \
        --source "/path/to/aligned_clips" \
        --output "./dataset_output" \
        --repo-name "username/dataset-name" \
        --license "mit" \
        --skip-upload
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json

from datasets import Dataset, Audio, DatasetDict
from huggingface_hub import HfApi, create_repo
import pandas as pd
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("build_tts_hf_dataset")


def scan_for_pairs(source_dir: Path) -> Tuple[List[Path], List[Path], Dict[str, Any]]:
    """
    Recursively scan for .wav and .txt files and validate pairs.

    Args:
        source_dir: Root directory to scan.

    Returns:
        Tuple of (valid_pairs, missing_pairs, validation_report)
    """
    logger.info(f"Scanning directory: {source_dir}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    wav_files = sorted(list(source_dir.rglob("*.wav")))
    txt_files = sorted(list(source_dir.rglob("*.txt")))

    logger.info(f"Found {len(wav_files)} .wav files")
    logger.info(f"Found {len(txt_files)} .txt files")

    # Create lookup for txt files
    txt_lookup = {txt_file.stem: txt_file for txt_file in txt_files}

    valid_pairs = []
    missing_wav = []
    missing_txt = []
    empty_transcripts = []

    for wav_path in tqdm(wav_files, desc="Validating pairs"):
        stem = wav_path.stem
        txt_path = txt_lookup.get(stem)

        if txt_path is None:
            missing_txt.append((wav_path, stem))
            continue

        # Check if txt file is empty
        try:
            text_content = txt_path.read_text(encoding="utf-8").strip()
            if not text_content:
                empty_transcripts.append((wav_path, txt_path))
                continue
        except Exception as e:
            logger.warning(f"Could not read text file {txt_path}: {e}")
            missing_txt.append((wav_path, stem))
            continue

        valid_pairs.append((wav_path, txt_path, text_content))

    # Check for txt files without matching wav
    wav_stems = {wav_path.stem for wav_path in wav_files}
    for txt_path in txt_files:
        if txt_path.stem not in wav_stems:
            missing_wav.append((txt_path, txt_path.stem))

    validation_report = {
        "total_wav_files": len(wav_files),
        "total_txt_files": len(txt_files),
        "valid_pairs": len(valid_pairs),
        "missing_txt": len(missing_txt),
        "missing_wav": len(missing_wav),
        "empty_transcripts": len(empty_transcripts),
    }

    return valid_pairs, [missing_txt, missing_wav, empty_transcripts], validation_report


def log_validation_issues(issues: List[List[Tuple[Path, Any]]]) -> None:
    """
    Log validation issues found during scanning.

    Args:
        issues: List of issue tuples containing missing_txt, missing_wav, empty_transcripts.
    """
    missing_txt, missing_wav, empty_transcripts = issues

    if missing_txt:
        logger.warning(f"Found {len(missing_txt)} .wav files without matching .txt files:")
        for wav_path, stem in missing_txt[:10]:
            logger.warning(f"  - {wav_path}")
        if len(missing_txt) > 10:
            logger.warning(f"  ... and {len(missing_txt) - 10} more")

    if missing_wav:
        logger.warning(f"Found {len(missing_wav)} .txt files without matching .wav files:")
        for txt_path, stem in missing_wav[:10]:
            logger.warning(f"  - {txt_path}")
        if len(missing_wav) > 10:
            logger.warning(f"  ... and {len(missing_wav) - 10} more")

    if empty_transcripts:
        logger.warning(f"Found {len(empty_transcripts)} pairs with empty transcripts:")
        for wav_path, txt_path in empty_transcripts[:10]:
            logger.warning(f"  - {wav_path}")
        if len(empty_transcripts) > 10:
            logger.warning(f"  ... and {len(empty_transcripts) - 10} more")


def build_dataset(valid_pairs: List[Tuple[Path, Path, str]]) -> Dataset:
    """
    Build a Hugging Face Dataset from valid audio-text pairs.

    Args:
        valid_pairs: List of (wav_path, txt_path, text_content) tuples.

    Returns:
        Hugging Face Dataset with 'audio' and 'text' columns.
    """
    logger.info("Building dataset from valid pairs...")

    data = []
    for wav_path, txt_path, text_content in tqdm(valid_pairs, desc="Processing pairs"):
        # Extract speaker ID from parent directory name
        speaker_id = wav_path.parent.name

        data.append({
            "audio": str(wav_path),
            "text": text_content,
            "speaker_id": speaker_id,
        })

    logger.info(f"Created dataset with {len(data)} samples")

    # Create Dataset
    dataset = Dataset.from_list(data)

    # Cast audio column to Audio type
    dataset = dataset.cast_column("audio", Audio())

    return dataset


def save_dataset_parquet(dataset: Dataset, output_dir: Path) -> Path:
    """
    Save dataset to Parquet format.

    Args:
        dataset: Hugging Face Dataset.
        output_dir: Output directory path.

    Returns:
        Path to saved parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "train.parquet"

    logger.info(f"Saving dataset to Parquet: {parquet_path}")
    dataset.to_parquet(parquet_path)

    logger.info(f"Dataset saved to {parquet_path}")
    return parquet_path


def generate_dataset_card(
    repo_name: str,
    validation_report: Dict[str, Any],
    license: str = "mit",
    description: str = None,
) -> str:
    """
    Generate a README.md dataset card.

    Args:
        repo_name: Repository name.
        validation_report: Validation report dict.
        license: Dataset license.
        description: Optional description.

    Returns:
        README content as string.
    """
    if description is None:
        description = f"This dataset contains paired audio and text samples for TTS training."

    readme = f"""---
license: {license}
---

# {repo_name}

{description}

## Dataset Overview

- **Total samples:** {validation_report['valid_pairs']}
- **Total .wav files scanned:** {validation_report['total_wav_files']}
- **Total .txt files scanned:** {validation_report['total_txt_files']}
- **Valid audio-text pairs:** {validation_report['valid_pairs']}
- **Skipped (missing transcript):** {validation_report['missing_txt']}
- **Skipped (missing audio):** {validation_report['missing_wav']}
- **Skipped (empty transcript):** {validation_report['empty_transcripts']}

## Dataset Structure

This dataset follows the same structure as the MrDragonFox/Elise dataset with:

- `audio`: Audio samples (WAV format)
- `text`: Transcript text
- `speaker_id`: Speaker identifier extracted from directory name

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_name}", split="train")

for sample in dataset:
    audio = sample["audio"]
    text = sample["text"]
    speaker_id = sample["speaker_id"]
    # Use for TTS training...
```

## License

{license.upper()}
"""
    return readme


def upload_to_hub(
    dataset: Dataset,
    repo_name: str,
    validation_report: Dict[str, Any],
    output_dir: Path,
    license: str = "mit",
    description: str = None,
    token: str = None,
) -> None:
    """
    Upload dataset to Hugging Face Hub.

    Args:
        dataset: Hugging Face Dataset.
        repo_name: Repository name (e.g., "username/dataset-name").
        validation_report: Validation report dict.
        output_dir: Output directory.
        license: Dataset license.
        description: Optional description.
        token: Hugging Face token (uses cached token if None).
    """
    logger.info(f"Uploading dataset to Hugging Face Hub: {repo_name}")

    # Create repository if it doesn't exist
    api = HfApi(token=token)
    try:
        repo_id = api.repo_id_from_name(repo_name)
        if not api.repo_exists(repo_id):
            logger.info(f"Creating repository: {repo_name}")
            create_repo(repo_name, repo_type="dataset", token=token)
    except Exception as e:
        logger.info(f"Repository may already exist or error checking: {e}")

    # Generate and save README
    readme_content = generate_dataset_card(repo_name, validation_report, license, description)
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Create DatasetDict
    dataset_dict = DatasetDict({"train": dataset})

    # Push to hub
    logger.info("Pushing to Hub...")
    dataset_dict.push_to_hub(repo_name, token=token)
    logger.info(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a Hugging Face TTS dataset from paired audio and text files."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source directory containing aligned audio/text pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./tts_dataset_output"),
        help="Output directory for the dataset (default: ./tts_dataset_output).",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default=None,
        help="Hugging Face Hub repository name (e.g., 'username/dataset-name'). "
             "If not provided, only local Parquet files will be created.",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="mit",
        help="Dataset license (default: mit).",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Dataset description for README (auto-generated if not provided).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (uses cached token if not provided).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading to Hugging Face Hub (only create local files).",
    )

    args = parser.parse_args()

    # Scan and validate pairs
    valid_pairs, issues, validation_report = scan_for_pairs(args.source)

    # Log validation issues
    log_validation_issues(issues)

    logger.info("Validation Report:")
    for key, value in validation_report.items():
        logger.info(f"  {key}: {value}")

    if not valid_pairs:
        logger.error("No valid audio-text pairs found. Exiting.")
        return

    # Build dataset
    dataset = build_dataset(valid_pairs)

    # Save to Parquet
    save_dataset_parquet(dataset, args.output)

    # Save validation report as JSON
    report_path = args.output / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    logger.info(f"Validation report saved to: {report_path}")

    # Upload to Hub if requested
    if args.skip_upload:
        logger.info("Skipping upload to Hugging Face Hub (--skip-upload flag set).")
    elif args.repo_name is None:
        logger.info("No repository name provided. Skipping upload to Hugging Face Hub.")
    else:
        upload_to_hub(
            dataset=dataset,
            repo_name=args.repo_name,
            validation_report=validation_report,
            output_dir=args.output,
            license=args.license,
            description=args.description,
            token=args.token,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
