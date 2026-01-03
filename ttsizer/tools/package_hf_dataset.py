"""
Script: package_hf_dataset.py

Purpose:
    To package a directory of audio clips and text transcripts into a Hugging Face Dataset compatible format (AudioFolder).

Description:
    This script traverses a source directory containing speaker subdirectories (or a flat structure) with paired
    .wav and .txt files. It copies the audio files to a new destination directory and generates a 'metadata.csv' file.
    The resulting structure allows the dataset to be loaded easily using the Hugging Face `datasets` library:
    
    >>> from datasets import load_dataset
    >>> dataset = load_dataset("audiofolder", data_dir="/path/to/packaged_dataset")
    
    It also prepares the dataset for uploading to the Hugging Face Hub.
"""

import argparse
import csv
import shutil
import os
from pathlib import Path
from tqdm.auto import tqdm
from ttsizer.utils.logger import get_logger

logger = get_logger("package_hf_dataset")

def package_dataset(source_dir: Path, output_dir: Path, split: str = "train"):
    """
    Packages audio and text files into an AudioFolder structure. 
    
    Args:
        source_dir: Path to the directory containing processed audio/text pairs.
        output_dir: Path where the new dataset will be created.
        split: The dataset split name (e.g., 'train', 'test').
    """
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return

    # Create output directory structure: output_dir/data
    # AudioFolder expects metadata.csv at root and audio files usually in 'data' or root.
    # We will put audio in 'data/{split}' to be organized. 
    
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Scanning source: {source_dir}")
    
    wav_files = sorted(list(source_dir.rglob("*.wav")))
    
    if not wav_files:
        logger.warning("No .wav files found in source directory.")
        return

    logger.info(f"Found {len(wav_files)} files. Starting packaging...")
    
    metadata_rows = []
    
    # Counter for unique filenames if needed, but we'll try to preserve relative structure or unique names
    
    for wav_path in tqdm(wav_files, desc="Packaging"):
        txt_path = wav_path.with_suffix(".txt")
        
        # Determine transcription
        transcription = ""
        if txt_path.exists():
            try:
                transcription = txt_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"Could not read text for {wav_path.name}: {e}")
        
        if not transcription:
            logger.debug(f"Skipping empty transcription for {wav_path.name}")
            # Optionally continue or keep empty? Usually for training we want text.
            # Let's keep it but warn, or skip. User cleaned it previously hopefully.
            # If "fill_missing_transcripts" was run, it should be fine.
            # We will include it to match file count, but empty text might be filtered by trainers.
        
        # Define new filename to avoid collisions if flattening
        # Strategy: Use parent folder name (Speaker ID) + filename
        speaker_id = wav_path.parent.name
        
        # Check if parent is the source dir (flat structure)
        if wav_path.parent == source_dir:
            speaker_id = "unknown"
            
        # Create unique filename: SPEAKER_ID_OriginalName.wav
        # If original name already starts with SPEAKER_ID, don't repeat
        safe_name = wav_path.name
        if not safe_name.startswith(speaker_id) and speaker_id != "unknown":
            safe_name = f"{speaker_id}_{safe_name}"
            
        # Copy audio
        dest_wav_path = data_dir / safe_name
        shutil.copy2(wav_path, dest_wav_path)
        
        # Record metadata
        # AudioFolder expects 'file_name' relative to the metadata.csv file
        metadata_rows.append({
            "file_name": f"data/{safe_name}",
            "transcription": transcription,
            "speaker_id": speaker_id
        })

    # Write metadata.csv
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription", "speaker_id"])
        writer.writeheader()
        writer.writerows(metadata_rows)
        
    logger.info("Packaging complete!")
    logger.info(f"Dataset created at: {output_dir}")
    logger.info(f"Total samples: {len(metadata_rows)}")
    logger.info("\nTo load this dataset using Hugging Face datasets:")
    logger.info(f'    from datasets import load_dataset')
    logger.info(f'    dataset = load_dataset("audiofolder", data_dir="{output_dir.resolve()}")')
    logger.info("\nTo push to Hub (e.g., 'MrDragonFox/Elise'):")
    logger.info(f'    dataset.push_to_hub("MrDragonFox/Elise")')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package cleaned audio/text pairs into a HF AudioFolder dataset.")
    parser.add_argument("--source", type=Path, required=True, help="Input directory containing cleaned segments (e.g. final/cleaned)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for the packaged dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    
    args = parser.parse_args()
    
    package_dataset(args.source, args.output, args.split)
