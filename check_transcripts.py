"""
Script: check_transcripts.py

Purpose:
    To audit a directory of audio clips and verify the status of their corresponding transcription files.

Description:
    This script iterates through all .wav files in a specified directory and checks if a corresponding
    .txt file exists and if it contains any text. It counts valid transcripts, empty/whitespace-only
    transcripts, and missing transcript files, outputting a summary report. This helps identify data
    integrity issues after processing or cleaning stages.
"""

import os
from pathlib import Path

def check_transcripts(target_dir: str):
    root = Path(target_dir)
    if not root.exists():
        print(f"Error: Directory not found: {root}")
        return

    empty_count = 0
    missing_count = 0
    valid_count = 0
    total_wavs = 0

    wav_files = list(root.glob("*.wav"))
    total_wavs = len(wav_files)

    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            missing_count += 1
            continue
        
        content = txt_path.read_text(encoding="utf-8").strip()
        if not content:
            empty_count += 1
        else:
            valid_count += 1

    print(f"\n--- Transcript Check Summary ---")
    print(f"Directory: {root}")
    print(f"Total .wav files found: {total_wavs}")
    print(f"  - Valid transcripts: {valid_count}")
    print(f"  - Empty/Whitespace only: {empty_count}")
    print(f"  - Missing .txt files: {missing_count}")
    print("-" * 32 + "\n")

if __name__ == "__main__":
    target = "/mnt/sdc3/Documents/speech-dataset-generator/data/output/Family Guy.S01-S21.1080p.H265/final/cleaned/SPEAKER_02"
    check_transcripts(target)
