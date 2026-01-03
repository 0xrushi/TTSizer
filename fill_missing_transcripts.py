"""
Script: fill_missing_transcripts.py

Purpose:
    To identify audio clips with empty or missing transcription files (.txt) in a specific directory
    and fill them by generating new transcriptions using the Gemini Flash API.

Description:
    This script scans a given directory for .wav files. For each audio file, it checks the corresponding
    .txt file. If the .txt file is empty or contains only whitespace, the script uses the Google Gemini
    Flash model (via GeminiASRBackend) to transcribe the audio and overwrites the .txt file with the
    result. This is useful for fixing gaps in the dataset cleaning process.
"""

import sys
import os
from pathlib import Path
from tqdm.auto import tqdm

# Ensure project root is in path to import ttsizer modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ttsizer.asr_backends.gemini_asr import GeminiASRBackend
from ttsizer.utils.logger import get_logger

logger = get_logger("fill_missing_transcripts")

def fill_missing_transcripts(target_dir: str, api_key: str):
    root = Path(target_dir)
    if not root.exists():
        logger.error(f"Directory not found: {root}")
        return

    logger.info(f"Scanning for missing transcripts in: {root}")
    
    wav_files = sorted(list(root.glob("*.wav")))
    to_process = []

    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            to_process.append(wav_path)
            continue
        
        content = txt_path.read_text(encoding="utf-8").strip()
        if not content:
            to_process.append(wav_path)

    if not to_process:
        logger.info("No missing or empty transcripts found!")
        return

    logger.info(f"Found {len(to_process)} files to transcribe.")

    # Initialize backend
    # Note: Using gemini-2.0-flash-lite as per project convention or similar fast model
    model_name = "gemini-2.0-flash-lite"
    backend = GeminiASRBackend(model_name=model_name, api_key=api_key)

    # Process in batches or individually. The backend supports batching logic internally 
    # but let's just do simple batching here to match backend capability.
    batch_size = 10 
    
    for i in tqdm(range(0, len(to_process), batch_size), desc="Transcribing"):
        batch_paths = [str(p) for p in to_process[i:i+batch_size]]
        wav_batch_objs = to_process[i:i+batch_size]
        
        try:
            results = backend.transcribe_batch(batch_paths)
            
            for wav_path, result in zip(wav_batch_objs, results):
                if result and result.text:
                    txt_path = wav_path.with_suffix(".txt")
                    txt_path.write_text(result.text.strip(), encoding="utf-8")
                else:
                    logger.warning(f"Failed to get transcript for {wav_path.name}")
                    
        except Exception as e:
            logger.error(f"Error processing batch starting at {i}: {e}")

    logger.info("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fill empty transcripts using Gemini.")
    parser.add_argument("target_dir", help="Directory containing .wav and .txt files")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    fill_missing_transcripts(args.target_dir, api_key)
