import yaml
import csv
from pathlib import Path
from tqdm import tqdm

def generate_metadata():
    # 1. Load config to get paths
    cfg_path = "configs/config.yaml"
    if not Path(cfg_path).exists():
        print(f"Error: Config not found at {cfg_path}")
        return

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    series_name = cfg["project_setup"]["series_name"]
    output_base = Path(cfg["project_setup"]["output_base_dir"])
    series_dir = output_base / series_name
    
    # ASR processor output folder is 'final' in config
    asr_folder = cfg["asr_processor"]["output_folder"]
    final_dir = series_dir / asr_folder
    
    if not final_dir.exists():
        print(f"Error: Final directory not found at {final_dir}")
        return

    metadata = []
    
    # 2. Scan each speaker directory
    # Structure: final/{Speaker_Name}/definite/*.wav
    speaker_dirs = sorted([d for d in final_dir.iterdir() if d.is_dir()])
    
    print(f"Scanning {len(speaker_dirs)} speaker directories...")
    
    for spkr_dir in speaker_dirs:
        speaker_name = spkr_dir.name.replace("_", " ")
        definite_dir = spkr_dir / "definite"
        
        if not definite_dir.exists():
            continue
            
        wav_files = sorted(list(definite_dir.glob("*.wav")))
        if not wav_files:
            continue

        for wav_path in tqdm(wav_files, desc=f"Processing {speaker_name}", unit="file"):
            txt_path = wav_path.with_suffix(".txt")
            
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        transcript = f.read().strip().replace("\n", " ")
                    
                    if not transcript:
                        continue

                    # We store the relative path from the series directory
                    rel_path = wav_path.relative_to(series_dir)
                    metadata.append([str(rel_path), speaker_name, transcript])
                except Exception as e:
                    print(f"Error reading {txt_path}: {e}")

    if not metadata:
        print("No metadata found to write.")
        return

    # 3. Write to CSV
    output_csv = series_dir / "metadata.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        # Standard LJSpeech uses |
        writer = csv.writer(f, delimiter="|")
        writer.writerows(metadata)

    print(f"\n✅ Success! Generated {len(metadata)} rows in:")
    print(f"   {output_csv.resolve()}")

if __name__ == "__main__":
    generate_metadata()
