import json
import yaml
from pathlib import Path
from collections import Counter

# ==========================================
# USER CONFIGURATION: SPEAKER MAPPING
# ==========================================
# Map existing incorrect labels to the correct ones.
# Format: "Current Label": "Correct Label"
SPEAKER_MAPPING = {
    # "Speaker 1": "Peter",
    # "Speaker 2": "Lois",
    # "Peter": "Homer",   # Example: If Peter was actually Homer
}
# ==========================================

def fix_speaker_labels():
    # 1. Load project config
    cfg_path = "configs/config.yaml"
    if not Path(cfg_path).exists():
        print(f"Error: Config not found at {cfg_path}")
        return

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    series_name = cfg["project_setup"]["series_name"]
    base_dir = Path(cfg["project_setup"]["output_base_dir"]) / series_name
    json_dir = base_dir / cfg["llm_diarizer"]["output_folder"]

    if not json_dir.exists():
        print(f"Error: JSON directory not found at {json_dir}")
        return

    json_files = sorted([f for f in json_dir.glob("*.json") if not f.name.endswith(".uploaded.json")])
    print(f"Found {len(json_files)} transcription files.")

    total_changes = 0
    
    for json_path in json_files:
        changed_file = False
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            
            # Count speakers before
            speakers_before = Counter([s.get("speaker") for s in segments])
            
            new_segments = []
            for seg in segments:
                old_speaker = seg.get("speaker")
                
                # Apply mapping
                if old_speaker in SPEAKER_MAPPING:
                    seg["speaker"] = SPEAKER_MAPPING[old_speaker]
                    changed_file = True
                    total_changes += 1
                
                new_segments.append(seg)
            
            if changed_file:
                # Save backup
                backup_path = json_path.with_suffix(".json.bak")
                if not backup_path.exists():
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(segments, f, indent=4, ensure_ascii=False)
                
                # Overwrite original
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(new_segments, f, indent=4, ensure_ascii=False)
                
                print(f"Updated {json_path.name}")
                print(f"  - Changes: {total_changes}")
                
        except Exception as e:
            print(f"Error processing {json_path.name}: {e}")

    print("\n" + "="*50)
    print("DONE!")
    if total_changes > 0:
        print(f"Total segments updated: {total_changes}")
        print("IMPORTANT: Now you must re-run 'ctc_aligner' (Stage 5) to generate correct audio clips.")
        print("Run: python -m ttsizer.main")
    else:
        print("No changes made. Did you edit the SPEAKER_MAPPING dictionary in this script?")

if __name__ == "__main__":
    fix_speaker_labels()
