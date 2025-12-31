import yaml
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def audit_dataset():
    cfg_path = "configs/config.yaml"
    if not Path(cfg_path).exists():
        print(f"Error: Config not found at {cfg_path}")
        return

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    series_name = cfg["project_setup"]["series_name"]
    # check both 'aligned_clips' and 'final'
    base_dir = Path(cfg["project_setup"]["output_base_dir"]) / series_name
    
    dirs_to_check = [
        base_dir / cfg["ctc_aligner"]["output_folder"],
        base_dir / cfg["asr_processor"]["output_folder"]
    ]
    
    print(f"Auditing dataset in: {base_dir}")
    
    suspicious_files = []
    
    for root_dir in dirs_to_check:
        if not root_dir.exists(): continue
        
        wav_files = sorted(list(root_dir.rglob("*.wav")))
        if not wav_files: continue
        
        print(f"Checking {len(wav_files)} files in {root_dir.name}...")
        
        for wav_path in tqdm(wav_files, unit="file"):
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists(): continue
            
            try:
                # 1. Check Audio Duration
                info = sf.info(str(wav_path))
                dur = info.duration
                
                # 2. Check Text Length
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                char_count = len(text)
                word_count = len(text.split())
                
                if dur == 0:
                    suspicious_files.append((wav_path, "Zero duration", dur, text))
                    continue
                
                # 3. Calculate Rate (Chars per Second)
                cps = char_count / dur
                
                # Thresholds for suspicion
                # Normal speech is ~15-20 CPS max. >30 is likely an error (text too long for audio).
                # < 1 CPS is likely silence or error (audio too long for text).
                if cps > 35: 
                    suspicious_files.append((wav_path, f"Too Fast ({cps:.1f} cps)", dur, text))
                elif cps < 0.5 and dur > 2.0:
                    suspicious_files.append((wav_path, f"Too Slow ({cps:.1f} cps)", dur, text))
                elif dur < 0.3:
                     suspicious_files.append((wav_path, f"Too Short ({dur:.2f}s)", dur, text))

            except Exception as e:
                print(f"Error checking {wav_path.name}: {e}")

    print("\n" + "="*50)
    print(f"Found {len(suspicious_files)} suspicious files:")
    print("="*50)
    
    # Print top 20
    for path, reason, dur, text in suspicious_files[:20]:
        print(f"[{reason}] {path.name}")
        print(f"  Dur: {dur:.2f}s | Text: {text[:50]}...")
        print("-" * 30)
    
    if len(suspicious_files) > 20:
        print(f"... and {len(suspicious_files) - 20} more.")

if __name__ == "__main__":
    audit_dataset()
