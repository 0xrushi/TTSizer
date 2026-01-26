"""
Script: inference.py

Purpose:
    To generate speech (TTS) using a finetuned Spark-TTS model.

Description:
    This script loads the base Spark-TTS model and locally trained LoRA adapters.
    It then takes a text input and generates an audio file using the finetuned voice.
"""

import sys
import torch
import numpy as np
import re
import soundfile as sf
from unsloth import FastModel
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure Spark-TTS submodule is in path
sys.path.append('Spark-TTS')
from sparktts.models.audio_tokenizer import BiCodecTokenizer

# --- Configuration ---
# Set LORA_PATH to "tts_lora" for the final model
# or "outputs_tts/checkpoint-XXX" for intermediate steps.
# LORA_PATH = "tts_lora"
LORA_PATH =  "outputs_tts/checkpoint-1200"
BASE_MODEL_NAME = "Spark-TTS-0.5B/LLM"
MAX_SEQ_LENGTH = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILENAME = "generated.wav"

# Example input text
INPUT_TEXT = "to be honest this is a whole workflow, but if executed well the payoff is insane." 

def generate_speech(text: str, model, tokenizer, audio_tokenizer, device):
    """Generates audio from text using the finetuned model."""
    
    print(f"Generating speech for: '{text}'")
    
    # 1. Prepare Prompt
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>"
    ])

    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    # 2. Generate Tokens
    print("Generating token sequence...")
    # Enable native 2x faster inference
    FastModel.for_inference(model) 
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048, 
            do_sample=True,
            temperature=0.15,
            top_k=150,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    print("Token sequence generated.")

    # 3. Process Output
    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]

    # 4. Extract Audio Tokens
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)

    if not semantic_matches:
        print("Error: No semantic tokens found in output.")
        return None

    pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
    
    # Determine expected global token count from model config
    # The speaker encoder projects (latent_dim * token_num) -> out_dim
    # We can inspect the model to find token_num
    try:
        num_global_tokens = audio_tokenizer.model.speaker_encoder.perceiver_sampler.num_latents
    except AttributeError:
        # Fallback to 32 if structure is different, though Spark-TTS defaults to 32
        num_global_tokens = 32

    if not global_matches:
        print(f"Warning: No global tokens found. Using default zeros ({num_global_tokens} tokens).")
        # Create (1, num_global_tokens)
        pred_global_ids = torch.zeros((1, num_global_tokens), dtype=torch.long).to(device)
    else:
        # Ensure we match the expected length. If generated length differs, pad or truncate?
        # Usually exact match is expected. If model generated fewer/more, we might have issues.
        # But usually it generates exactly one block.
        tokens = [int(token) for token in global_matches]
        if len(tokens) != num_global_tokens:
             print(f"Warning: Generated {len(tokens)} global tokens, expected {num_global_tokens}. Adjusting.")
             if len(tokens) < num_global_tokens:
                 tokens += [0] * (num_global_tokens - len(tokens))
             else:
                 tokens = tokens[:num_global_tokens]
        
        pred_global_ids = torch.tensor(tokens).long().unsqueeze(0).to(device) # (1, N)

    # BiCodec.detokenize expects global_tokens as (1, N) if we use audio_tokenizer.detokenize?
    # Wait, audio_tokenizer.detokenize does:
    # global_tokens = global_tokens.unsqueeze(1) -> (1, 1, N)
    # Then speaker_encoder.detokenize(global_tokens) -> indices=(1, 1, N)
    # indices.transpose(1, 2) -> (1, N, 1) if indices is 3D?
    
    # Actually, in BiCodecTokenizer.detokenize:
    # wav_rec = self.model.detokenize(semantic_tokens, global_tokens.unsqueeze(1))
    
    # So we should pass (1, N).
    
    # However, in my previous code I did:
    # pred_global_ids = ... .unsqueeze(0).unsqueeze(0) -> (1, 1, N)
    # And then .squeeze(0) -> (1, N)
    # So passing (1, N) to audio_tokenizer.detokenize is correct.

    print(f"Found {pred_semantic_ids.shape[1]} semantic tokens.")
    print(f"Using {pred_global_ids.shape[1]} global tokens.")

    # 5. Detokenize to Audio
    print("Detokenizing audio tokens...")
    audio_tokenizer.device = device
    audio_tokenizer.model.to(device)
    
    # Ensure inputs are on device
    pred_global_ids = pred_global_ids.to(device)
    pred_semantic_ids = pred_semantic_ids.to(device)

    wav_np = audio_tokenizer.detokenize(
        pred_global_ids, 
        pred_semantic_ids
    )
    print("Detokenization complete.")
    return wav_np

def main():
    if not Path(LORA_PATH).exists():
        print(f"Error: LoRA adapter path '{LORA_PATH}' not found. Did you run finetune.py?")
        return

    print(f"Loading model from {BASE_MODEL_NAME} with adapters from {LORA_PATH}...")
    
    # Load Model with LoRA
    model, tokenizer = FastModel.from_pretrained(
        model_name=LORA_PATH, # Load the adapter config directly
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float32,
        load_in_4bit=False,
    )
    
    # Initialize Audio Tokenizer (Bicodec)
    # We download/load it separately as it's part of the Spark-TTS repo structure
    audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", str(DEVICE))

    # Generate
    waveform = generate_speech(INPUT_TEXT, model, tokenizer, audio_tokenizer, DEVICE)

    # Save
    if waveform is not None and waveform.size > 0:
        sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
        sf.write(OUTPUT_FILENAME, waveform, sample_rate)
        print(f"\nSuccess! Audio saved to: {OUTPUT_FILENAME}")
    else:
        print("\nGeneration failed.")

if __name__ == "__main__":
    main()
