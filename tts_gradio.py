import sys
import torch
import numpy as np
import re
import os
from pathlib import Path
import gradio as gr

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Ensure Spark-TTS submodule is in path
sys.path.append('Spark-TTS')

from unsloth import FastModel
from sparktts.models.audio_tokenizer import BiCodecTokenizer

# Global variables for model/tokenizer
model = None
tokenizer = None
audio_tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LORA_PATH = "tts_lora"

def load_model():
    global model, tokenizer, audio_tokenizer, LORA_PATH
    
    path_to_load = LORA_PATH
    if not os.path.exists(path_to_load):
        # Fallback to outputs_tts if default doesn't exist
        if os.path.exists("outputs_tts"):
                # Find latest checkpoint
                checkpoints = [d for d in os.listdir("outputs_tts") if d.startswith("checkpoint-")]
                if checkpoints:
                    # sort by number
                    latest = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
                    path_to_load = os.path.join("outputs_tts", latest)
    
    if not os.path.exists(path_to_load):
            return f"Error: Model path '{path_to_load}' not found."

    print(f"Loading from: {path_to_load}")
    
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=path_to_load,
            max_seq_length=2048,
            dtype=torch.float32,
            load_in_4bit=False,
        )
        
        audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", str(device))
        
        # Enable inference mode
        FastModel.for_inference(model)
        
        return f"Successfully loaded model from {path_to_load}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def trim_end_silence(audio_array, threshold=0.01, padding_samples=500):
    """
    Trims silence/noise from the end of the numpy audio array.
    """
    # If audio is stereo (2, N), average it to mono for detection
    if len(audio_array.shape) > 1:
        energy_check = np.mean(audio_array, axis=0)
    else:
        energy_check = audio_array

    # Flip array to search from the end
    reversed_audio = energy_check[::-1]
    
    # Find the index of the first sample (from the end) that is above threshold
    is_sound = np.abs(reversed_audio) > threshold
    
    if not np.any(is_sound):
        # If the whole clip is silence, return original (or empty)
        return audio_array

    # Get index of first sound from the end
    trim_index = np.argmax(is_sound)
    
    # Calculate cut point
    cut_point = len(audio_array) - trim_index
    
    final_cut = min(len(audio_array), cut_point + padding_samples)
    
    return audio_array[:final_cut]

def generate_speech(text, temperature, top_k, top_p, repetition_penalty, trim_threshold):
    global model, tokenizer, audio_tokenizer, device
    
    if model is None:
        load_msg = load_model()
        if "Error" in load_msg:
            raise gr.Error(load_msg)
            
    try:
        # 1. Prepare Prompt
        prompt = "".join([
            "<|task_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>"
        ])
        
        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
        
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=temperature,
                top_k=int(top_k),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
        
        # EOS Slicing
        if tokenizer.eos_token_id in generated_ids_trimmed[0]:
            eos_index = (generated_ids_trimmed[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
            generated_ids_trimmed = generated_ids_trimmed[:, :eos_index]

        predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
        
        # 2. Extract Tokens
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
        
        if not semantic_matches:
            raise gr.Error("No semantic tokens generated.")

        pred_semantic_ids = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
        
        # Handle global tokens
        num_global_tokens = 32 # Default
        try:
            if hasattr(audio_tokenizer.model.speaker_encoder, 'perceiver_sampler'):
                    num_global_tokens = audio_tokenizer.model.speaker_encoder.perceiver_sampler.num_latents
        except AttributeError:
            pass # Use default 32

        if not global_matches:
            print("Warning: No global tokens. Using zeros.")
            pred_global_ids = torch.zeros((1, num_global_tokens), dtype=torch.long).to(device)
        else:
            tokens = [int(t) for t in global_matches]
            if len(tokens) != num_global_tokens:
                # Resize/Pad
                if len(tokens) < num_global_tokens:
                    tokens += [0] * (num_global_tokens - len(tokens))
                else:
                    tokens = tokens[:num_global_tokens]
            pred_global_ids = torch.tensor(tokens).long().unsqueeze(0).to(device)

        # 3. Detokenize
        audio_tokenizer.device = device
        audio_tokenizer.model.to(device)
        pred_global_ids = pred_global_ids.to(device)
        pred_semantic_ids = pred_semantic_ids.to(device)

        wav_np = audio_tokenizer.detokenize(
            pred_global_ids,
            pred_semantic_ids
        )
        
        # Trim Silence
        wav_np = trim_end_silence(wav_np, threshold=trim_threshold, padding_samples=500)
        
        sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
        
        return (sample_rate, wav_np)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(str(e))

# --- Gradio UI ---

with gr.Blocks(title="TTS Generator") as demo:
    gr.Markdown("# 🎤 TTS Generator")
    gr.Markdown("Enter text below to generate speech.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Input Text", placeholder="Enter text to generate speech...", lines=3)
            
            with gr.Accordion("Advanced Settings", open=True):
                temp_slider = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.05, label="Temperature")
                top_k_slider = gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Top-K")
                top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Top-P")
                rep_pen_slider = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.05, label="Repetition Penalty")
                trim_thresh_slider = gr.Slider(minimum=0.0, maximum=0.1, value=0.03, step=0.005, label="Silence Trim Threshold")
            
            generate_btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio", type="numpy")
            
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, temp_slider, top_k_slider, top_p_slider, rep_pen_slider, trim_thresh_slider],
        outputs=audio_output
    )

if __name__ == "__main__":
    # Load model on startup
    print(load_model())
    demo.launch(server_name="0.0.0.0", share=True)
