import sys
import torch
import numpy as np
import soundfile as sf
import re
import os
from pathlib import Path
from rich.text import Text

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Ensure Spark-TTS submodule is in path
sys.path.append('Spark-TTS')

from unsloth import FastModel
from sparktts.models.audio_tokenizer import BiCodecTokenizer

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, Static, Label
from textual.containers import Container
from textual.worker import Worker, WorkerState

def trim_end_silence(audio_array, threshold=0.01, chunk_size=1000):
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
    
    # Calculate cut point (Total length - trim_index)
    cut_point = len(audio_array) - trim_index
    
    # Add a tiny bit of padding (e.g., 0.1s) so it doesn't sound abrupt
    padding_samples = 500 # Reduced padding
    final_cut = min(len(audio_array), cut_point + padding_samples)
    
    return audio_array[:final_cut]

class TTSTUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    .box {
        height: 1fr;
        border: solid green;
    }
    #input-container {
        height: auto;
        padding: 1;
        border-top: solid white;
    }
    #status-bar {
        height: 1;
        background: $primary;
        color: white;
        dock: bottom;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "clear_log", "Clear Log"),
    ]

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.audio_tokenizer = None
        
        # Default Params
        self.lora_path = "tts_lora"
        self.temperature = 0.7
        self.top_k = 50
        self.top_p = 1.0
        self.repetition_penalty = 1.1
        self.max_new_tokens = 2048
        self.trim_threshold = 0.03 # Default threshold
        
        self.output_dir = "output_tui"
        os.makedirs(self.output_dir, exist_ok=True)
        self.gen_count = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            RichLog(id="log_view", markup=True),
            classes="box"
        )
        yield Container(
            Label("Type text or /cmd (e.g. /temp 0.5):"),
            Input(placeholder="Enter text to generate speech...", id="input_box"),
            id="input-container"
        )
        yield Static(id="status-bar", content="Ready")
        yield Footer()

    async def on_mount(self) -> None:
        self.log_msg("Welcome to TTS TUI!", style="bold green")
        self.log_msg(f"Device: {self.device}")
        self.log_msg("Loading model... please wait.", style="yellow")

        self.run_worker(self.load_model_worker, exclusive=True, thread=True)

    def load_model_worker(self):
        try:
            path_to_load = self.lora_path
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
                 self.log_msg(f"Error: Model path '{path_to_load}' not found.", style="bold red")
                 return

            self.log_msg(f"Loading from: {path_to_load}")
            
            model, tokenizer = FastModel.from_pretrained(
                model_name=path_to_load,
                max_seq_length=2048,
                dtype=torch.float32,
                load_in_4bit=False,
            )
            
            audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", str(self.device))
            
            # Enable inference mode
            FastModel.for_inference(model)
            
            self.model = model
            self.tokenizer = tokenizer
            self.audio_tokenizer = audio_tokenizer
            self.lora_path = path_to_load # update if we fell back
            
            self.log_msg("Model loaded successfully!", style="bold green")
            self.update_status(f"Loaded: {path_to_load} | T={self.temperature}")
            
        except Exception as e:
            self.log_msg(f"Error loading model: {str(e)}", style="bold red")
            import traceback
            self.log_msg(traceback.format_exc())

    def action_clear_log(self):
        self.query_one("#log_view", RichLog).clear()

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        val = message.value.strip()
        if not val:
            return
        
        self.query_one("#input_box", Input).value = ""
        
        if val.startswith("/"):
            self.handle_command(val)
        else:
            if self.model is None:
                self.log_msg("Model not loaded yet. Please wait.", style="red")
                return
            self.log_msg(f"Generating: '{val}'", style="cyan")
            self.run_worker(lambda: self.generate_speech_worker(val), exclusive=True, thread=True)

    def handle_command(self, cmd_str: str):
        parts = cmd_str.split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd == "/temp" or cmd == "/temperature":
            if args:
                try:
                    self.temperature = float(args[0])
                    self.log_msg(f"Temperature set to {self.temperature}", style="green")
                except ValueError:
                    self.log_msg("Invalid float.", style="red")
            else:
                self.log_msg(f"Current Temperature: {self.temperature}")
                
        elif cmd == "/top_k":
            if args:
                try:
                    self.top_k = int(args[0])
                    self.log_msg(f"Top_k set to {self.top_k}", style="green")
                except ValueError:
                    self.log_msg("Invalid int.", style="red")
            else:
                 self.log_msg(f"Current top_k: {self.top_k}")
        
        elif cmd == "/top_p":
            if args:
                try:
                    self.top_p = float(args[0])
                    self.log_msg(f"Top_p set to {self.top_p}", style="green")
                except ValueError:
                    self.log_msg("Invalid float.", style="red")
            else:
                 self.log_msg(f"Current top_p: {self.top_p}")

        elif cmd == "/trim":
            if args:
                try:
                    self.trim_threshold = float(args[0])
                    self.log_msg(f"Trim threshold set to {self.trim_threshold}", style="green")
                except ValueError:
                    self.log_msg("Invalid float.", style="red")
            else:
                 self.log_msg(f"Current Trim Threshold: {self.trim_threshold}")

        elif cmd == "/help":
            self.log_msg("Commands:", style="yellow")
            self.log_msg("  /temp <float>   : Set temperature")
            self.log_msg("  /top_k <int>    : Set top_k")
            self.log_msg("  /top_p <float>  : Set top_p")
            self.log_msg("  /trim <float>   : Set silence trim threshold (default 0.03)")
            self.log_msg("  /model <path>   : Load different model/checkpoint (restarts app logic)")
            
        elif cmd == "/model":
            if args:
                new_path = args[0]
                if os.path.exists(new_path):
                    self.lora_path = new_path
                    self.log_msg(f"Switching to model: {new_path} ...")
                    self.run_worker(self.load_model_worker, exclusive=True)
                else:
                    self.log_msg(f"Path not found: {new_path}", style="red")
        else:
            self.log_msg(f"Unknown command: {cmd}", style="red")
            
        self.update_status(f"Loaded: {self.lora_path} | T={self.temperature}")

    def generate_speech_worker(self, text: str):
        try:
            device = self.device
            tokenizer = self.tokenizer
            model = self.model
            audio_tokenizer = self.audio_tokenizer
            
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
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
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
                self.app.call_from_thread(self.log_msg, "Error: No semantic tokens generated.", style="red")
                return

            pred_semantic_ids = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
            
            # Handle global tokens
            num_global_tokens = 32 # Default
            try:
                if hasattr(audio_tokenizer.model.speaker_encoder, 'perceiver_sampler'):
                     num_global_tokens = audio_tokenizer.model.speaker_encoder.perceiver_sampler.num_latents
            except AttributeError:
                pass # Use default 32

            if not global_matches:
                self.app.call_from_thread(self.log_msg, "Warning: No global tokens. Using zeros.", style="yellow")
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
            self.app.call_from_thread(self.log_msg, f"Trimming noise (thresh={self.trim_threshold})...", style="dim")
            wav_np = trim_end_silence(wav_np, threshold=self.trim_threshold)
            
            # 4. Save
            self.gen_count += 1
            filename = f"gen_{self.gen_count:03d}.wav"
            filepath = os.path.join(self.output_dir, filename)
            
            sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
            sf.write(filepath, wav_np, sample_rate)
            
            self.app.call_from_thread(self.log_msg, f"Saved: {filepath}", style="bold green")
            
        except Exception as e:
            self.app.call_from_thread(self.log_msg, f"Generation Error: {str(e)}", style="bold red")
            import traceback
            print(traceback.format_exc())

    def log_msg(self, msg: str, style: str = ""):
        log = self.query_one("#log_view", RichLog)
        if style:
            log.write(Text(msg, style=style))
        else:
            log.write(msg)

    def update_status(self, text: str):
        self.query_one("#status-bar", Static).update(text)

if __name__ == "__main__":
    app = TTSTUI()
    app.run()
