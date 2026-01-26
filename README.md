# TTSizer 🎙️✨
### Transform Raw Audio/Video into Production-Ready TTS Datasets

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

**Watch the TTSizer Demo & See It In Action:**
<a href="https://youtu.be/POwMVTwsZDQ?si=rxNy7grLyROhdIEd" target="_blank">
  <img src="https://img.youtube.com/vi/POwMVTwsZDQ/maxresdefault.jpg" alt="TTSizer Demo Video" style="width:100%; max-width:960px; height:auto; display:block; margin-left:auto; margin-right:auto;">
</a>
<em>(The demo above showcases the <a href="https://huggingface.co/datasets/taresh18/AnimeVox" target="_blank">AnimeVox Character TTS Corpus</a>, a dataset created using TTSizer.)</em>

## 🎯 What It Does

TTSizer automates the tedious process of creating high-quality Text-To-Speech datasets from raw media. Input a video or audio file, and get back perfectly aligned audio-text pairs for each speaker.

## ✨ Key Features

🎯 **End-to-End Automation**: From raw media files to cleaned, TTS-ready datasets   
🗣️ **Advanced Multi-Speaker Diarization**: Handles complex audio with multiple speakers  
🤖 **State-of-the-Art Models** - MelBandRoformer, Gemini, CTC-Aligner, Wespeaker       
🧐 **Quality Control**: Automatic outlier detection and flagging  
⚙️ **Fully Configurable**: Control every aspect via `config.yaml` 

## 📊 Pipeline Flow

```mermaid
graph LR
    A[🎬 Raw Media] --> B[🎤 Extract Audio]
    B --> C[🔇 Vocal Separation]  
    C --> D[🔊 Normalize Volume]
    D --> E[✍️ Speaker Diarization]
    E --> F[⏱️ Forced Alignment]
    F --> G[🧐 Outlier Detection]
    G --> H[🚩 ASR Validation]
    H --> I[✅ TTS Dataset]
```

## 🏃 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/taresh18/TTSizer.git
cd TTSizer
pip install -r requirements.txt
```

### 2. Setup Models & API Key
- Download pre-trained models (see [Setup Guide](#setup--installation))
- Add `GEMINI_API_KEY` to `.env` file in the project root:
```bash
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### 3. Configure
Edit `configs/config.yaml`:
```yaml
project_setup:
  video_input_base_dir: "/path/to/your/videos"
  output_base_dir: "/path/to/output"
  target_speaker_labels: ["Speaker1", "Speaker2"]
```

### 4. Run TTSizer!
```bash
python -m ttsizer.main
```

## 🎤 Finetuning & Inference Scripts

In addition to the main dataset generation pipeline, TTSizer includes scripts for finetuning Spark-TTS models and generating speech using the trained models.

### Finetuning Script (`finetune.py`)

**Purpose**: Fine-tune the Spark-TTS (0.5B) model on a custom audio dataset.

**How to run**:
```bash
python finetune.py
```

**What it does**:
- Loads a local AudioFolder dataset (default: `tts_dataset_output`)
- Uses unsloth for optimized LoRA training
- Trains the model on speaker-specific data
- Saves LoRA adapters to `tts_lora/` directory
- Outputs intermediate checkpoints to `outputs_tts/`

**Configuration**:
Edit the following variables in the script:
- `dataset_path`: Path to your AudioFolder dataset with `transcription` field
- `max_seq_length`: Maximum sequence length (default: 2048)
- Training parameters: `max_steps`, `learning_rate`, `batch_size` in SFTConfig

### TTS TUI (`tts_tui.py`)

**Purpose**: A terminal-based user interface for generating speech using a fine-tuned model.

**How to run**:
```bash
python tts_tui.py
```

**What it does**:
- Loads a fine-tuned model from `tts_lora/` or `outputs_tts/checkpoint-*/`
- Provides an interactive command-line interface
- Supports real-time text-to-speech generation
- Includes advanced controls: temperature, top_k, top_p, repetition penalty, silence trim
- Saves generated audio to `output_tui/`

**Available commands** (type in the input box):
- `/temp <float>` - Set temperature (controls randomness)
- `/top_k <int>` - Set top-k sampling
- `/top_p <float>` - Set top-p (nucleus) sampling
- `/trim <float>` - Set silence trim threshold (default 0.03)
- `/model <path>` - Load a different model/checkpoint
- `/help` - Show all available commands

**Keyboard shortcuts**:
- `Ctrl+Q` - Quit the application
- `Ctrl+C` - Clear the log

### TTS Gradio Interface (`tts_gradio.py`)

**Purpose**: A web-based Gradio interface for generating speech using a fine-tuned model.

**How to run**:
```bash
python tts_gradio.py
```

**What it does**:
- Provides a user-friendly web UI for TTS generation
- Supports all the same features as the TUI
- Generates a shareable link for remote access
- Includes sliders for all generation parameters
- Displays generated audio directly in the browser

**Parameters available in the UI**:
- **Temperature**: Controls randomness (0.1 - 1.5)
- **Top-K**: Limits token choices to top K candidates (1 - 200)
- **Top-P**: Nucleus sampling threshold (0.0 - 1.0)
- **Repetition Penalty**: Penalizes repeated tokens (1.0 - 2.0)
- **Silence Trim Threshold**: Cuts trailing silence (0.0 - 0.1)

### Inference Script (`inference.py`)

**Purpose**: A simple command-line script for generating speech using a fine-tuned model.

**How to run**:
```bash
python inference.py
```

**What it does**:
- Loads a fine-tuned model from `tts_lora/` or a checkpoint path
- Generates speech from a pre-configured input text
- Saves the output audio to `generated.wav`

**Configuration**:
Edit the following variables in the script:
- `LORA_PATH`: Path to LoRA adapters (default: `outputs_tts/checkpoint-1200`)
- `INPUT_TEXT`: Text to convert to speech
- `OUTPUT_FILENAME`: Output audio filename (default: `generated.wav`)

## 🛠️ Setup & Installation

<details>
<summary>Click to expand detailed setup instructions</summary>

### Prerequisites
- Python 3.9+
- CUDA enabled GPU (>4GB VRAM)
- FFmpeg (Must be installed and accessible in your system's PATH)
- Google Gemini API key

### Manual Model Downloads
1. **Vocal Extraction**: Download `kimmel_unwa_ft2_bleedless.ckpt` from [HuggingFace](https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT)
2. **Speaker Embeddings**: Download from [wespeaker-voxceleb-resnet293-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM)

Update model paths in `config.yaml`.

</details>

## ⚙️ Advanced Configuration

<details>
<summary>Click for pipeline control and other advanced options</summary>

### Selective Stage Execution
You can control which parts of the pipeline run, useful for debugging or reprocessing:

```yaml
pipeline_control:
  run_only_stage: "ctc_aligner"      # Run specific stage only (alias: ctc_align)
  start_stage: "llm_diarizer"        # Start from specific stage (alias: llm_diarize)
  end_stage: "outlier_detector"      # Stop at specific stage (alias: outlier_detect)
```

CTC alignment can also be parallelized across episodes (each worker loads its own model copy on the GPU):

```yaml
ctc_aligner:
  num_workers: 4
```

</details>

## 🏗️ Project Structure

The project is organized as follows:

```
TTSizer/
├── configs/
│   └── config.yaml                 # Pipeline & model configurations
├── ttsizer/
│   ├── __init__.py
│   ├── main.py                     # Main script to run the pipeline
│   │── core/                       # Core components of the pipeline
│   ├── models/                     # Vocal removal models
│   └── utils/                      # Utility programs
├── finetune.py                     # Script to fine-tune Spark-TTS model
├── inference.py                    # Simple command-line TTS inference
├── tts_tui.py                      # Terminal-based TTS inference UI
├── tts_gradio.py                   # Web-based TTS inference UI (Gradio)
├── .env                            # For API keys
├── README.md                       # This file
├── requirements.txt                # Python package dependencies
└── weights/                        # For storing downloaded model weights (gitignored)
```

## 📜 License

This project is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 📚 References

*   **Vocals Extraction** [pcunwa/Kim-Mel-Band-Roformer-FT](https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT) by Unwa
*   **Forced Alignment:** [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) by MahmoudAshraf97
*   **ASR:** [NVIDIA NeMo Parakeet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#fast-conformer)
*   **Speaker Embeddings:** [Wespeaker/wespeaker-voxceleb-resnet293-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM) from Wespeaker
