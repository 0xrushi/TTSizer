import argparse
import math
import torch
import torch.nn.functional as F
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def tokens_to_words_whisper(tokenizer, token_ids, token_logprobs):
    """
    Correct Whisper BPE word merging.
    Uses leading-space tokens as word boundaries.
    """
    words = []
    cur_word = ""
    cur_logps = []

    for tid, lp in zip(token_ids, token_logprobs):
        tok = tokenizer.decode([tid], skip_special_tokens=True)

        if tok == "":
            continue

        if tok.startswith(" ") and cur_word:
            words.append((cur_word.strip(), cur_logps))
            cur_word = tok.lstrip()
            cur_logps = [lp]
        else:
            cur_word += tok
            cur_logps.append(lp)

    if cur_word:
        words.append((cur_word.strip(), cur_logps))

    return words


def word_confidence_from_logprobs(logps):
    """
    Standard ASR confidence:
    exp(mean log probability)
    """
    return math.exp(sum(logps) / len(logps))


def transcribe(audio_path, model_id="Oriserve/Whisper-Hindi2Hinglish-Prime"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {model_id} on {device}...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    audio, _ = librosa.load(audio_path, sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {
        k: v.to(device=device, dtype=dtype if v.dtype.is_floating_point else None)
        for k, v in inputs.items()
    }

    # 1) Generate tokens
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
            forced_decoder_ids=forced_decoder_ids
        )[0]

    # Reconstruct the full sequence including start tokens if missing
    # Standard Whisper sequence: <|startoftranscript|> <|lang|> <|task|> <|notimestamps|> ...
    
    # Start with decoder_start_token_id
    full_prefix = [model.config.decoder_start_token_id]
    
    # Sort forced_ids by step just in case
    forced_decoder_ids.sort(key=lambda x: x[0])
    
    for _, tid in forced_decoder_ids:
        full_prefix.append(tid)
        
    full_prefix_tensor = torch.tensor(full_prefix, device=device)
    
    # Check if generated_ids already has the prefix
    if len(generated_ids) >= len(full_prefix) and torch.equal(generated_ids[:len(full_prefix)], full_prefix_tensor):
        full_ids = generated_ids
    else:
        full_ids = torch.cat([full_prefix_tensor, generated_ids])

    # 2) Forced-decoding scoring pass
    decoder_input_ids = full_ids[:-1].unsqueeze(0)
    labels = full_ids[1:].unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            **inputs,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )

    # logits → log-probs (float32 for stability)
    log_probs = F.log_softmax(outputs.logits[0].float(), dim=-1)

    token_logprobs = log_probs.gather(
        1, labels[0].unsqueeze(1)
    ).squeeze(1).cpu().tolist()

    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    words = tokens_to_words_whisper(
        processor.tokenizer,
        full_ids[1:].cpu().tolist(),
        token_logprobs,
    )

    print("\nTranscription:")
    print("-" * 60)
    print(text)
    print("-" * 60)

    print("\nWord-level confidence (true ASR likelihood):")
    for word, logps in words:
        conf = word_confidence_from_logprobs(logps)
        print(f"{word:<15} {conf:.6f}")

    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str)
    args = parser.parse_args()

    transcribe(args.audio_path)
