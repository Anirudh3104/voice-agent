import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# Load model & tokenizers
# -----------------------------
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts"
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "ai4bharat/indic-parler-tts"
)

description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

# -----------------------------
# Telugu text to speak
# -----------------------------
prompt = "మీకు ఏ ప్రభుత్వ పథకం కావాలి?"

# Voice / style description
description = (
    "Lalitha speaks in a clear, calm Telugu voice. "
    "The speech is neutral, natural, and easy to understand. "
    "The recording is very clear with no background noise."
)

# -----------------------------
# Tokenize
# -----------------------------
prompt_inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

description_inputs = description_tokenizer(
    description,
    return_tensors="pt"
).to(device)

# -----------------------------
# Generate speech
# -----------------------------
with torch.no_grad():
    generation = model.generate(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask,
    )

# -----------------------------
# Save audio
# -----------------------------
audio = generation.cpu().numpy().squeeze()
sf.write("tts_output_telugu.wav", audio, model.config.sampling_rate)

print("🔊 Telugu TTS saved as tts_output_telugu.wav")