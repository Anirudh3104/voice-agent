import torch
import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import (
    SeamlessM4Tv2ForSpeechToText,
    SeamlessM4TFeatureExtractor,
    SeamlessM4TTokenizer
)

# -----------------------------
# 1. Record audio
# -----------------------------
fs = 16000
seconds = 10

print("🎙️ తెలుగులో మాట్లాడండి (Speak Telugu for 5 seconds)...")

audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

audio_path = "audio/input.wav"
write(audio_path, fs, audio)

# -----------------------------
# 2. Load model (CPU)
# -----------------------------
device = "cpu"

print("Loading IndicSeamless model (this may take time first run)...")

model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
    "ai4bharat/indic-seamless"
).to(device)

processor = SeamlessM4TFeatureExtractor.from_pretrained(
    "ai4bharat/indic-seamless"
)

tokenizer = SeamlessM4TTokenizer.from_pretrained(
    "ai4bharat/indic-seamless"
)

# -----------------------------
# 3. Load & resample audio
# -----------------------------
waveform, sr = sf.read(audio_path)

# Convert to torch tensor and shape [1, num_samples]
waveform = torch.tensor(waveform).unsqueeze(0)

# Resample if needed
if sr != 16000:
    import torchaudio.functional as F
    waveform = F.resample(waveform, sr, 16000)


# -----------------------------
# 4. Inference
# -----------------------------
inputs = processor(
    waveform,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        tgt_lang="tel"   # Telugu output
    )

text = tokenizer.decode(
    output_tokens[0],
    skip_special_tokens=True
)

print("📝 గుర్తించిన వాక్యం:")
print(text)