import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os, wave
import pyaudio

CHUNK      = 1024
FORMAT     = pyaudio.paInt16
CHANNELS   = 1
RATE       = 16000
SECONDS    = 5
SAVE_DIR   = "recordings"

os.makedirs(SAVE_DIR, exist_ok=True)

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 frames_per_buffer=CHUNK,
                 input=True)

device = "cpu" 

model_name = "openai/whisper-base"

processor = WhisperProcessor.from_pretrained(
    model_name,
    language="zh",
    task="transcribe"
)

model = WhisperForConditionalGeneration.from_pretrained(
    model_name
).to(device)

model.eval()

def generate_text_from_audio(audio_path):
    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} does not exist.")
        return None

    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = model.generate(
            **inputs,
            max_new_tokens=100
        )

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return text

def get_voice_to_text():
    print(f"start recording for {SECONDS} seconds…")

    frames = []
    for _ in range(int(RATE / CHUNK * SECONDS)):
        frames.append(stream.read(CHUNK))

    print("transfer…")

    wav_path = os.path.join(SAVE_DIR, "voice.wav")

    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    result = generate_text_from_audio(wav_path)

    return result

