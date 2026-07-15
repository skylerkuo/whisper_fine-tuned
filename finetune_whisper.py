# pip install transformers datasets torch torchaudio librosa evaluate jiwer pandas

import os
import pandas as pd
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchaudio.transforms as T
import random

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 微調後的val loss 大概在 0.7 0.6

def prepare_dataset():
    voice_dir = "voice"
    csv_path = "voice.csv"

    try:
        df = pd.read_csv(csv_path, header=0)
        transcripts = df['question'].tolist()
    except:
        df = pd.read_csv(csv_path, header=None)
        transcripts = df[0].tolist()

    data = []
    for i, text in enumerate(transcripts):
        wav_path = os.path.join(voice_dir, f"{i}.wav")
        if os.path.exists(wav_path):
            data.append((wav_path, str(text).strip()))
        print(i, wav_path, text)
   
    return data

MODEL_NAME = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="zh", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

peft_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, peft_cfg)

class WhisperDataset(torch.utils.data.Dataset):
    def __init__(self, data, apply_pitch_shift=True):
        self.data = data
        self.apply_pitch_shift = apply_pitch_shift

    def __getitem__(self, idx):
        path, text = self.data[idx]
        waveform, sr = torchaudio.load(path)

        if sr != 16000:
            resample = T.Resample(orig_freq=sr, new_freq=16000)
            waveform = resample(waveform)
            sr = 16000

        if self.apply_pitch_shift:
            n_steps = random.randint(-5.0, 5.0)
            if n_steps != 0:
                try:
                    pitch_shift = T.PitchShift(sr, n_steps=n_steps)
                    waveform = pitch_shift(waveform)
                except Exception as e:
                    print(f"[Warning] Pitch shift failed: {e}")

        input_features = processor(waveform.squeeze().detach().numpy(), sampling_rate=16000).input_features[0]
        labels = processor.tokenizer(text).input_ids
        return torch.from_numpy(input_features).float(), torch.tensor(labels)

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    labels_padded[labels_padded == processor.tokenizer.pad_token_id] = -100
    return {"input_features": inputs, "labels": labels_padded}

def train():
    all_data = prepare_dataset()
    train_size = int(0.9 * len(all_data))
    val_size = len(all_data) - train_size
    train_data, val_data = random_split(all_data, [train_size, val_size])

    train_set = WhisperDataset(train_data, apply_pitch_shift=True)
    val_set = WhisperDataset(val_data, apply_pitch_shift=False)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(12):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]"):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]"):
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_features=input_features, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")

    model.save_pretrained("./whisper-finetuned-model-base_V1")
    processor.save_pretrained("./whisper-finetuned-model-base_V1")
    print("訓練完成並儲存模型")

if __name__ == "__main__":
    train()

