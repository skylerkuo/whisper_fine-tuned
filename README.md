# Whisper Fine-tuning with LoRA (Speech Recognition)

本專案使用 **OpenAI Whisper Base** 搭配 **LoRA (Low-Rank Adaptation)** 進行語音辨識微調，並使用自建的語音資料集進行訓練。

---

## 專案特色

- 使用 **Whisper Base** 作為預訓練模型
- 使用 **LoRA** 進行參數高效微調
- 支援中文語音辨識
- 自動將音訊重採樣至 16kHz
- 使用 Pitch Shift 進行資料增強 (Data Augmentation)
- 自動切分 Training / Validation Dataset
- 訓練完成後自動儲存模型與 Processor

---

## 專案架構

```
project/
│
├── finetune_whisper.py
├── voice.csv
├── voice/
│   ├── 0.wav
│   ├── 1.wav
│   ├── 2.wav
│   └── ...
└── README.md
```

---

## Dataset 格式

### voice.csv

CSV 第一欄需為文字內容，例如：

| question |
|----------|
| 你好 |
| 今天天氣很好 |
| 我要去上班 |

如果沒有欄位名稱，也可以只有一欄文字。

---

### voice 資料夾

音檔名稱需依照 CSV 順序命名：

```
voice/
├──0.wav
├──1.wav
├──2.wav
...
```

例如：

| CSV 第幾列 | 對應音檔 |
|------------|----------|
| 第一列 | 0.wav |
| 第二列 | 1.wav |
| 第三列 | 2.wav |

---

## 安裝套件

```bash
pip install torch torchaudio  #自己版本要自己看
pip install transformers
pip install peft
pip install pandas
pip install tqdm
pip install librosa
pip install datasets
pip install evaluate
pip install jiwer
```

---

## 模型

本專案使用：

```
openai/whisper-base
```

LoRA 設定：

```python
r = 8
lora_alpha = 32

target_modules = [
    "q_proj",
    "v_proj"
]
```

---

## 訓練流程

1. 讀取 `voice.csv`
2. 對應每筆文字與 wav 音檔
3. 將音訊重採樣為 16kHz
4. 訓練集進行 Pitch Shift 資料增強
5. 使用 Whisper Processor 轉換為 Mel Spectrogram
6. Tokenize 文字標籤
7. 使用 LoRA 微調 Whisper
8. 計算 Training Loss
9. 計算 Validation Loss
10. 儲存模型

---

## Train / Validation Split

資料比例：

```
Training : 90%

Validation : 10%
```

---

## Data Augmentation

訓練集使用 Pitch Shift：

```python
random.randint(-5, 5)
```
隨機將聲音調高或調低五度範圍以內

可增加模型對不同音高的適應能力

Validation 不進行資料增強。

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | Whisper Base |
| Epoch | 12 |
| Batch Size | 1 |
| Learning Rate | 1e-5 |
| Optimizer | AdamW |
| Sample Rate | 16000 Hz |

---

## 執行方式

```bash
python finetune_whisper.py
```

---



