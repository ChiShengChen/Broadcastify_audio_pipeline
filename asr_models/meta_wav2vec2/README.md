# Meta Wav2Vec2 音頻轉錄工具

這個工具使用 Meta 的 Wav2Vec2 模型將音頻文件轉換為英文文字。

## 模型特點

- **模型**: facebook/wav2vec2-large-960h-lv60-self
- **語言**: 英文 (English)
- **參數量**: ~300M
- **特色**: 在 LibriSpeech 960小時英文語料庫上訓練的高性能模型

## 安裝依賴

在使用前，請先安裝必要的依賴：

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法1：使用簡單運行腳本（推薦）

```bash
python run_transcription.py
```

### 方法2：直接運行主腳本

```bash
python wav2vec_transcribe.py
```

## 功能特點

- 自動遍歷 `/media/meow/One Touch/ems_call/long_calls_filtered` 目錄中的所有子目錄
- 自動找到每個子目錄中的 .wav 文件
- 使用 Wav2Vec2 英文模型進行轉錄
- 轉錄結果保存為 `wav2vec-xls-r_{原始文件名}.txt` 格式
- 自動將長音頻分割成30秒片段處理，避免記憶體問題
- 自動跳過已存在的轉錄文件
- 支持 GPU 加速（如果可用）
- 詳細的進度日誌

## 輸出格式

轉錄文件將保存在與原始音頻文件相同的目錄中，文件名格式為：
```
wav2vec-xls-r_{原始wav文件名}.txt
```

例如：
- 原始文件：`202412061019-707526-14744_call_6.wav`
- 轉錄文件：`wav2vec-xls-r_202412061019-707526-14744_call_6.txt`

## 處理方式

- 音頻文件會自動重新採樣到 16kHz
- 長音頻文件會被分割成30秒片段進行處理
- 每個片段的轉錄結果會自動合併

## 系統要求

- Python 3.8+
- PyTorch（支持CUDA優化）
- Transformers 庫
- Librosa 音頻處理庫
- 足夠的磁盤空間（首次運行會下載約1.2GB的模型文件）

## 與 Whisper 的比較

| 特性 | Wav2Vec2 | Whisper |
|------|----------|---------|
| 模型大小 | ~1.2GB | ~3GB |
| 語言專精 | 英文優化 | 多語言通用 |
| 記憶體需求 | 較低 | 較高 |
| 處理方式 | 片段處理 | 整體處理 |

## 注意事項

- 首次運行時會自動下載 Wav2Vec2 模型（約1.2GB）
- 如果系統有GPU，會自動使用GPU加速
- 長音頻文件會被自動分割處理，確保記憶體使用效率
- 模型專為英文語音識別優化，對英文語音有更好的識別效果 