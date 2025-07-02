# OpenAI Whisper 英文音頻轉錄工具

這個工具使用 OpenAI Whisper large-v3 模型將音頻文件轉換為英文文字。

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
python whisper_transcribe.py
```

## 功能特點

- 自動遍歷 `/media/meow/One Touch/ems_call/long_calls_filtered` 目錄中的所有子目錄
- 自動找到每個子目錄中的 .wav 文件
- 使用 Whisper large-v3 模型進行英文轉錄
- 轉錄結果保存為 `large-v3_{原始文件名}.txt` 格式
- 自動跳過已存在的轉錄文件
- 支持 GPU 加速（如果可用）
- 專為英文語音優化
- 詳細的進度日誌

## 輸出格式

轉錄文件將保存在與原始音頻文件相同的目錄中，文件名格式為：
```
large-v3_{原始wav文件名}.txt
```

例如：
- 原始文件：`202412061019-707526-14744_call_6.wav`
- 轉錄文件：`large-v3_202412061019-707526-14744_call_6.txt`

## 系統要求

- Python 3.8+
- PyTorch（支持CUDA優化）
- OpenAI Whisper
- 足夠的磁盤空間（首次運行會下載約3GB的模型文件）

## 注意事項

- 首次運行時會自動下載 Whisper large-v3 模型（約3GB），請確保網絡連接穩定
- 如果系統有GPU，會自動使用GPU加速，大幅提升轉錄速度
- 轉錄語言設定為英文（en），如需更改請修改代碼中的 `language="en"` 參數 