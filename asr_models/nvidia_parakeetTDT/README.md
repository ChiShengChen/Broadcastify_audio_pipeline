# NVIDIA Parakeet TDT 0.6B v2 音頻轉錄工具

這個工具使用 NVIDIA Parakeet TDT 0.6B v2 模型將音頻文件轉換為英文文字。

## 模型特點

- **模型**: NVIDIA Parakeet TDT 0.6B v2
- **架構**: CTC (Connectionist Temporal Classification)
- **語言**: 英文 (English)
- **參數量**: ~600M
- **特色**: NVIDIA 開發的高性能語音識別模型，專為企業級應用設計

## 安裝依賴

在使用前，請先安裝必要的依賴：

```bash
pip install -r requirements.txt
```

**注意**: NeMo 套件較大，安裝時間可能較長。建議在良好的網絡環境下進行安裝。

## 使用方法

### 方法1：使用簡單運行腳本（推薦）

```bash
python run_transcription.py
```

### 方法2：直接運行主腳本

```bash
python parakeet_transcribe.py
```

## 功能特點

- 自動遍歷 `/media/meow/One Touch/ems_call/long_calls_filtered` 目錄中的所有子目錄
- 自動找到每個子目錄中的 .wav 文件
- 使用 NVIDIA Parakeet TDT 0.6B v2 模型進行英文轉錄
- 轉錄結果保存為 `parakeet-tdt-0.6b-v2_{原始文件名}.txt` 格式
- 自動將長音頻分割成30秒片段處理，避免記憶體問題
- 自動跳過已存在的轉錄文件
- 支持 GPU 加速（如果可用）
- 支持 Hugging Face Transformers 和 NVIDIA NeMo 兩種載入方式
- 詳細的進度日誌

## 輸出格式

轉錄文件將保存在與原始音頻文件相同的目錄中，文件名格式為：
```
parakeet-tdt-0.6b-v2_{原始wav文件名}.txt
```

例如：
- 原始文件：`202412061019-707526-14744_call_6.wav`
- 轉錄文件：`parakeet-tdt-0.6b-v2_202412061019-707526-14744_call_6.txt`

## 處理方式

- 音頻文件會自動重新採樣到 16kHz
- 長音頻文件會被分割成30秒片段進行處理
- 每個片段的轉錄結果會自動合併
- 支持錯誤恢復：單個片段失敗不會影響整體轉錄

## 模型載入方式

工具支持兩種模型載入方式：

1. **Hugging Face Transformers** (優先嘗試)
2. **NVIDIA NeMo** (備用方案)

系統會自動選擇可用的載入方式。

## 系統要求

- Python 3.8+
- PyTorch（支持CUDA優化）
- NVIDIA NeMo 工具包
- Transformers 庫
- Librosa 音頻處理庫
- 足夠的磁盤空間（首次運行會下載約2.4GB的模型文件）

## 與其他模型的比較

| 特性 | Parakeet TDT | Whisper | Wav2Vec2 |
|------|-------------|---------|----------|
| 模型大小 | ~2.4GB | ~3GB | ~1.2GB |
| 架構 | CTC | Transformer | CTC |
| 語言專精 | 英文優化 | 多語言通用 | 英文優化 |
| 記憶體需求 | 中等 | 較高 | 較低 |
| 處理方式 | 片段處理 | 整體處理 | 片段處理 |
| 企業級 | ✓ | ✓ | - |

## 注意事項

- 首次運行時會自動下載 NVIDIA Parakeet TDT 模型（約2.4GB）
- 如果系統有GPU，會自動使用GPU加速
- NeMo 套件安裝可能需要較長時間，請耐心等待
- 長音頻文件會被自動分割處理，確保記憶體使用效率
- 模型專為英文語音識別優化，對英文語音有優秀的識別效果
- 支持多種音頻格式，會自動轉換為模型所需的格式

## 故障排除

### 模型載入失敗
如果遇到模型載入問題，請確保：
1. 網絡連接穩定
2. 有足夠的磁盤空間
3. CUDA 驅動程序已正確安裝（如使用GPU）

### 記憶體不足
如果遇到記憶體問題：
1. 系統會自動使用片段處理
2. 可以調整 `chunk_length` 參數
3. 考慮使用CPU模式 