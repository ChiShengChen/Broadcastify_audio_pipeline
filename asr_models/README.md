# ASR 模型轉錄工具總覽

這個目錄包含多個自動語音識別（ASR）模型的實現，用於將 EMS 無線電通訊音頻轉錄為英文文字。

## 可用模型

### 1. OpenAI Whisper Large-v3
**目錄**: `openai_whisper/`

- **模型**: OpenAI Whisper Large-v3
- **架構**: Transformer (Encoder-Decoder)
- **模型大小**: ~3GB
- **語言**: 英文
- **特點**: 
  - 業界領先的語音識別準確度
  - 多語言支持（已配置為英文）
  - 強大的噪音抗性
  - 適合複雜音頻環境

**使用方法**:
```bash
cd openai_whisper/
python run_transcription.py
```

### 2. Meta Wav2Vec2
**目錄**: `meta_wav2vec2/`

- **模型**: facebook/wav2vec2-large-960h-lv60-self
- **架構**: CTC (Connectionist Temporal Classification)
- **模型大小**: ~1.2GB
- **語言**: 英文
- **特點**:
  - 輕量級，記憶體需求較低
  - 快速推理速度
  - 專為英文優化
  - 片段處理，記憶體效率高

**使用方法**:
```bash
cd meta_wav2vec2/
python run_transcription.py
```

### 3. NVIDIA Parakeet TDT 0.6B v2
**目錄**: `nvidia_parakeetTDT/`

- **模型**: NVIDIA Parakeet TDT 0.6B v2
- **架構**: CTC (Connectionist Temporal Classification)
- **模型大小**: ~2.4GB
- **語言**: 英文
- **特點**:
  - 企業級性能
  - NVIDIA 優化
  - 平衡的準確度和效率
  - 支援 NeMo 和 Hugging Face 兩種載入方式

**使用方法**:
```bash
cd nvidia_parakeetTDT/
python run_transcription.py
```

## 模型比較

| 特性 | Whisper Large-v3 | Wav2Vec2 | Parakeet TDT |
|------|-----------------|----------|--------------|
| **準確度** | 最高 | 高 | 高 |
| **速度** | 中等 | 快 | 快 |
| **模型大小** | 3GB | 1.2GB | 2.4GB |
| **記憶體需求** | 高 | 低 | 中等 |
| **噪音抗性** | 最佳 | 良好 | 良好 |
| **企業支援** | ✓ | - | ✓ |
| **安裝複雜度** | 簡單 | 簡單 | 複雜 |

## 通用功能

所有模型都支援以下功能：

- ✅ **自動目錄遍歷**: 處理 `long_calls_filtered` 中的所有子目錄
- ✅ **批次處理**: 自動處理多個音頻文件
- ✅ **斷點續傳**: 跳過已存在的轉錄文件
- ✅ **GPU 加速**: 自動檢測並使用 CUDA
- ✅ **片段處理**: 長音頻自動分割，避免記憶體問題
- ✅ **詳細日誌**: 完整的處理過程記錄
- ✅ **錯誤恢復**: 單一文件失敗不影響整體處理

## 輸出格式

所有模型的轉錄結果都保存在原始音頻文件的同一目錄中：

- **Whisper**: `large-v3_{原始文件名}.txt`
- **Wav2Vec2**: `wav2vec-xls-r_{原始文件名}.txt`
- **Parakeet TDT**: `parakeet-tdt-0.6b-v2_{原始文件名}.txt`

## 建議使用場景

### 高準確度需求
推薦使用 **OpenAI Whisper Large-v3**
- 最高的轉錄準確度
- 最佳的噪音處理能力
- 適合重要的音頻內容

### 快速處理需求
推薦使用 **Meta Wav2Vec2**
- 最快的處理速度
- 最低的資源需求
- 適合大批量處理

### 企業環境需求
推薦使用 **NVIDIA Parakeet TDT**
- 企業級支持
- 平衡的性能和準確度
- NVIDIA 生態系統整合

## 系統要求

### 最低要求
- Python 3.8+
- 8GB RAM
- 10GB 可用磁盤空間

### 推薦配置
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 50GB+ 可用磁盤空間
- 穩定的網絡連接（首次模型下載）

## 安裝指南

每個模型目錄都包含獨立的 `requirements.txt` 文件。請在使用前進入相應目錄並安裝依賴：

```bash
cd [模型目錄]/
pip install -r requirements.txt
```

## 注意事項

1. **首次運行**: 所有模型首次運行時都會下載模型文件，請確保網絡連接穩定
2. **CUDA 支援**: 強烈建議使用 GPU 以獲得最佳性能
3. **磁盤空間**: 確保有足夠空間存儲模型文件和轉錄結果
4. **音頻格式**: 所有模型都支援 WAV 格式，會自動處理採樣率轉換

## 故障排除

如果遇到問題，請查看各模型目錄中的 README.md 文件，或檢查：

1. 依賴包是否正確安裝
2. CUDA 驅動是否最新
3. 磁盤空間是否充足
4. 網絡連接是否穩定 