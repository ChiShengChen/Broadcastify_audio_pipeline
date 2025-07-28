# 整合ASR管道使用指南

## 概述

這個整合的ASR管道將音頻預處理功能與原有的ASR評估管道完美結合，確保所有音頻文件都能在所有ASR模型上運行。

## ? 主要特色

### 1. **音頻預處理整合**
- 自動處理不同模型的音頻限制
- 支援時長調整、採樣率轉換、音量標準化
- 確保所有音頻符合各模型要求

### 2. **完整工作流程**
- 音頻預處理 → VAD處理 → ASR轉錄 → 評估
- 支援長音頻分割和標註預處理
- 完整的錯誤處理和日誌記錄

### 3. **模型兼容性**
- **Whisper (large-v3)**: 最靈活，幾乎無限制
- **Canary-1b**: 0.5-60秒，16kHz，單聲道
- **Parakeet-tdt-0.6b-v2**: 1.0-300秒，16kHz，單聲道
- **Wav2Vec2-xls-r**: 0.1秒以上，16kHz，單聲道

## ? 文件結構

```
ems_call/
├── run_integrated_pipeline.sh      # 整合管道主腳本
├── audio_preprocessor.py           # 音頻預處理程式
├── generate_test_data.py           # 測試數據生成器
├── test_audio_preprocessor.py      # 預處理測試腳本
├── run_complete_test.sh            # 完整測試腳本
├── AUDIO_PREPROCESSING_GUIDE.md   # 預處理詳細指南
└── INTEGRATED_PIPELINE_GUIDE.md   # 本指南
```

## ? 快速開始

### 1. **生成測試數據**
```bash
# 生成測試音頻文件和標註
python3 generate_test_data.py \
    --output_dir ./test_data \
    --create_ground_truth \
    --verbose
```

### 2. **運行完整測試**
```bash
# 運行完整的測試套件
./run_complete_test.sh
```

### 3. **使用整合管道**
```bash
# 基本使用
./run_integrated_pipeline.sh \
    --input_dir ./test_data \
    --output_dir ./pipeline_results

# 高級選項
./run_integrated_pipeline.sh \
    --input_dir ./test_data \
    --output_dir ./pipeline_results \
    --use-audio-preprocessing \
    --use-vad \
    --use-long-audio-split \
    --preprocess-ground-truth
```

## ? 配置選項

### 音頻預處理選項
```bash
--use-audio-preprocessing          # 啟用音頻預處理
--no-audio-preprocessing           # 禁用音頻預處理
```

### VAD選項
```bash
--use-vad                         # 啟用VAD處理
--vad-threshold FLOAT             # 語音檢測閾值 (預設: 0.5)
--vad-min-speech FLOAT            # 最小語音持續時間 (預設: 0.5s)
--vad-min-silence FLOAT           # 最小靜音持續時間 (預設: 0.3s)
```

### 長音頻分割選項
```bash
--use-long-audio-split            # 啟用長音頻分割
--max-segment-duration FLOAT      # 最大片段持續時間 (預設: 120.0s)
```

### 標註預處理選項
```bash
--preprocess-ground-truth          # 啟用標註預處理
--no-preprocess-ground-truth       # 禁用標註預處理
--preprocess-mode MODE             # 預處理模式 (conservative/aggressive)
```

## ? 輸出結構

```
pipeline_results_YYYYMMDD_HHMMSS/
├── preprocessed_audio/            # 預處理後的音頻文件
│   ├── audio1_large-v3.wav
│   ├── audio1_canary-1b.wav
│   ├── audio1_parakeet-tdt-0.6b-v2.wav
│   └── audio1_wav2vec-xls-r.wav
├── long_audio_segments/           # 長音頻分割結果
├── vad_segments/                  # VAD處理結果
├── asr_transcripts/               # ASR轉錄結果
├── merged_transcripts/            # 合併的轉錄結果
├── asr_evaluation_results.csv     # 評估結果
├── model_file_analysis.txt        # 模型文件分析
├── integration_summary.txt        # 整合摘要
└── error.log                      # 錯誤日誌
```

## ? 測試功能

### 1. **快速測試**
```bash
# 檢查依賴項並運行基本測試
./quick_start.sh
```

### 2. **預處理測試**
```bash
# 測試音頻預處理功能
python3 test_audio_preprocessor.py
```

### 3. **完整測試**
```bash
# 運行完整的測試套件
./run_complete_test.sh
```

### 4. **自定義測試**
```bash
# 生成自定義測試數據
python3 generate_test_data.py \
    --output_dir ./my_test_data \
    --num_files 5 \
    --create_ground_truth \
    --verbose

# 測試預處理
python3 audio_preprocessor.py \
    --input_dir ./my_test_data \
    --output_dir ./my_preprocessed \
    --verbose
```

## ? 性能優化

### 1. **批量處理**
```bash
# 一次處理多個音頻文件
./run_integrated_pipeline.sh \
    --input_dir /path/to/large/audio/dataset \
    --output_dir /path/to/results
```

### 2. **並行處理**
```python
# 在 audio_preprocessor.py 中可以啟用多進程
# 修改 num_workers 參數
```

### 3. **記憶體優化**
```bash
# 使用長音頻分割避免OOM
./run_integrated_pipeline.sh \
    --use-long-audio-split \
    --max-segment-duration 60
```

## ? 故障排除

### 1. **常見問題**

**問題：音頻文件無法讀取**
```bash
# 解決方案：檢查文件格式
file audio_file.wav
# 確保文件未損壞且格式正確
```

**問題：記憶體不足**
```bash
# 解決方案：啟用長音頻分割
./run_integrated_pipeline.sh \
    --use-long-audio-split \
    --max-segment-duration 60
```

**問題：預處理失敗**
```bash
# 解決方案：檢查依賴項
python3 -c "import soundfile, librosa, numpy; print('Dependencies OK')"
```

### 2. **調試技巧**

```bash
# 啟用詳細日誌
./run_integrated_pipeline.sh \
    --input_dir ./test_data \
    --output_dir ./debug_results \
    --verbose

# 檢查音頻信息
python3 -c "
import soundfile as sf
info = sf.info('audio.wav')
print(f'Duration: {info.duration}s')
print(f'Sample rate: {info.samplerate}Hz')
print(f'Channels: {info.channels}')
"
```

## ? 使用範例

### 範例1：基本使用
```bash
# 使用預設配置運行管道
./run_integrated_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv
```

### 範例2：高級配置
```bash
# 使用所有功能
./run_integrated_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv \
    --use-audio-preprocessing \
    --use-vad \
    --vad-threshold 0.6 \
    --use-long-audio-split \
    --max-segment-duration 90 \
    --preprocess-ground-truth \
    --preprocess-mode aggressive
```

### 範例3：測試模式
```bash
# 生成測試數據並運行
python3 generate_test_data.py --create_ground_truth
./run_integrated_pipeline.sh \
    --input_dir ./test_data \
    --output_dir ./test_results
```

## ? 監控和報告

### 1. **進度監控**
```bash
# 查看處理進度
tail -f pipeline_results_*/integration_summary.txt
```

### 2. **結果分析**
```bash
# 查看評估結果
cat pipeline_results_*/asr_evaluation_results.csv

# 查看模型分析
cat pipeline_results_*/model_file_analysis.txt
```

### 3. **錯誤檢查**
```bash
# 檢查錯誤日誌
cat pipeline_results_*/error.log

# 檢查警告
grep "WARNING" pipeline_results_*/error.log
```

## ? 最佳實踐

### 1. **數據準備**
- 確保音頻文件格式正確（WAV, MP3, M4A, FLAC）
- 檢查標註文件格式（CSV with Filename, transcript columns）
- 驗證文件路徑和權限

### 2. **系統要求**
- Python 3.7+
- 足夠的磁碟空間（預處理會生成額外文件）
- 建議8GB+ RAM用於大文件處理

### 3. **性能優化**
- 使用SSD存儲以提高I/O性能
- 根據系統配置調整並行處理參數
- 定期清理臨時文件

## ? 更新和維護

### 1. **檢查更新**
```bash
# 檢查腳本版本和依賴項
python3 -c "import soundfile, librosa, numpy; print('All dependencies up to date')"
```

### 2. **備份配置**
```bash
# 備份重要配置
cp run_integrated_pipeline.sh run_integrated_pipeline.sh.backup
```

### 3. **清理舊文件**
```bash
# 清理舊的測試和結果文件
rm -rf test_data_* pipeline_results_* preprocessed_*
```

## ? 支援

如果遇到問題，請檢查：

1. **依賴項安裝**
```bash
pip install soundfile librosa numpy scipy
```

2. **文件權限**
```bash
chmod +x run_integrated_pipeline.sh
```

3. **Python路徑**
```bash
python3 --version
which python3
```

4. **系統資源**
```bash
df -h  # 檢查磁碟空間
free -h  # 檢查記憶體
```

## ? 總結

這個整合的ASR管道提供了：

- ? **完整的音頻預處理**：確保所有音頻符合模型要求
- ? **靈活的配置選項**：支援各種使用場景
- ? **完整的測試套件**：確保功能正確性
- ? **詳細的錯誤處理**：提供清晰的錯誤信息
- ? **全面的文檔**：包含使用指南和最佳實踐

通過這個整合管道，您可以確保所有音頻文件都能在所有ASR模型上成功運行，大大提高了管道的穩定性和成功率！ 