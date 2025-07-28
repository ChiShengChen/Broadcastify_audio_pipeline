# Audio Filter Module

## 概述

這個模組將 band-pass filter 功能從 VAD 中獨立出來，讓你可以單獨使用音頻濾波器而不需要 VAD 處理。

## 功能特點

### 1. 獨立的音頻濾波器 (`audio_filter.py`)
- **High-pass filter**: 移除低頻噪音（如 AC 嗡嗡聲）
- **Band-pass filter**: 專注於語音頻率範圍（300-3000Hz）
- **Wiener filter**: 可選的噪音減少濾波器
- **重採樣**: 自動調整到目標採樣率（預設 16000Hz）

### 2. Pipeline 整合
- 在 `run_pipeline.sh` 中添加了 `--use-audio-filtering` 選項
- 可以單獨使用濾波器，或與 VAD 結合使用
- 完整的配置選項和錯誤處理

## 使用方法

### 基本音頻濾波器使用

```bash
# 基本濾波器（band-pass + high-pass）
python3 audio_filter.py \
    --input_dir /path/to/audio \
    --output_dir /path/to/output \
    --enable-filters

# 包含 Wiener filter
python3 audio_filter.py \
    --input_dir /path/to/audio \
    --output_dir /path/to/output \
    --enable-filters \
    --enable-wiener

# 自定義濾波器參數
python3 audio_filter.py \
    --input_dir /path/to/audio \
    --output_dir /path/to/output \
    --enable-filters \
    --highpass_cutoff 250.0 \
    --lowcut 250.0 \
    --highcut 3500.0 \
    --filter_order 6 \
    --enable-wiener
```

### Pipeline 整合使用

```bash
# 只使用音頻濾波器（無 VAD）
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv \
    --use-audio-filtering

# 音頻濾波器 + VAD
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv \
    --use-audio-filtering \
    --use-vad

# 自定義濾波器參數
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv \
    --use-audio-filtering \
    --filter-highpass-cutoff 250.0 \
    --filter-lowcut 250.0 \
    --filter-highcut 3500.0 \
    --filter-order 6 \
    --filter-enable-wiener
```

## 配置選項

### 音頻濾波器參數
- `--enable-filters`: 啟用音頻濾波器
- `--no-filters`: 禁用音頻濾波器
- `--highpass_cutoff FLOAT`: High-pass 濾波器截止頻率 (Hz)，預設 300.0
- `--lowcut FLOAT`: Band-pass 濾波器低截止頻率 (Hz)，預設 300.0
- `--highcut FLOAT`: Band-pass 濾波器高截止頻率 (Hz)，預設 3000.0
- `--filter_order INT`: 濾波器階數，預設 5
- `--enable-wiener`: 啟用 Wiener 濾波器
- `--target_sample_rate INT`: 目標採樣率 (Hz)，預設 16000

### Pipeline 參數
- `--use-audio-filtering`: 啟用音頻濾波器
- `--no-audio-filtering`: 禁用音頻濾波器
- `--filter-highpass-cutoff FLOAT`: High-pass 濾波器截止頻率
- `--filter-lowcut FLOAT`: Band-pass 濾波器低截止頻率
- `--filter-highcut FLOAT`: Band-pass 濾波器高截止頻率
- `--filter-order INT`: 濾波器階數
- `--filter-enable-wiener`: 啟用 Wiener 濾波器
- `--filter-disable-wiener`: 禁用 Wiener 濾波器

## 測試腳本

運行測試腳本來驗證功能：

```bash
./test_audio_filter.sh
```

這個腳本會執行以下測試：
1. 基本音頻濾波器測試
2. 包含 Wiener 濾波器的測試
3. Pipeline 只使用濾波器（無 VAD）的測試
4. Pipeline 使用濾波器 + VAD 的測試

## 輸出結構

當使用音頻濾波器時，pipeline 會產生以下目錄結構：

```
output_directory/
├── filtered_audio/                    # 音頻濾波器結果
│   ├── filter_processing_metadata.json
│   └── [filtered audio files]
├── asr_transcripts/                   # ASR 轉錄結果
├── asr_evaluation_results.csv         # 評估結果
└── pipeline_summary.txt               # Pipeline 摘要
```

## 與原有功能的比較

### 原有功能
- VAD 包含內建的濾波器功能
- 必須使用 VAD 才能使用濾波器
- 濾波器參數固定

### 新功能
- 獨立的音頻濾波器模組
- 可以單獨使用濾波器而不需要 VAD
- 完全可配置的濾波器參數
- 更好的錯誤處理和日誌記錄

## 注意事項

1. **濾波器順序**: 濾波器按以下順序應用：
   - High-pass filter (移除低頻噪音)
   - Band-pass filter (語音頻率範圍)
   - Wiener filter (可選的噪音減少)

2. **音頻品質**: 濾波器可能會影響音頻品質，建議在測試環境中驗證效果

3. **處理時間**: 添加濾波器會增加處理時間，特別是在處理大量音頻文件時

4. **記憶體使用**: Wiener 濾波器會增加記憶體使用量

## 故障排除

### 常見問題

1. **濾波器沒有生效**
   - 檢查是否正確啟用了 `--enable-filters` 選項
   - 確認濾波器參數設置正確

2. **音頻品質下降**
   - 嘗試調整濾波器參數
   - 考慮禁用 Wiener 濾波器
   - 檢查原始音頻品質

3. **處理速度慢**
   - 考慮禁用 Wiener 濾波器
   - 減少濾波器階數
   - 檢查系統資源

### 錯誤日誌

所有錯誤都會記錄在 pipeline 的錯誤日誌中，位置：
```
output_directory/error_analysis.log
``` 