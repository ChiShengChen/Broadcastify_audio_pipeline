# Pipeline Enhancements

本文檔描述了對 `run_llm_pipeline.sh` 的新增功能。

## 新增功能

### 1. 多ASR比較功能 (Multi-ASR Comparison)

**描述**: 允許LLM比較和合併來自不同ASR系統（Canary和Whisper）的轉錄結果。

**開關**: `--enable_multi_asr_comparison`

**模型**: `MULTI_ASR_COMPARISON_MODEL` (預設: gpt-oss-20b)

**Prompt**: `MULTI_ASR_COMPARISON_PROMPT`

**用法**:
```bash
./run_llm_pipeline.sh \
  --input_dir /path/to/asr/results \
  --output_dir /path/to/output \
  --enable_multi_asr_comparison \
  --medical_correction_model gpt-oss-20b
```

**輸出**: 合併後的轉錄文件

### 2. ASR選擇功能 (ASR Selection) - 新功能

**描述**: 允許LLM比較Canary和Whisper的ASR結果，選擇更好的那個結果進入下一個處理階段，而不是合併它們。

**開關**: `--enable_asr_selection`

**模型**: `MEDICAL_CORRECTION_MODEL` (使用相同的醫療校正模型)

**Prompt**: `ASR_SELECTION_PROMPT`

**用法**:
```bash
./run_llm_pipeline.sh \
  --input_dir /path/to/asr/results \
  --output_dir /path/to/output \
  --enable_asr_selection \
  --medical_correction_model gpt-oss-20b
```

**輸出**: 
- 選擇的轉錄文件
- CSV報告文件 (`asr_selection_results.csv`)

**CSV報告內容**:
- filename: 文件名
- selected_asr: 選擇的ASR (canary/whisper)
- reason: 選擇原因
- accuracy_score: 準確性評分 (1-10)
- completeness_score: 完整性評分 (1-10)
- medical_terminology_score: 醫療術語評分 (1-10)

### 3. 信息提取功能 (Information Extraction)

**描述**: 從轉錄中提取結構化信息（JSON格式）。

**開關**: `--enable_information_extraction`

**模型**: `EXTRACTION_MODEL` (預設: gpt-oss-20b)

**Prompt**: `INFORMATION_EXTRACTION_PROMPT`

**用法**:
```bash
./run_llm_pipeline.sh \
  --input_dir /path/to/asr/results \
  --output_dir /path/to/output \
  --enable_information_extraction \
  --extraction_model gpt-oss-20b
```

**輸出**: JSON格式的結構化信息

### 4. 增強處理功能 (Enhanced Processing)

**描述**: 使用提取的JSON數據進行進一步處理。

**開關**: `--enable_enhanced_processing`

**模型**: `ENHANCEMENT_MODEL` (預設: gpt-oss-20b)

**Prompt**: `ENHANCED_PROCESSING_PROMPT`

**用法**:
```bash
./run_llm_pipeline.sh \
  --input_dir /path/to/asr/results \
  --output_dir /path/to/output \
  --enable_information_extraction \
  --enable_enhanced_processing \
  --extraction_model gpt-oss-20b \
  --enhancement_model gpt-oss-20b
```

**輸出**: 增強處理後的結果

## 工作流程

### 標準工作流程
1. ASR轉錄 → 2. 醫療術語校正 → 3. 緊急頁面生成 → 4. 評估

### 增強工作流程
1. ASR轉錄 → 2. 醫療術語校正 (可選多ASR比較/選擇) → 3. 信息提取 → 4. 增強處理 → 5. 緊急頁面生成 (可選) → 6. 評估

## 輸出結構

```
output_dir/
├── medical_corrected/           # 醫療校正結果
│   ├── *.txt                   # 校正後的轉錄
│   └── asr_selection_results.csv  # ASR選擇報告 (僅選擇模式)
├── extracted_information/       # 提取的信息 (JSON)
├── enhanced_processing/         # 增強處理結果
├── emergency_pages/            # 緊急頁面 (可選)
└── evaluation/                 # 評估結果
```

## 模式比較

| 功能 | 合併模式 | 選擇模式 |
|------|----------|----------|
| 開關 | `--enable_multi_asr_comparison` | `--enable_asr_selection` |
| 行為 | LLM合併兩個ASR結果 | LLM選擇更好的ASR結果 |
| 輸出 | 單一合併後的轉錄 | 選擇的轉錄 + CSV報告 |
| 適用場景 | 需要綜合兩個ASR的優勢 | 需要選擇最佳ASR結果 |

## 最佳實踐

1. **模式選擇**: 根據需求選擇合併模式或選擇模式
2. **模型選擇**: 使用較大的模型獲得更好的結果
3. **批量處理**: 支持批量處理多個文件
4. **錯誤處理**: 包含fallback機制
5. **報告分析**: 定期分析CSV報告以優化策略

## 兼容性

- ✅ 與現有功能完全兼容
- ✅ 支持所有LLM模型
- ✅ 支持自動檢測多ASR結果
- ✅ 與信息提取和增強處理功能兼容
