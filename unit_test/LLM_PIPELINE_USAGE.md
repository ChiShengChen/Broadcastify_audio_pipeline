# LLM-Enhanced ASR Pipeline 使用說明

## 🎯 功能概述

這個整合版本的LLM pipeline包含了所有功能：

1. **Whisper過濾**: 自動過濾只包含 `large-v3_` 的ASR文件（可選）
2. **醫學名詞修正**: 使用LLM修正醫學術語
3. **急診Page生成**: 基於修正後的文本生成結構化急診頁面
4. **可選評估**: 與ground truth進行比較

## 🚀 快速開始

### 基本用法（推薦 - 包含Whisper過濾）

```bash
# 處理Whisper結果，使用默認設置
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902"
```

### 處理所有ASR結果（不限制Whisper）

```bash
# 禁用Whisper過濾，處理所有ASR結果
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902" \
    --disable_whisper_filter
```

### 自定義模型

```bash
# 使用醫學專用模型
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902" \
    --medical_correction_model BioMistral-7B \
    --page_generation_model Meditron-7B
```

### 只進行醫學修正

```bash
# 只修正醫學名詞，不生成急診頁面
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902" \
    --disable_page_generation
```

### 包含評估

```bash
# 添加與ground truth的比較
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902" \
    --ground_truth "/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv"
```

## 📁 輸出結構

```
llm_enhanced_results_YYYYMMDD_HHMMSS/
├── whisper_filtered/              # 過濾後的Whisper文件（如果啟用）
├── corrected_transcripts/         # 醫學名詞修正後的文本
├── emergency_pages/              # 生成的急診頁面
├── llm_enhanced_evaluation_results.csv  # 評估結果（如果啟用）
├── llm_enhanced_pipeline_summary.txt    # 處理摘要
└── error_analysis.log            # 錯誤日誌
```

## ⚙️ 主要配置選項

### 功能開關
- `--enable_whisper_filter` / `--disable_whisper_filter`: 控制Whisper過濾
- `--enable_medical_correction` / `--disable_medical_correction`: 控制醫學修正
- `--enable_page_generation` / `--disable_page_generation`: 控制頁面生成
- `--enable_evaluation` / `--disable_evaluation`: 控制評估

### 模型選擇
- **醫學修正模型**: `gpt-oss-20b`, `BioMistral-7B`, `Meditron-7B`, `Llama-3-8B-UltraMedica`
- **頁面生成模型**: 同上

### 設備配置
- `--device auto`: 自動選擇（推薦）
- `--device cpu`: 強制使用CPU
- `--device cuda`: 強制使用GPU

### 量化選項
- `--load_in_8bit`: 8位量化（節省內存）
- `--load_in_4bit`: 4位量化（最節省內存）

## 🔧 故障排除

### 常見問題

1. **模型下載失敗**
   ```bash
   # 使用量化選項
   --load_in_8bit --device cpu
   ```

2. **內存不足**
   ```bash
   # 使用4位量化
   --load_in_4bit --batch_size 1
   ```

3. **找不到Whisper文件**
   ```bash
   # 禁用過濾處理所有文件
   --disable_whisper_filter
   ```

### 日誌檢查

```bash
# 查看錯誤日誌
cat llm_enhanced_results_*/error_analysis.log

# 查看處理摘要
cat llm_enhanced_results_*/llm_enhanced_pipeline_summary.txt
```

## 📊 性能優化

### 推薦配置

**CPU環境**:
```bash
--device cpu --load_in_8bit --batch_size 1
```

**GPU環境**:
```bash
--device cuda --batch_size 3
```

**內存受限環境**:
```bash
--device cpu --load_in_4bit --batch_size 1
```

## 🎯 使用場景

1. **快速測試**: 使用默認設置處理少量文件
2. **生產環境**: 使用醫學專用模型進行高質量處理
3. **研究用途**: 啟用評估功能進行性能分析
4. **批量處理**: 調整批處理大小優化效率

## 📝 注意事項

- Whisper過濾功能默認啟用，只處理 `large-v3_` 文件
- 大模型需要較多內存，建議使用量化選項
- 處理時間取決於文件數量和模型大小
- 建議先在小數據集上測試配置
- 所有功能都可以獨立開關，靈活配置 