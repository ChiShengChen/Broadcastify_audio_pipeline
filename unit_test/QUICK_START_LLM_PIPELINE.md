# LLM-Enhanced Pipeline 快速開始指南

## 前置需求

### 1. Python環境
```bash
# 安裝Python依賴
pip install -r llm_pipeline_requirements.txt
```

### 2. 系統需求檢查
```bash
# 檢查系統需求
python3 setup_local_models.py --check_system
```

### 3. 下載本地模型

#### 查看可用模型
```bash
python3 setup_local_models.py --list
```

#### 下載模型
```bash
# 下載BioMistral-7B (推薦用於醫學任務)
python3 setup_local_models.py --models BioMistral-7B

# 下載Meditron-7B
python3 setup_local_models.py --models Meditron-7B

# 下載多個模型
python3 setup_local_models.py --models BioMistral-7B Meditron-7B

# 指定下載路徑
python3 setup_local_models.py --models BioMistral-7B --download_path /path/to/models
```

## 快速測試

### 1. 運行基本測試
```bash
# 測試pipeline組件
python3 test_llm_pipeline.py

# 清理測試文件
python3 test_llm_pipeline.py --cleanup
```

### 2. 使用測試數據運行pipeline
```bash
# 運行完整pipeline (使用本地模型)
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir test_llm_pipeline/asr_results \
    --ground_truth test_llm_pipeline/asr_results/test_ground_truth.csv \
    --medical_correction_model BioMistral-7B \
    --page_generation_model Meditron-7B
```

## 實際使用

### 1. 準備ASR結果
首先運行原始ASR pipeline:
```bash
./run_pipeline.sh \
    --input_dir /path/to/audio/files \
    --output_dir /path/to/asr/results
```

### 2. 運行LLM-Enhanced Pipeline
```bash
# 基本使用
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results

# 自定義模型
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --medical_correction_model BioMistral-7B \
    --page_generation_model Meditron-7B

# 包含評估
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --ground_truth /path/to/ground_truth.csv
```

## 常見配置

### 1. 只進行醫學名詞修正
```bash
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --disable_page_generation
```

### 2. 只生成急診Page
```bash
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --disable_medical_correction
```

### 3. 自定義模型路徑
```bash
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --model_path /path/to/custom/models
```

### 4. 調整處理參數
```bash
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --batch_size 1 \
    --device cuda \
    --load_in_8bit
```

## 輸出結構

```
llm_enhanced_results_YYYYMMDD_HHMMSS/
├── corrected_transcripts/          # 醫學名詞修正結果
├── emergency_pages/                # 急診Page生成結果
├── llm_enhanced_evaluation_results.csv  # 評估結果
├── llm_enhanced_pipeline_summary.txt    # 摘要報告
└── error_analysis.log              # 錯誤日誌
```

## 故障排除

### 1. 模型下載問題
```bash
# 檢查模型是否已下載
ls -la ./models/

# 重新下載模型
python3 setup_local_models.py --models BioMistral-7B --force
```

### 2. 內存不足
- 使用量化選項: `--load_in_8bit` 或 `--load_in_4bit`
- 減少批次大小: `--batch_size 1`
- 使用CPU: `--device cpu`

### 3. CUDA問題
- 檢查CUDA安裝: `nvidia-smi`
- 檢查PyTorch CUDA支持: `python -c "import torch; print(torch.cuda.is_available())"`

### 3. 處理速度慢
- 減少batch_size
- 檢查LLM服務器性能
- 考慮使用更快的模型

## 進階配置

### 1. 自定義Prompt
```bash
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --medical_correction_prompt "你是一個醫學專家，請修正以下轉錄中的醫學術語..." \
    --page_generation_prompt "你是一個急診調度員，請根據以下信息生成急診page..."
```

### 2. 使用模板
```bash
# 在Python腳本中添加 --use_template 參數
python3 llm_emergency_page_generator.py \
    --input_dirs /path/to/transcripts \
    --output_dir /path/to/output \
    --model BioMistral-7B \
    --use_template
```

## 性能優化建議

1. **並行處理**: 調整batch_size以平衡速度和資源使用
2. **模型選擇**: 根據任務需求選擇合適的模型
3. **API優化**: 使用本地API以減少網絡延遲
4. **資源管理**: 監控內存和CPU使用情況

## 下一步

1. 查看詳細文檔: `LLM_ENHANCED_PIPELINE_README.md`
2. 自定義prompt以適應特定需求
3. 調整模型參數以優化性能
4. 整合到現有的工作流程中 