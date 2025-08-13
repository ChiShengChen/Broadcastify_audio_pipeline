# LLM-Enhanced ASR Pipeline

這個pipeline擴展了原有的ASR pipeline，加入了LLM功能來修正醫學名詞並生成急診page。

## 功能概述

### 1. 醫學名詞修正 (Medical Term Correction)
- 使用LLM模型修正ASR結果中的醫學名詞、藥物名稱、解剖術語等
- 支援多種LLM模型：gpt-oss-20b, BioMistral-7B, Meditron-7B, Llama-3-8B-UltraMedica
- 可自定義prompt來控制修正的範圍和風格

### 2. 急診Page生成 (Emergency Page Generation)
- 基於修正後的醫學轉錄生成結構化的急診page
- 包含患者狀況摘要、位置詳情、所需醫療資源、優先級等
- 支援使用模板來確保一致的格式

### 3. 評估功能 (Evaluation)
- 可選擇性地評估修正後的結果與ground truth的比較
- 生成新的WER等驗證指標

## 支援的LLM模型

| 模型 | 類型 | API端點 | 適用場景 |
|------|------|---------|----------|
| gpt-oss-20b | OpenAI兼容 | OpenAI API | 醫學名詞修正、Page生成 |
| BioMistral-7B | 本地模型 | Local API | 醫學名詞修正、Page生成 |
| Meditron-7B | 本地模型 | Local API | 醫學名詞修正、Page生成 |
| Llama-3-8B-UltraMedica | 本地模型 | Local API | 醫學名詞修正、Page生成 |

## 安裝需求

### Python依賴
```bash
pip install requests tqdm concurrent-futures
```

### LLM模型設置
1. **本地模型** (BioMistral, Meditron, Llama):
   - 需要運行本地API服務器 (例如使用vLLM, Ollama等)
   - 默認端點: `http://localhost:8000/v1`

2. **OpenAI兼容模型** (GPT-OSS):
   - 需要運行OpenAI兼容的API服務器
   - 默認端點: `http://localhost:8000/v1`

## 使用方法

### 基本用法

```bash
# 使用默認設置
./run_llm_enhanced_pipeline.sh --asr_results_dir /path/to/asr/results

# 指定輸出目錄
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --output_dir /path/to/output
```

### 自定義模型選擇

```bash
# 使用BioMistral進行醫學名詞修正，Meditron進行Page生成
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --medical_correction_model BioMistral-7B \
    --page_generation_model Meditron-7B
```

### 功能開關

```bash
# 只進行醫學名詞修正，不生成Page
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --disable_page_generation

# 只生成Page，不進行醫學名詞修正
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --disable_medical_correction

# 包含評估
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --ground_truth /path/to/ground_truth.csv
```

### 自定義API端點

```bash
# 使用自定義API端點
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --local_model_endpoint http://192.168.1.100:8000/v1 \
    --openai_api_base http://192.168.1.100:8000/v1
```

### 自定義Prompt

```bash
# 使用自定義醫學名詞修正prompt
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --medical_correction_prompt "你是一個醫學轉錄專家，請修正以下轉錄中的醫學術語..."

# 使用自定義Page生成prompt
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --page_generation_prompt "你是一個急診調度員，請根據以下醫學轉錄生成急診page..."
```

## 輸出結構

```
llm_enhanced_results_YYYYMMDD_HHMMSS/
├── corrected_transcripts/          # 醫學名詞修正後的轉錄
│   ├── file1.txt
│   ├── file2.txt
│   └── medical_correction_summary.json
├── emergency_pages/                # 生成的急診page
│   ├── file1_emergency_page.txt
│   ├── file2_emergency_page.txt
│   ├── emergency_page_template.txt
│   └── emergency_page_generation_summary.json
├── llm_enhanced_evaluation_results.csv  # 評估結果
├── llm_enhanced_pipeline_summary.txt    # Pipeline摘要
└── error_analysis.log              # 錯誤分析日誌
```

## 配置選項

### 命令行參數

#### 必需參數
- `--asr_results_dir DIR`: ASR結果目錄路徑

#### 可選參數
- `--output_dir DIR`: 輸出目錄 (默認自動生成)
- `--ground_truth FILE`: Ground truth CSV文件
- `--medical_correction_model MODEL`: 醫學名詞修正模型
- `--page_generation_model MODEL`: Page生成模型
- `--enable_medical_correction`: 啟用醫學名詞修正 (默認)
- `--disable_medical_correction`: 禁用醫學名詞修正
- `--enable_page_generation`: 啟用Page生成 (默認)
- `--disable_page_generation`: 禁用Page生成
- `--enable_evaluation`: 啟用評估 (默認)
- `--disable_evaluation`: 禁用評估
- `--local_model_endpoint URL`: 本地模型API端點
- `--openai_api_base URL`: OpenAI兼容API端點
- `--batch_size INT`: 並行處理文件數量 (默認: 5)
- `--medical_correction_prompt TEXT`: 醫學名詞修正prompt
- `--page_generation_prompt TEXT`: Page生成prompt

## 工作流程

1. **ASR結果讀取**: 從指定的ASR結果目錄讀取轉錄文件
2. **醫學名詞修正** (可選): 使用LLM修正醫學術語
3. **急診Page生成** (可選): 基於轉錄生成結構化急診page
4. **評估** (可選): 與ground truth比較並計算指標
5. **結果輸出**: 保存所有處理結果和摘要

## 錯誤處理

- 自動重試機制 (最多3次)
- 指數退避策略
- 詳細的錯誤日誌
- 部分失敗時繼續處理其他文件

## 性能優化

- 並行處理多個文件
- 可配置的批次大小
- 請求超時設置
- 連接池管理

## 故障排除

### 常見問題

1. **API連接失敗**
   - 檢查LLM服務器是否運行
   - 驗證API端點URL
   - 檢查網絡連接

2. **模型不可用**
   - 確認模型名稱正確
   - 檢查模型是否已載入到服務器

3. **處理速度慢**
   - 減少批次大小
   - 增加並行處理數量
   - 檢查LLM服務器性能

4. **內存不足**
   - 減少批次大小
   - 關閉不必要的功能
   - 增加系統內存

### 日誌文件

- `llm_medical_correction.log`: 醫學名詞修正日誌
- `llm_emergency_page_generator.log`: Page生成日誌
- `error_analysis.log`: 錯誤分析日誌

## 示例

### 完整工作流程示例

```bash
# 1. 首先運行原始ASR pipeline
./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/asr/results

# 2. 然後運行LLM-enhanced pipeline
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --ground_truth /path/to/ground_truth.csv \
    --medical_correction_model BioMistral-7B \
    --page_generation_model Meditron-7B
```

### 快速測試示例

```bash
# 只測試醫學名詞修正
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --disable_page_generation \
    --disable_evaluation

# 只測試Page生成
./run_llm_enhanced_pipeline.sh \
    --asr_results_dir /path/to/asr/results \
    --disable_medical_correction \
    --disable_evaluation
```

## 注意事項

1. **API依賴**: 確保LLM API服務器正在運行
2. **模型可用性**: 確認選擇的模型已在服務器中載入
3. **資源需求**: LLM處理可能需要較多計算資源
4. **網絡穩定性**: 本地API需要穩定的網絡連接
5. **文件格式**: 確保ASR結果是正確的文本格式

## 更新日誌

- v1.0: 初始版本，支援基本的醫學名詞修正和Page生成
- 支援多種LLM模型
- 並行處理和錯誤處理
- 完整的評估功能 