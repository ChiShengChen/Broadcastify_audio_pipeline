# Tool 目錄

此目錄包含所有工具和輔助文件，用於支持 `run_pipeline.sh` 和 `fix_missing_asr_integrated.sh` 的運行。

## 📁 目錄結構

### 🔍 分析工具
- `analyze_*.py` - 各種分析腳本
- `check_*.py` - 檢查和驗證工具
- `debug_*.py` - 調試工具

### 🔧 預處理工具
- `*preprocess*.py` - 預處理相關腳本
- `*preprocessing*` - 預處理相關文件
- `enhanced_ground_truth_preprocessor.py` - 增強預處理器

### 🛠️ 修復工具
- `fix_missing_asr*.sh` - 修復缺失 ASR 文件的腳本
- `fix_missing_asr_integrated.sh` - 整合修復腳本

### 📚 文檔和指南
- `*.md` - Markdown 文檔和指南
- `*_GUIDE.md` - 使用指南
- `*_README.md` - 說明文檔

### 📊 分析結果
- `*analysis*.txt` - 分析結果文本文件
- `*analysis*.csv` - 分析結果 CSV 文件

## 🛠️ 主要工具文件

### 分析工具
- `analyze_asr_number_processing.py` - ASR 數字處理分析
- `analyze_evaluation_issue.py` - 評估問題分析
- `analyze_model_files_enhanced.py` - 增強模型文件分析
- `analyze_model_files.py` - 模型文件分析

### 預處理工具
- `smart_preprocess_ground_truth.py` - 智能預處理器
- `enhanced_ground_truth_preprocessor.py` - 增強預處理器
- `preprocess_ground_truth.py` - 基本預處理器

### 修復工具
- `fix_missing_asr_integrated.sh` - 整合修復腳本
- `fix_missing_asr_correct.sh` - 修正版修復腳本
- `fix_missing_asr_files.sh` - 修復缺失文件腳本

### 文檔
- `ERROR_HANDLING_GUIDE.md` - 錯誤處理指南
- `PIPELINE_STATUS_GUIDE.md` - 管道狀態指南
- `GROUND_TRUTH_PREPROCESSING_GUIDE.md` - 預處理指南
- `ENHANCED_PREPROCESSOR_USAGE_GUIDE.md` - 增強預處理器指南

## 🚀 使用方法

### 分析工具
```bash
cd tool

# ASR 數字處理分析
python3 analyze_asr_number_processing.py

# 模型文件分析
python3 analyze_model_files_enhanced.py --transcript_dir /path/to/transcripts --ground_truth_file /path/to/gt.csv

# 評估問題分析
python3 analyze_evaluation_issue.py
```

### 預處理工具
```bash
# 智能預處理
python3 smart_preprocess_ground_truth.py --input_file gt.csv --output_file processed_gt.csv --mode conservative

# 增強預處理
python3 enhanced_ground_truth_preprocessor.py --input_file gt.csv --output_file enhanced_gt.csv --mode aggressive
```

### 修復工具
```bash
# 修復缺失文件
./fix_missing_asr_integrated.sh \
    --pipeline_output_dir /path/to/pipeline_results \
    --fix_output_dir /path/to/fix_results \
    --ground_truth_file /path/to/gt.csv
```

## 📊 分析結果

### 數字處理分析
- `asr_number_processing_analysis.txt` - 詳細分析結果
- `asr_number_processing_comparison.csv` - 比較數據

### 預處理分析
- `comprehensive_preprocessing_analysis.txt` - 綜合預處理分析

## 📚 文檔說明

### 錯誤處理指南
詳細說明錯誤處理功能，包括：
- 自動錯誤檢測
- 詳細日誌記錄
- 錯誤分類和處理
- 故障排除指南

### 管道狀態指南
說明管道狀態報告功能：
- 成功狀態報告
- 失敗狀態報告
- 狀態判斷邏輯
- 故障排除步驟

### 預處理指南
介紹預處理功能：
- 基本預處理器
- 增強預處理器
- 保守模式 vs 激進模式
- 使用示例

## 🔧 工具分類

### 核心分析工具
- **模型文件分析**：分析 ASR 模型輸出文件
- **評估問題分析**：診斷評估過程中的問題
- **數字處理分析**：分析 ASR 模型的數字轉換行為

### 預處理工具
- **基本預處理器**：簡單的文本正規化
- **智能預處理器**：自適應文本預處理
- **增強預處理器**：全面的文本正規化

### 修復工具
- **缺失文件修復**：自動修復缺失的 ASR 文件
- **結果整合**：將修復結果整合到原始結果中

### 輔助工具
- **音頻分析**：分析音頻文件屬性
- **數據統計**：統計數據集信息
- **格式轉換**：各種格式轉換工具

## 📝 注意事項

- 工具文件需要特定的 Python 依賴
- 某些工具需要管理員權限
- 分析結果僅供參考
- 修復工具會修改原始文件，請先備份
