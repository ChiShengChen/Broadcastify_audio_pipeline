# EMS Call ASR Pipeline 項目結構

📖 **For English documentation, see [README.md](README.md)**

## 📁 項目概覽

這是一個完整的 ASR（自動語音識別）評估管道項目，主要包含兩個核心腳本：
- `run_pipeline.sh` - 主要的 ASR 評估管道
- `fix_missing_asr_integrated.sh` - 修復缺失 ASR 文件的整合工具

## 🏗️ 目錄結構

```
ems_call/
├── 📁 unit_test/           # 測試文件目錄
├── 📁 tool/               # 工具和輔助文件目錄
├── 📁 asr_models/         # ASR 模型相關文件
├── 📁 data/               # 數據集目錄
├── 📁 vb_ems_anotation/   # 標註數據
├── 📁 long_audio_test_dataset/  # 長音頻測試數據集
├── 📁 pipeline_results_*/ # 管道執行結果
├── 📄 run_pipeline.sh     # 主要管道腳本
├── 📄 evaluate_asr.py     # ASR 評估核心
├── 📄 run_all_asrs.py     # ASR 模型執行
├── 📄 long_audio_splitter.py  # 長音頻分割
├── 📄 merge_split_transcripts.py  # 轉錄合併
├── 📄 vad_pipeline.py     # VAD 處理
├── 📄 enhanced_vad_pipeline.py  # 增強 VAD
└── 📄 README.md           # 項目說明
```

## 🔧 核心腳本

### `run_pipeline.sh`
主要的 ASR 評估管道，功能包括：
- **可選 VAD 預處理**：提取語音片段
- **長音頻分割**：防止 OOM 錯誤
- **ASR 轉錄**：支持多個模型
- **Ground Truth 預處理**：提高匹配準確性
- **評估計算**：WER, MER, WIL 指標
- **錯誤處理**：完整的錯誤檢測和報告
- **狀態報告**：清晰的成功/失敗狀態

### `fix_missing_asr_integrated.sh`
修復缺失 ASR 文件的整合工具：
- **缺失文件分析**：自動檢測缺失的轉錄文件
- **原因分析**：分析缺失的可能原因
- **自動修復**：生成修復腳本
- **詳細報告**：提供完整的分析報告

## 📁 unit_test/ 目錄

包含所有測試相關文件：

### 核心功能測試
- `test_error_handling.py` - 錯誤處理功能測試
- `test_enhanced_preprocessor_integration.py` - 增強預處理器整合測試
- `test_pipeline_status.py` - 管道狀態報告測試
- `test_preprocessing_impact.py` - 預處理影響測試

### 組件測試
- `test_asr_long_audio.py` - 長音頻 ASR 測試
- `test_long_audio_split.py` - 長音頻分割測試
- `test_missing_files.py` - 缺失文件測試
- `test_model_counters.py` - 模型計數器測試
- `test_vad_pipeline.py` - VAD 管道測試

### 測試數據
- `test_*_data*` - 測試數據集
- `test_*_results` - 測試結果目錄
- `test_*_fix*` - 修復測試相關文件

## 📁 tool/ 目錄

包含所有工具和輔助文件：

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

### 文檔和指南
- `ERROR_HANDLING_GUIDE.md` - 錯誤處理指南
- `PIPELINE_STATUS_GUIDE.md` - 管道狀態指南
- `GROUND_TRUTH_PREPROCESSING_GUIDE.md` - 預處理指南
- `ENHANCED_PREPROCESSOR_USAGE_GUIDE.md` - 增強預處理器指南

## 🚀 快速開始

### 1. 基本使用
```bash
# 運行主要管道
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv

# 修復缺失文件
./fix_missing_asr_integrated.sh \
    --pipeline_output_dir /path/to/pipeline_results \
    --fix_output_dir /path/to/fix_results \
    --ground_truth_file /path/to/ground_truth.csv
```

### 2. 高級功能
```bash
# 使用 VAD 和長音頻分割
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-vad \
    --use-long-audio-split \
    --preprocess-ground-truth \
    --use-enhanced-preprocessor
```

### 3. 運行測試
```bash
# 運行錯誤處理測試
cd unit_test
python3 test_error_handling.py

# 運行預處理器測試
python3 test_enhanced_preprocessor_integration.py
```

## 🔍 主要功能

### ASR 評估管道
- **多模型支持**：Whisper Large-v3, Wav2Vec2, Parakeet, Canary-1B
- **VAD 預處理**：可選的語音活動檢測
- **長音頻處理**：自動分割長音頻文件
- **Ground Truth 預處理**：智能文本正規化
- **完整評估**：WER, MER, WIL 指標計算
- **錯誤處理**：自動錯誤檢測和報告
- **狀態報告**：清晰的成功/失敗狀態

### 修復工具
- **自動檢測**：識別缺失的轉錄文件
- **原因分析**：分析缺失的可能原因
- **智能修復**：生成針對性的修復腳本
- **結果整合**：將修復結果整合到原始結果中

### 預處理功能
- **基本預處理器**：簡單的文本正規化
- **智能預處理器**：自適應文本預處理
- **增強預處理器**：全面的文本正規化
- **多種模式**：保守模式和激進模式

## 📊 輸出結果

### 管道輸出
- `asr_evaluation_results.csv` - 評估結果
- `model_file_analysis.txt` - 模型文件分析
- `pipeline_summary.txt` - 管道摘要
- `error_analysis.log` - 錯誤分析日誌

### 修復輸出
- `missing_analysis.json` - 缺失文件分析
- `rerun_missing_asr.sh` - 修復腳本
- `missing_files_report.txt` - 詳細報告

## 🔧 依賴要求

### Python 依賴
```bash
pip install pandas jiwer torch transformers torchaudio nemo_toolkit[asr] openai-whisper tqdm scipy numpy pathlib2 soundfile pydub librosa
```

### 系統依賴
- Python 3.7+
- FFmpeg（用於音頻處理）
- 足夠的磁盤空間和內存

## 📝 注意事項

1. **文件組織**：測試文件在 `unit_test/`，工具文件在 `tool/`
2. **錯誤處理**：管道包含完整的錯誤檢測和報告
3. **狀態報告**：執行完成後會顯示清晰的成功/失敗狀態
4. **備份建議**：修復工具會修改文件，建議先備份
5. **資源要求**：長音頻處理需要較多內存和磁盤空間

## 🤝 貢獻指南

1. 新功能請添加相應的測試
2. 工具文件請放在 `tool/` 目錄
3. 測試文件請放在 `unit_test/` 目錄
4. 更新相關文檔和指南 