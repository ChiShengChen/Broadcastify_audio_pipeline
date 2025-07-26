# Unit Test 目錄

此目錄包含所有測試相關的文件，用於驗證 `run_pipeline.sh` 和 `fix_missing_asr_integrated.sh` 的功能。

## 📁 目錄結構

### 測試腳本
- `test_*.py` - Python 測試腳本
- `test_*.sh` - Shell 測試腳本
- `test_*.txt` - 測試結果文件
- `test_*.log` - 測試日誌文件
- `test_*.csv` - 測試數據文件

### 測試數據
- `test_*_data*` - 測試數據集
- `test_*_results` - 測試結果目錄
- `test_*_fix*` - 修復測試相關文件

## 🧪 主要測試文件

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

### 調試工具
- `debug_evaluation.py` - 評估調試工具

## 🚀 使用方法

### 運行所有測試
```bash
cd unit_test
python3 test_error_handling.py
python3 test_enhanced_preprocessor_integration.py
python3 test_pipeline_status.py
```

### 運行特定測試
```bash
# 測試錯誤處理
python3 test_error_handling.py

# 測試預處理器整合
python3 test_enhanced_preprocessor_integration.py

# 測試管道狀態
python3 test_pipeline_status.py
```

## 📊 測試結果

測試結果會保存在對應的結果目錄中：
- `test_error_handling_results/` - 錯誤處理測試結果
- `test_fix_results/` - 修復測試結果

## 🔍 測試覆蓋範圍

1. **錯誤處理測試**
   - 文件不存在錯誤
   - 格式錯誤
   - 編碼錯誤
   - 空數據錯誤

2. **預處理器測試**
   - 基本預處理器
   - 增強預處理器
   - 保守模式 vs 激進模式

3. **管道狀態測試**
   - 成功狀態報告
   - 失敗狀態報告
   - 警告處理

4. **功能整合測試**
   - 完整管道流程
   - 組件間協作
   - 錯誤恢復

## 📝 注意事項

- 測試文件使用臨時目錄，不會影響實際數據
- 測試完成後會自動清理臨時文件
- 某些測試需要特定的依賴庫（如 librosa）
- 測試結果僅供參考，實際環境可能有所不同 