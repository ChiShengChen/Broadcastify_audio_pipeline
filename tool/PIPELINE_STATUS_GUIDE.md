# 管道狀態報告功能指南

## 概述

`run_pipeline.sh` 現在在執行完成後會自動檢查整個管道的執行狀態，並在最後顯示清晰的成功或失敗信息。

## 新增功能

### ✅ **成功狀態報告**

當管道成功完成時，會顯示：

```
=== Pipeline Completed Successfully ===

Results structure:
  /path/to/output/long_audio_segments/   # Long audio split segments
  /path/to/output/vad_segments/          # VAD extracted speech segments
  /path/to/output/asr_transcripts/       # ASR transcription results
  /path/to/output/merged_transcripts/    # Merged transcripts for evaluation
  /path/to/output/asr_evaluation_results.csv         # Evaluation metrics
  /path/to/output/model_file_analysis.txt # Model file processing analysis
  /path/to/output/pipeline_summary.txt        # Detailed summary

Check the summary file for detailed results: /path/to/output/pipeline_summary.txt

⚠️  Note: 2 warnings were detected during processing.
   Check /path/to/output/error_analysis.log for details.
```

### ❌ **失敗狀態報告**

當管道遇到問題時，會顯示：

```
=== Pipeline Completed with Errors ===

❌ Pipeline encountered issues during execution.

Error Summary:
  - Errors detected: 3
  - Warnings detected: 2

Troubleshooting:
  1. Check the error log: /path/to/output/error_analysis.log
  2. Review the pipeline summary: /path/to/output/pipeline_summary.txt
  3. Verify input files and configuration
  4. Check system resources (disk space, memory)

Available results (may be incomplete):
  - Output directory: /path/to/output
  - Pipeline summary: /path/to/output/pipeline_summary.txt
  - Error analysis: /path/to/output/error_analysis.log
```

## 狀態判斷邏輯

### 🔍 **成功條件**

管道被認為成功完成，必須滿足以下所有條件：

1. **無錯誤記錄**：`error_analysis.log` 中沒有 `[ERROR]` 條目
2. **評估文件存在**：`asr_evaluation_results.csv` 文件存在
3. **轉錄目錄存在**：`merged_transcripts` 目錄存在

### ⚠️ **警告處理**

- **警告不影響成功狀態**：只有 `[WARNING]` 條目不會導致管道被標記為失敗
- **警告會顯示**：成功時會顯示警告數量，提醒用戶檢查
- **警告詳情**：可以查看 `error_analysis.log` 了解具體警告內容

### ❌ **失敗條件**

管道被標記為失敗，如果滿足以下任一條件：

1. **有錯誤記錄**：`error_analysis.log` 中有 `[ERROR]` 條目
2. **缺少評估文件**：`asr_evaluation_results.csv` 文件不存在
3. **缺少轉錄目錄**：`merged_transcripts` 目錄不存在

## 錯誤類型分類

### 🔴 **嚴重錯誤（導致失敗）**

| 錯誤類型 | 描述 | 影響 |
|----------|------|------|
| `FILE_NOT_FOUND` | 文件或目錄不存在 | 阻止處理繼續 |
| `INVALID_FORMAT` | 文件格式不正確 | 數據無法讀取 |
| `LOAD_ERROR` | 數據加載失敗 | 無法進行分析 |
| `NO_MODELS_FOUND` | 未找到模型文件 | 無法進行評估 |

### 🟡 **警告（不影響成功）**

| 警告類型 | 描述 | 影響 |
|----------|------|------|
| `EMPTY_DATA` | 數據為空或無效 | 可能影響結果質量 |
| `MISSING_GROUND_TRUTH` | 缺少對應的 ground truth | 部分文件無法評估 |
| `SUSPICIOUS_CONTENT` | 內容可疑或異常 | 需要人工檢查 |
| `SHORT_CONTENT` | 內容過短 | 可能影響準確性 |

## 使用示例

### 1. **成功執行示例**

```bash
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-enhanced-preprocessor \
    --enhanced-preprocessor-mode conservative
```

**輸出**：
```
=== Pipeline Completed Successfully ===

Results structure:
  /path/to/results/long_audio_segments/   # Long audio split segments
  /path/to/results/vad_segments/          # VAD extracted speech segments
  /path/to/results/asr_transcripts/       # ASR transcription results
  /path/to/results/merged_transcripts/    # Merged transcripts for evaluation
  /path/to/results/asr_evaluation_results.csv         # Evaluation metrics
  /path/to/results/model_file_analysis.txt # Model file processing analysis
  /path/to/results/pipeline_summary.txt        # Detailed summary

Check the summary file for detailed results: /path/to/results/pipeline_summary.txt
```

### 2. **失敗執行示例**

當遇到錯誤時：

```bash
./run_pipeline.sh \
    --input_dir /path/to/nonexistent \
    --output_dir /path/to/results
```

**輸出**：
```
=== Pipeline Completed with Errors ===

❌ Pipeline encountered issues during execution.

Error Summary:
  - Errors detected: 2
  - Warnings detected: 1

Troubleshooting:
  1. Check the error log: /path/to/results/error_analysis.log
  2. Review the pipeline summary: /path/to/results/pipeline_summary.txt
  3. Verify input files and configuration
  4. Check system resources (disk space, memory)

Available results (may be incomplete):
  - Output directory: /path/to/results
  - Pipeline summary: /path/to/results/pipeline_summary.txt
  - Error analysis: /path/to/results/error_analysis.log
```

## 故障排除指南

### 1. **檢查錯誤日誌**

```bash
# 查看錯誤數量
grep -c "\[ERROR\]" /path/to/results/error_analysis.log

# 查看警告數量
grep -c "\[WARNING\]" /path/to/results/error_analysis.log

# 查看最新錯誤
tail -20 /path/to/results/error_analysis.log
```

### 2. **檢查關鍵文件**

```bash
# 檢查評估結果
ls -la /path/to/results/asr_evaluation_results.csv

# 檢查轉錄目錄
ls -la /path/to/results/merged_transcripts/

# 檢查摘要文件
cat /path/to/results/pipeline_summary.txt
```

### 3. **常見問題解決**

#### 問題：缺少評估文件
**原因**：ASR 處理失敗或評估步驟出錯
**解決方案**：
- 檢查 ASR 依賴是否正確安裝
- 確認輸入音頻文件格式正確
- 檢查磁盤空間是否充足

#### 問題：缺少轉錄目錄
**原因**：ASR 處理失敗或文件組織出錯
**解決方案**：
- 檢查 ASR 處理日誌
- 確認音頻文件可讀
- 檢查文件權限

#### 問題：錯誤記錄過多
**原因**：輸入數據質量問題或配置錯誤
**解決方案**：
- 檢查 ground truth 文件格式
- 確認音頻文件完整性
- 調整預處理參數

## 最佳實踐

### 1. **運行前檢查**

```bash
# 檢查輸入文件
ls -la /path/to/audio/
ls -la /path/to/ground_truth.csv

# 檢查系統資源
df -h
free -h

# 檢查依賴
python3 -c "import transformers, torch, torchaudio; print('Dependencies OK')"
```

### 2. **運行後驗證**

```bash
# 檢查狀態
tail -10 /path/to/results/pipeline_summary.txt

# 檢查結果文件
ls -la /path/to/results/

# 檢查錯誤日誌
if [ -f "/path/to/results/error_analysis.log" ]; then
    echo "Errors: $(grep -c '\[ERROR\]' /path/to/results/error_analysis.log)"
    echo "Warnings: $(grep -c '\[WARNING\]' /path/to/results/error_analysis.log)"
fi
```

### 3. **監控和維護**

- 定期檢查磁盤空間
- 監控系統資源使用
- 保持依賴庫更新
- 備份重要結果文件

## 總結

新的狀態報告功能提供了：

✅ **清晰的執行狀態**：一目了然地知道管道是否成功  
✅ **詳細的錯誤信息**：幫助快速定位問題  
✅ **實用的故障排除**：提供具體的解決步驟  
✅ **完整的結果概覽**：顯示所有生成的文件和目錄  
✅ **警告提醒**：即使成功也會提醒潛在問題  

這大大提高了管道的可用性和可維護性，讓用戶能夠快速了解執行結果並採取相應的行動。 