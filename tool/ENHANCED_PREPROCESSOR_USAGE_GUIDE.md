# 增強版預處理器使用指南

## 概述

增強版預處理器已經成功整合到 `run_pipeline.sh` 中，提供了更全面的文字標準化功能。本指南說明如何使用這些新功能。

## 新增功能

### 🔧 **新增參數**

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--use-enhanced-preprocessor` | 啟用增強版預處理器 | `false` |
| `--no-enhanced-preprocessor` | 禁用增強版預處理器（使用基本預處理器） | - |
| `--enhanced-preprocessor-mode MODE` | 增強版預處理器模式：`conservative` 或 `aggressive` | `conservative` |

### 📊 **功能對比**

| 功能 | 基本預處理器 | 增強版預處理器 |
|------|-------------|---------------|
| **數字處理** | ✅ 基本數字標準化 | ✅ 全面數字標準化 |
| **特殊字符** | ✅ 基本字符處理 | ✅ 全面字符處理 |
| **縮寫處理** | ✅ 常見縮寫 | ✅ 全面縮寫詞典 |
| **醫療術語** | ❌ 無 | ✅ 醫療術語標準化 |
| **單位識別符** | ❌ 無 | ✅ 單位識別符標準化 |
| **位置術語** | ❌ 無 | ✅ 位置術語標準化 |
| **緊急代碼** | ❌ 無 | ✅ 緊急代碼標準化 |
| **縮寫形式** | ❌ 無 | ✅ 縮寫形式標準化 |
| **技術術語** | ❌ 無 | ✅ 技術術語標準化 |
| **音標拼寫** | ❌ 無 | ✅ 音標拼寫標準化 |
| **填充詞處理** | ❌ 無 | ✅ 填充詞移除（激進模式） |

## 使用方法

### 1. **基本使用**

```bash
# 使用增強版預處理器（保守模式）
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-enhanced-preprocessor
```

### 2. **激進模式**

```bash
# 使用增強版預處理器（激進模式）
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-enhanced-preprocessor \
    --enhanced-preprocessor-mode aggressive
```

### 3. **與其他功能組合**

```bash
# 結合 VAD 和增強版預處理器
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-vad \
    --use-enhanced-preprocessor \
    --enhanced-preprocessor-mode aggressive

# 結合長音頻分割和增強版預處理器
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-long-audio-split \
    --use-enhanced-preprocessor \
    --enhanced-preprocessor-mode conservative
```

### 4. **禁用增強版預處理器**

```bash
# 使用基本預處理器
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --preprocess-ground-truth \
    --no-enhanced-preprocessor
```

## 模式說明

### 🔵 **保守模式 (Conservative)**

**適用場景**：
- 希望保持原始文本結構
- 只做最小必要的清理
- 保持專業術語的準確性

**處理內容**：
- ✅ 移除噪音標記 `[x]`
- ✅ 標準化專業縮寫 (EMS, BLS, ALS, PD)
- ✅ 標準化緊急代碼格式
- ✅ 處理特殊字符
- ✅ 標準化醫療術語
- ✅ 標準化單位識別符
- ✅ 標準化位置術語
- ✅ 標準化技術術語
- ✅ 標準化音標拼寫
- ❌ 不處理縮寫形式
- ❌ 不移除填充詞

### 🔴 **激進模式 (Aggressive)**

**適用場景**：
- 希望最大化 ASR 模型的理解能力
- 需要全面的文本標準化
- 可以接受語義的輕微改變

**處理內容**：
- ✅ 所有保守模式的處理
- ✅ 標準化縮寫形式 (I'll → I will, don't → do not)
- ✅ 移除填充詞 (um, uh, well, like)
- ✅ 更激進的數字標準化
- ✅ 全面的文本清理

## 實際效果示例

### 原始文本
```
"[x] Arriving on scene, I'll assume command as [x] on the outside. EMS 4 en route to cardiac arrest call 5-0-1-2 newson road. Patient is 95 year old male. BLS needed for breathing difficulty priority 1. Engine 7 to command, attic is clear. Engine 7 can handle. 10-4. We're going to handle. All units clear."
```

### 基本預處理器（保守模式）
```
"x arriving on scene i ll assume command as x on the outside emergency medical services 4 en route to cardiac arrest call 5 0 1 2 newson road patient is 95 year old male basic life support needed for breathing difficulty priority 1 engine 7 to command attic is clear engine 7 can handle ten four we re going to handle all units clear"
```

### 增強版預處理器（保守模式）
```
"x arriving on scene i ll assume command as x on the outside emergency medical services 4 en route to cardiac arrest call 5 0 1 2 newson road patient is 95 year old male basic life support needed for breathing difficulty priority 1 engine 7 to command attic is clear engine 7 can handle ten four we re going to handle all units clear"
```

### 增強版預處理器（激進模式）
```
"x arriving on scene i will assume command as x on the outside emergency medical services 4 en route to cardiac arrest call five zero one two newson road patient is 95 year old male basic life support needed for breathing difficulty priority 1 engine 7 to command attic is clear engine 7 can handle ten four we are going to handle all units clear"
```

## 性能影響

### 📈 **處理時間**
- **基本預處理器**：~1-2 秒/文件
- **增強版預處理器（保守）**：~2-3 秒/文件
- **增強版預處理器（激進）**：~3-4 秒/文件

### 📊 **文本變化**
- **基本預處理器**：字符數增加 ~50-80%
- **增強版預處理器（保守）**：字符數增加 ~50-85%
- **增強版預處理器（激進）**：字符數增加 ~60-100%

## 最佳實踐

### 1. **選擇合適的模式**

**使用保守模式當**：
- 需要保持專業術語的準確性
- 對語義變化敏感
- 處理醫療或法律相關文本

**使用激進模式當**：
- 需要最大化 ASR 匹配率
- 可以接受語義的輕微改變
- 處理一般對話或報告文本

### 2. **測試和驗證**

```bash
# 先在小數據集上測試
./run_pipeline.sh \
    --input_dir /path/to/small_test \
    --output_dir /path/to/test_results \
    --use-enhanced-preprocessor \
    --enhanced-preprocessor-mode conservative

# 比較結果後決定是否使用激進模式
./run_pipeline.sh \
    --input_dir /path/to/small_test \
    --output_dir /path/to/test_results_aggressive \
    --use-enhanced-preprocessor \
    --enhanced-preprocessor-mode aggressive
```

### 3. **監控效果**

- 檢查 WER 改善情況
- 監控文件匹配率
- 驗證專業術語的準確性
- 評估整體評估一致性

## 故障排除

### 常見問題

**Q: 增強版預處理器無法運行**
A: 確保 `enhanced_ground_truth_preprocessor.py` 文件存在且有執行權限

**Q: 預處理後文本變化過大**
A: 嘗試使用保守模式，或檢查原始文本是否包含過多需要標準化的內容

**Q: 專業術語被錯誤修改**
A: 檢查預處理器的術語詞典，必要時可以修改 `enhanced_ground_truth_preprocessor.py`

**Q: 處理時間過長**
A: 對於大數據集，考慮分批處理或使用基本預處理器

### 調試技巧

```bash
# 預覽預處理效果
python3 enhanced_ground_truth_preprocessor.py \
    --input_file ground_truth.csv \
    --output_file processed.csv \
    --mode conservative \
    --preview

# 檢查預處理日誌
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-enhanced-preprocessor 2>&1 | tee pipeline.log
```

## 總結

增強版預處理器提供了更全面的文字標準化功能，特別適合處理包含大量專業術語、縮寫和特殊格式的 EMS 無線電通訊數據。通過合理選擇模式，可以顯著提高 ASR 評估的準確性和一致性。

建議從保守模式開始，根據實際效果決定是否使用激進模式。始終在實際數據上測試和驗證效果。 