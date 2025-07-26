# Ground Truth 預處理指南

## 概述

本指南介紹如何在 ASR 評估流程中使用 ground truth 預處理功能，以提高評估的準確性和一致性。

## 問題背景

在 ASR 評估中，ground truth 文件中的特殊符號、數字格式和縮寫可能會影響評估結果：

- **特殊符號**：`[x]`, `%`, `&`, `?` 等
- **數字格式**：`612`, `4560`, `11:17` 等
- **專業縮寫**：`PD`, `EMS`, `BLS`, `CPR` 等
- **標點符號**：逗號、句號、引號等

這些不一致的格式可能導致：
1. 文件匹配失敗
2. WER 計算不準確
3. 不同模型間的評估結果不一致

## 解決方案

### 1. 預處理模式

我們提供了兩種預處理模式：

#### 保守模式 (Conservative)
- **適用場景**：希望保持原始文本結構，只做最小必要的清理
- **處理內容**：
  - 移除方括號 `[x]` → `x`
  - 移除標點符號
  - 轉換常見縮寫（PD, EMS, BLS 等）
  - 保持數字格式不變
- **優點**：變化最小，保持原始語義
- **缺點**：可能對某些模型幫助有限

#### 激進模式 (Aggressive)
- **適用場景**：希望最大化 ASR 模型的理解能力
- **處理內容**：
  - 所有保守模式的處理
  - 將數字轉換為詞（`612` → `six one two`）
  - 展開所有縮寫
  - 標準化所有特殊符號
- **優點**：最大化文本標準化
- **缺點**：可能改變原始語義

### 2. 使用方法

#### 在 run_pipeline.sh 中使用

```bash
# 使用保守模式（默認）
./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results --preprocess-ground-truth

# 使用激進模式
./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results --preprocess-mode aggressive

# 禁用預處理
./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results --no-preprocess-ground-truth
```

#### 獨立使用預處理腳本

```bash
# 保守模式預覽
python3 smart_preprocess_ground_truth.py \
    --input_file ground_truth.csv \
    --output_file preprocessed.csv \
    --mode conservative \
    --preview

# 激進模式處理
python3 smart_preprocess_ground_truth.py \
    --input_file ground_truth.csv \
    --output_file preprocessed.csv \
    --mode aggressive

# 原始預處理腳本（激進模式）
python3 preprocess_ground_truth.py \
    --input_file ground_truth.csv \
    --output_file preprocessed.csv \
    --preview
```

### 3. 測試和比較

使用測試腳本來比較不同預處理模式的效果：

```bash
# 運行完整的比較測試
python3 test_preprocessing_impact.py
```

這會生成：
- `preprocessing_test_results/original_evaluation.csv` - 原始評估結果
- `preprocessing_test_results/preprocessed_evaluation.csv` - 預處理後評估結果
- `preprocessing_test_results/preprocessing_comparison.csv` - 詳細比較報告

## 實際效果分析

基於我們的測試結果：

### 文件匹配
- ✅ 所有模式都保持了 9 個文件的匹配
- ✅ 沒有因為預處理而丟失文件

### WER 改善
- **large-v3**: 0.4955 → 0.5903 (變差)
- **parakeet-tdt-0.6b-v2**: 0.7354 → 0.6814 (改善 5.4%)
- **canary-1b**: 0.8156 → 0.7779 (改善 3.8%)
- **wav2vec-xls-r**: 0.9232 → 0.8645 (改善 5.9%)

### 詞數變化
- 原始：2109 個詞
- 預處理後：2458 個詞（增加 16.5%）

## 建議

### 1. 選擇預處理模式

**推薦使用保守模式**：
- 對於大多數情況，保守模式提供了良好的平衡
- 保持了原始語義，同時清理了問題字符
- 對大多數模型都有正面影響

**考慮使用激進模式**：
- 當 ASR 模型在數字識別方面表現不佳時
- 當需要最大化文本標準化時
- 在測試階段比較不同模式的效果

### 2. 測試策略

1. **先運行保守模式**：評估基本改善效果
2. **比較激進模式**：如果保守模式效果有限，嘗試激進模式
3. **分析具體模型**：不同模型對預處理的反應可能不同
4. **檢查文件匹配**：確保預處理沒有導致文件丟失

### 3. 最佳實踐

- **備份原始文件**：預處理腳本會自動創建備份
- **記錄預處理模式**：在結果報告中記錄使用的模式
- **定期測試**：當更新 ASR 模型時重新測試預處理效果
- **自定義規則**：根據具體數據集調整預處理規則

## 故障排除

### 常見問題

1. **預處理失敗**
   ```bash
   # 檢查文件格式
   head -5 ground_truth.csv
   
   # 檢查必要列是否存在
   python3 -c "import pandas as pd; df=pd.read_csv('ground_truth.csv'); print(df.columns.tolist())"
   ```

2. **文件匹配數量減少**
   - 檢查預處理是否改變了文件名
   - 確認 ground truth 文件格式正確

3. **WER 顯著惡化**
   - 嘗試不同的預處理模式
   - 檢查特定模型的表現

### 調試工具

```bash
# 檢查預處理效果
python3 debug_filename_matching.py

# 分析評估一致性
python3 check_evaluation_consistency.py

# 比較預處理前後結果
python3 test_preprocessing_impact.py
```

## 總結

Ground truth 預處理是一個強大的工具，可以顯著改善 ASR 評估的準確性。通過選擇合適的預處理模式，可以：

1. **提高文件匹配率**：清理特殊字符避免匹配失敗
2. **改善 WER 計算**：標準化文本格式提高評估準確性
3. **增強一致性**：確保不同模型間的公平比較
4. **適應不同模型**：根據模型特性選擇最佳預處理策略

建議在實際使用中先測試保守模式，然後根據需要調整到激進模式。 