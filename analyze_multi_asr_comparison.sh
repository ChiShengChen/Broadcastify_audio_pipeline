#!/bin/bash

# 詳細分析多ASR比較的原理、行為和輸出
echo "=== 多ASR比較詳細分析 ==="
echo ""

# 1. 比較原理
echo "## 1. 比較原理"
echo "多ASR比較使用LLM（大語言模型）來智能地比較和合併兩個不同ASR模型的轉錄結果："
echo "  - Canary ASR: 專注於語音識別準確性"
echo "  - Whisper ASR: 專注於上下文理解"
echo "  - LLM比較: 結合兩者優勢，生成最佳版本"
echo ""

# 2. 比較行為
echo "## 2. 比較行為"
echo "比較過程包括以下步驟："
echo "  a) 文件配對: 自動匹配相同音頻的Canary和Whisper轉錄"
echo "  b) 內容分析: LLM分析兩個轉錄的準確性、完整性和醫療術語"
echo "  c) 智能合併: 選擇每個部分的最佳版本"
echo "  d) 錯誤修正: 修正明顯的醫療術語錯誤"
echo "  e) 輸出生成: 生成單一、連貫的修正轉錄"
echo ""

# 3. 實際比較示例
echo "## 3. 實際比較示例"
echo "讓我們查看一個具體的比較案例："
echo ""

# 找到測試結果目錄
TEST_DIR="./test_multi_asr_fix_20250823_105552"
if [ -d "$TEST_DIR" ]; then
    echo "使用測試結果目錄: $TEST_DIR"
    
    # 選擇一個示例文件
    SAMPLE_FILE=$(find "$TEST_DIR" -name "*.txt" | head -1)
    if [ -n "$SAMPLE_FILE" ]; then
        BASE_NAME=$(basename "$SAMPLE_FILE" .txt.txt)
        echo "分析文件: $BASE_NAME"
        echo ""
        
        # 顯示原始轉錄
        echo "### 原始轉錄比較："
        echo ""
        
        CANARY_FILE="./pipeline_results_20250823_095857/merged_segmented_transcripts/canary-1b_${BASE_NAME}.txt"
        WHISPER_FILE="./pipeline_results_20250823_095857/merged_segmented_transcripts/large-v3_${BASE_NAME}.txt"
        
        if [ -f "$CANARY_FILE" ] && [ -f "$WHISPER_FILE" ]; then
            echo "**Canary ASR結果:**"
            cat "$CANARY_FILE"
            echo ""
            echo "**Whisper ASR結果:**"
            cat "$WHISPER_FILE"
            echo ""
            echo "**LLM合併結果:**"
            cat "$SAMPLE_FILE"
            echo ""
            
            # 分析差異
            echo "### 差異分析："
            echo ""
            
            # 計算字數
            CANARY_WORDS=$(wc -w < "$CANARY_FILE")
            WHISPER_WORDS=$(wc -w < "$WHISPER_FILE")
            COMBINED_WORDS=$(wc -w < "$SAMPLE_FILE")
            
            echo "  - Canary字數: $CANARY_WORDS"
            echo "  - Whisper字數: $WHISPER_WORDS"
            echo "  - 合併字數: $COMBINED_WORDS"
            echo ""
            
            # 檢查醫療術語
            echo "### 醫療術語檢查："
            echo ""
            
            # 檢查常見醫療術語
            MEDICAL_TERMS=("BP" "HR" "RR" "GCS" "ETA" "MOI" "LOC" "O2" "IV" "CPR")
            
            for term in "${MEDICAL_TERMS[@]}"; do
                CANARY_COUNT=$(grep -o "$term" "$CANARY_FILE" | wc -l)
                WHISPER_COUNT=$(grep -o "$term" "$WHISPER_FILE" | wc -l)
                COMBINED_COUNT=$(grep -o "$term" "$SAMPLE_FILE" | wc -l)
                
                if [ $CANARY_COUNT -gt 0 ] || [ $WHISPER_COUNT -gt 0 ] || [ $COMBINED_COUNT -gt 0 ]; then
                    echo "  - $term: Canary($CANARY_COUNT) Whisper($WHISPER_COUNT) 合併($COMBINED_COUNT)"
                fi
            done
        else
            echo "找不到對應的原始文件"
        fi
    fi
else
    echo "找不到測試結果目錄"
fi

echo ""
echo "## 4. 比較算法詳解"
echo ""

# 顯示比較算法
echo "### 4.1 文件配對算法："
echo "```python"
echo "def create_multi_asr_mapping(input_dir):"
echo "    # 1. 掃描所有轉錄文件"
echo "    # 2. 按模型分類（canary-1b_* vs large-v3_*）"
echo "    # 3. 提取基礎文件名（去除模型前綴）"
echo "    # 4. 創建配對映射"
echo "```"
echo ""

echo "### 4.2 LLM比較算法："
echo "```python"
echo "def compare_transcripts(canary_text, whisper_text, prompt):"
echo "    # 1. 格式化prompt模板"
echo "    # 2. 替換佔位符：{canary_transcript}, {whisper_transcript}"
echo "    # 3. 使用LLM生成比較結果"
echo "    # 4. 提取生成的文本部分"
echo "    # 5. 錯誤處理：返回較長的轉錄作為fallback"
echo "```"
echo ""

echo "### 4.3 Prompt模板："
echo "```"
echo "You are an expert medical transcription specialist..."
echo "COMPARISON GUIDELINES:"
echo "1. Analyze both transcriptions for accuracy, completeness, and medical terminology"
echo "2. Identify which transcription is more accurate for different parts of the message"
echo "3. Combine the best elements from both transcriptions"
echo "4. Correct any obvious medical terminology errors"
echo "5. Maintain the original meaning and context"
echo "6. Provide a single, coherent, corrected transcript"
echo "```"
echo ""

echo "## 5. 輸出結果分析"
echo ""

# 統計分析
if [ -d "$TEST_DIR" ]; then
    echo "### 5.1 處理統計："
    TOTAL_FILES=$(find "$TEST_DIR" -name "*.txt" | wc -l)
    echo "  - 總處理文件數: $TOTAL_FILES"
    
    # 計算平均字數
    TOTAL_WORDS=0
    FILE_COUNT=0
    
    for file in "$TEST_DIR"/*.txt; do
        if [ -f "$file" ]; then
            WORDS=$(wc -w < "$file")
            TOTAL_WORDS=$((TOTAL_WORDS + WORDS))
            FILE_COUNT=$((FILE_COUNT + 1))
        fi
    done
    
    if [ $FILE_COUNT -gt 0 ]; then
        AVG_WORDS=$((TOTAL_WORDS / FILE_COUNT))
        echo "  - 平均字數: $AVG_WORDS"
        echo "  - 總字數: $TOTAL_WORDS"
    fi
    echo ""
    
    echo "### 5.2 輸出目錄結構："
    echo "  $TEST_DIR/"
    echo "  ├── 202412010133-841696-14744_call_2.txt.txt"
    echo "  ├── 202412010731-280830-14744_call_1.txt.txt"
    echo "  └── ... (共 $TOTAL_FILES 個合併文件)"
    echo ""
    
    echo "### 5.3 文件命名規則："
    echo "  - 輸入: canary-1b_[filename].txt, large-v3_[filename].txt"
    echo "  - 輸出: [filename].txt.txt (去除模型前綴)"
    echo ""
fi

echo "## 6. 優勢和特點"
echo ""
echo "### 6.1 技術優勢："
echo "  ✅ 自動文件配對：無需手動匹配"
echo "  ✅ 智能內容分析：LLM理解語義"
echo "  ✅ 醫療術語修正：專業領域知識"
echo "  ✅ 錯誤處理機制：fallback策略"
echo "  ✅ 批量處理：支持大量文件"
echo ""
echo "### 6.2 實際效果："
echo "  📈 準確性提升：結合兩個ASR模型的優勢"
echo "  📈 完整性改善：補充缺失信息"
echo "  📈 專業性增強：醫療術語標準化"
echo "  📈 可讀性提高：語義連貫性改善"
echo ""

echo "## 7. 使用建議"
echo ""
echo "### 7.1 最佳實踐："
echo "  🔧 確保兩個ASR模型都有輸出"
echo "  🔧 使用相同的音頻文件作為輸入"
echo "  🔧 檢查輸出文件的完整性"
echo "  🔧 根據需要調整prompt模板"
echo ""
echo "### 7.2 性能優化："
echo "  ⚡ 使用GPU加速LLM推理"
echo "  ⚡ 調整batch_size提高效率"
echo "  ⚡ 設置適當的temperature參數"
echo "  ⚡ 監控內存使用情況"
echo ""

echo "=== 分析完成 ==="
