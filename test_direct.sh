#!/bin/bash

# 直接修改配置的ASR管道測試
# ============================

echo "🧪 直接修改配置的ASR管道測試"
echo "============================"
echo ""

# 設置測試目錄
PREPROCESSED_INPUT_DIR="/media/meow/One Touch/ems_call/preprocessed_test_integrated"
TEST_OUTPUT_DIR="/media/meow/One Touch/ems_call/test_direct_$(date +%Y%m%d_%H%M%S)"
TEST_GROUND_TRUTH="/media/meow/One Touch/ems_call/test_ground_truth_direct.csv"

echo "📁 測試配置:"
echo "  預處理音頻目錄: $PREPROCESSED_INPUT_DIR"
echo "  輸出目錄: $TEST_OUTPUT_DIR"
echo "  Ground Truth: $TEST_GROUND_TRUTH"
echo ""

# 檢查預處理目錄
echo "🔍 檢查預處理目錄..."
if [ ! -d "$PREPROCESSED_INPUT_DIR" ]; then
    echo "❌ 預處理目錄不存在: $PREPROCESSED_INPUT_DIR"
    exit 1
fi

# 統計音頻文件
AUDIO_COUNT=$(find "$PREPROCESSED_INPUT_DIR" -name "*.wav" | wc -l)
echo "✅ 找到 $AUDIO_COUNT 個音頻文件"
echo ""

# 創建測試Ground Truth文件
echo "📝 創建測試Ground Truth文件..."
cat > "$TEST_GROUND_TRUTH" << 'EOF'
Filename,transcript
normal_audio_large-v3.wav,This is a normal audio file for testing ASR models.
normal_audio_canary-1b.wav,This is a normal audio file for testing ASR models.
normal_audio_parakeet-tdt-0.6b-v2.wav,This is a normal audio file for testing ASR models.
normal_audio_wav2vec-xls-r.wav,This is a normal audio file for testing ASR models.
short_audio_large-v3.wav,This is a short audio file.
short_audio_canary-1b.wav,This is a short audio file.
short_audio_parakeet-tdt-0.6b-v2.wav,This is a short audio file.
short_audio_wav2vec-xls-r.wav,This is a short audio file.
EOF

echo "✅ Ground Truth文件已創建: $TEST_GROUND_TRUTH"
echo ""

# 創建臨時的run_pipeline腳本，禁用VAD和長音頻分割
echo "🔧 創建臨時配置..."
TEMP_PIPELINE="/tmp/temp_pipeline_$$.sh"

# 複製原始腳本並修改配置
cp run_pipeline.sh "$TEMP_PIPELINE"
chmod +x "$TEMP_PIPELINE"

# 修改配置
sed -i 's/USE_VAD=true/USE_VAD=false/' "$TEMP_PIPELINE"
sed -i 's/USE_LONG_AUDIO_SPLIT=true/USE_LONG_AUDIO_SPLIT=false/' "$TEMP_PIPELINE"

echo "✅ 臨時配置已創建: $TEMP_PIPELINE"
echo ""

# 運行測試
echo "🔧 運行基本ASR測試（禁用VAD和長音頻分割）"
echo "----------------------------------------"
echo "運行命令:"
echo "$TEMP_PIPELINE \\"
echo "    --input_dir $PREPROCESSED_INPUT_DIR \\"
echo "    --output_dir $TEST_OUTPUT_DIR \\"
echo "    --ground_truth $TEST_GROUND_TRUTH \\"
echo "    --no-audio-preprocessing \\"
echo "    --preprocess-ground-truth"
echo ""

# 實際運行
echo "🚀 開始運行測試..."
"$TEMP_PIPELINE" \
    --input_dir "$PREPROCESSED_INPUT_DIR" \
    --output_dir "$TEST_OUTPUT_DIR" \
    --ground_truth "$TEST_GROUND_TRUTH" \
    --no-audio-preprocessing \
    --preprocess-ground-truth

if [ $? -eq 0 ]; then
    echo "✅ 測試完成成功！"
    
    # 檢查結果文件
    echo ""
    echo "📁 檢查結果文件:"
    if [ -f "$TEST_OUTPUT_DIR/asr_evaluation_results.csv" ]; then
        echo "  ✅ ASR評估結果: $TEST_OUTPUT_DIR/asr_evaluation_results.csv"
        echo "  內容預覽:"
        head -5 "$TEST_OUTPUT_DIR/asr_evaluation_results.csv"
    fi
    
    if [ -f "$TEST_OUTPUT_DIR/pipeline_summary.txt" ]; then
        echo "  ✅ 管道摘要: $TEST_OUTPUT_DIR/pipeline_summary.txt"
    fi
    
    if [ -f "$TEST_OUTPUT_DIR/error_analysis.log" ]; then
        echo "  ⚠️  錯誤分析日誌: $TEST_OUTPUT_DIR/error_analysis.log"
    fi
    
    echo ""
    echo "🎉 測試成功完成！"
else
    echo "❌ 測試失敗"
fi

# 清理臨時文件
rm -f "$TEMP_PIPELINE"
echo ""
echo "🧹 臨時文件已清理" 