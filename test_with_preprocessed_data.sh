#!/bin/bash

# 使用預處理測試數據運行ASR管道
# ======================================

echo "🧪 使用預處理測試數據運行ASR管道"
echo "======================================"
echo ""

# 設置測試目錄
PREPROCESSED_INPUT_DIR="/media/meow/One Touch/ems_call/preprocessed_test_integrated"
TEST_OUTPUT_DIR="/media/meow/One Touch/ems_call/test_results_$(date +%Y%m%d_%H%M%S)"
TEST_GROUND_TRUTH="/media/meow/One Touch/ems_call/test_ground_truth.csv"

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
long_audio_large-v3.wav,This is a long audio file that has been processed for ASR testing.
long_audio_canary-1b_part001.wav,This is part one of a long audio file.
long_audio_canary-1b_part002.wav,This is part two of a long audio file.
long_audio_parakeet-tdt-0.6b-v2.wav,This is a long audio file that has been processed for ASR testing.
long_audio_wav2vec-xls-r.wav,This is a long audio file that has been processed for ASR testing.
very_long_audio_large-v3.wav,This is a very long audio file that has been processed for ASR testing and evaluation.
very_long_audio_canary-1b_part001.wav,This is part one of a very long audio file.
very_long_audio_canary-1b_part002.wav,This is part two of a very long audio file.
very_long_audio_canary-1b_part003.wav,This is part three of a very long audio file.
very_long_audio_canary-1b_part004.wav,This is part four of a very long audio file.
very_long_audio_canary-1b_part005.wav,This is part five of a very long audio file.
very_long_audio_canary-1b_part006.wav,This is part six of a very long audio file.
very_long_audio_parakeet-tdt-0.6b-v2.wav,This is a very long audio file that has been processed for ASR testing and evaluation.
very_long_audio_wav2vec-xls-r.wav,This is a very long audio file that has been processed for ASR testing and evaluation.
EOF

echo "✅ Ground Truth文件已創建: $TEST_GROUND_TRUTH"
echo ""

# 運行測試1: 基本ASR處理（不使用音頻預處理，因為數據已經預處理過）
echo "🔧 測試1: 基本ASR處理"
echo "----------------------------------------"
echo "運行命令:"
echo "./run_pipeline.sh \\"
echo "    --input_dir $PREPROCESSED_INPUT_DIR \\"
echo "    --output_dir $TEST_OUTPUT_DIR \\"
echo "    --ground_truth $TEST_GROUND_TRUTH \\"
echo "    --no-audio-preprocessing \\"
echo "    --no-vad \\"
echo "    --no-long-audio-split \\"
echo "    --preprocess-ground-truth"
echo ""

# 實際運行
echo "🚀 開始運行測試1..."
./run_pipeline.sh \
    --input_dir "$PREPROCESSED_INPUT_DIR" \
    --output_dir "$TEST_OUTPUT_DIR" \
    --ground_truth "$TEST_GROUND_TRUTH" \
    --no-audio-preprocessing \
    --no-vad \
    --no-long-audio-split \
    --preprocess-ground-truth

if [ $? -eq 0 ]; then
    echo "✅ 測試1完成成功！"
else
    echo "❌ 測試1失敗"
fi
echo ""

# 運行測試2: 使用VAD處理
echo "🔧 測試2: 使用VAD處理"
echo "----------------------------------------"
VAD_OUTPUT_DIR="${TEST_OUTPUT_DIR}_vad"

echo "運行命令:"
echo "./run_pipeline.sh \\"
echo "    --input_dir $PREPROCESSED_INPUT_DIR \\"
echo "    --output_dir $VAD_OUTPUT_DIR \\"
echo "    --ground_truth $TEST_GROUND_TRUTH \\"
echo "    --no-audio-preprocessing \\"
echo "    --use-vad \\"
echo "    --no-long-audio-split \\"
echo "    --preprocess-ground-truth"
echo ""

# 實際運行
echo "🚀 開始運行測試2..."
./run_pipeline.sh \
    --input_dir "$PREPROCESSED_INPUT_DIR" \
    --output_dir "$VAD_OUTPUT_DIR" \
    --ground_truth "$TEST_GROUND_TRUTH" \
    --no-audio-preprocessing \
    --use-vad \
    --no-long-audio-split \
    --preprocess-ground-truth

if [ $? -eq 0 ]; then
    echo "✅ 測試2完成成功！"
else
    echo "❌ 測試2失敗"
fi
echo ""

# 運行測試3: 使用長音頻分割
echo "🔧 測試3: 使用長音頻分割"
echo "----------------------------------------"
SPLIT_OUTPUT_DIR="${TEST_OUTPUT_DIR}_split"

echo "運行命令:"
echo "./run_pipeline.sh \\"
echo "    --input_dir $PREPROCESSED_INPUT_DIR \\"
echo "    --output_dir $SPLIT_OUTPUT_DIR \\"
echo "    --ground_truth $TEST_GROUND_TRUTH \\"
echo "    --no-audio-preprocessing \\"
echo "    --no-vad \\"
echo "    --use-long-audio-split \\"
echo "    --preprocess-ground-truth"
echo ""

# 實際運行
echo "🚀 開始運行測試3..."
./run_pipeline.sh \
    --input_dir "$PREPROCESSED_INPUT_DIR" \
    --output_dir "$SPLIT_OUTPUT_DIR" \
    --ground_truth "$TEST_GROUND_TRUTH" \
    --no-audio-preprocessing \
    --no-vad \
    --use-long-audio-split \
    --preprocess-ground-truth

if [ $? -eq 0 ]; then
    echo "✅ 測試3完成成功！"
else
    echo "❌ 測試3失敗"
fi
echo ""

# 運行測試4: 完整功能測試
echo "🔧 測試4: 完整功能測試"
echo "----------------------------------------"
FULL_OUTPUT_DIR="${TEST_OUTPUT_DIR}_full"

echo "運行命令:"
echo "./run_pipeline.sh \\"
echo "    --input_dir $PREPROCESSED_INPUT_DIR \\"
echo "    --output_dir $FULL_OUTPUT_DIR \\"
echo "    --ground_truth $TEST_GROUND_TRUTH \\"
echo "    --no-audio-preprocessing \\"
echo "    --use-vad \\"
echo "    --use-long-audio-split \\"
echo "    --preprocess-ground-truth"
echo ""

# 實際運行
echo "🚀 開始運行測試4..."
./run_pipeline.sh \
    --input_dir "$PREPROCESSED_INPUT_DIR" \
    --output_dir "$FULL_OUTPUT_DIR" \
    --ground_truth "$TEST_GROUND_TRUTH" \
    --no-audio-preprocessing \
    --use-vad \
    --use-long-audio-split \
    --preprocess-ground-truth

if [ $? -eq 0 ]; then
    echo "✅ 測試4完成成功！"
else
    echo "❌ 測試4失敗"
fi
echo ""

# 生成測試報告
echo "📊 生成測試報告..."
echo "======================================"

echo "測試結果摘要:"
echo "  測試1 (基本ASR): $([ -d "$TEST_OUTPUT_DIR" ] && echo "✅ 完成" || echo "❌ 失敗")"
echo "  測試2 (VAD): $([ -d "$VAD_OUTPUT_DIR" ] && echo "✅ 完成" || echo "❌ 失敗")"
echo "  測試3 (長音頻分割): $([ -d "$SPLIT_OUTPUT_DIR" ] && echo "✅ 完成" || echo "❌ 失敗")"
echo "  測試4 (完整功能): $([ -d "$FULL_OUTPUT_DIR" ] && echo "✅ 完成" || echo "❌ 失敗")"
echo ""

# 檢查結果文件
echo "📁 結果目錄:"
for dir in "$TEST_OUTPUT_DIR" "$VAD_OUTPUT_DIR" "$SPLIT_OUTPUT_DIR" "$FULL_OUTPUT_DIR"; do
    if [ -d "$dir" ]; then
        echo "  $dir:"
        if [ -f "$dir/asr_evaluation_results.csv" ]; then
            echo "    ✅ ASR評估結果"
        fi
        if [ -f "$dir/pipeline_summary.txt" ]; then
            echo "    ✅ 管道摘要"
        fi
        if [ -f "$dir/error_analysis.log" ]; then
            echo "    ⚠️  錯誤分析日誌"
        fi
    fi
done
echo ""

echo "🎉 測試完成！"
echo "📚 查看詳細結果請檢查上述目錄中的文件。" 