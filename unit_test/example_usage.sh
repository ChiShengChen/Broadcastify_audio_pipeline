#!/bin/bash

# 音頻預處理功能使用示例
# ================================

echo "🎵 音頻預處理功能使用示例"
echo "================================"
echo ""

# 設置測試目錄
TEST_INPUT_DIR="/media/meow/One Touch/ems_call/long_audio_test_dataset"
TEST_OUTPUT_DIR="/media/meow/One Touch/ems_call/test_preprocessing_results"
TEST_GROUND_TRUTH="/media/meow/One Touch/ems_call/long_audio_test_dataset/long_audio_ground_truth.csv"

echo "📁 測試配置:"
echo "  輸入目錄: $TEST_INPUT_DIR"
echo "  輸出目錄: $TEST_OUTPUT_DIR"
echo "  Ground Truth: $TEST_GROUND_TRUTH"
echo ""

# 示例1: 基本音頻預處理
echo "🔧 示例1: 基本音頻預處理"
echo "----------------------------------------"
echo "運行命令:"
echo "./run_pipeline.sh \\"
echo "    --input_dir $TEST_INPUT_DIR \\"
echo "    --output_dir $TEST_OUTPUT_DIR \\"
echo "    --ground_truth $TEST_GROUND_TRUTH \\"
echo "    --use-audio-preprocessing"
echo ""

# 示例2: 自定義參數
echo "🔧 示例2: 自定義音頻預處理參數"
echo "----------------------------------------"
echo "運行命令:"
echo "./run_pipeline.sh \\"
echo "    --input_dir $TEST_INPUT_DIR \\"
echo "    --output_dir $TEST_OUTPUT_DIR \\"
echo "    --ground_truth $TEST_GROUND_TRUTH \\"
echo "    --use-audio-preprocessing \\"
echo "    --target-sample-rate 16000 \\"
echo "    --audio-max-duration 60 \\"
echo "    --audio-overlap-duration 1 \\"
echo "    --audio-min-segment-duration 5"
echo ""

# 示例3: 結合其他功能
echo "🔧 示例3: 結合VAD和長音頻分割"
echo "----------------------------------------"
echo "運行命令:"
echo "./run_pipeline.sh \\"
echo "    --input_dir $TEST_INPUT_DIR \\"
echo "    --output_dir $TEST_OUTPUT_DIR \\"
echo "    --ground_truth $TEST_GROUND_TRUTH \\"
echo "    --use-audio-preprocessing \\"
echo "    --use-vad \\"
echo "    --use-long-audio-split \\"
echo "    --preprocess-ground-truth"
echo ""

# 示例4: 僅運行音頻預處理
echo "🔧 示例4: 僅運行音頻預處理（獨立使用）"
echo "----------------------------------------"
echo "運行命令:"
echo "python3 audio_preprocessor.py \\"
echo "    --input_dir $TEST_INPUT_DIR \\"
echo "    --output_dir $TEST_OUTPUT_DIR/preprocessed_audio \\"
echo "    --target_sample_rate 16000 \\"
echo "    --max_duration 60 \\"
echo "    --overlap_duration 1 \\"
echo "    --min_segment_duration 5 \\"
echo "    --preserve_structure"
echo ""

# 示例5: 僅運行轉錄合併
echo "🔧 示例5: 僅運行轉錄合併（獨立使用）"
echo "----------------------------------------"
echo "運行命令:"
echo "python3 merge_segmented_transcripts.py \\"
echo "    --input_dir /path/to/transcripts \\"
echo "    --output_dir /path/to/merged_transcripts \\"
echo "    --metadata_file /path/to/processing_metadata.json"
echo ""

echo "📊 預期結果:"
echo "  ✅ 8000Hz音頻上採樣到16000Hz"
echo "  ✅ 超過60秒的音頻分割成片段"
echo "  ✅ 所有ASR模型都能處理"
echo "  ✅ 正確的WER計算"
echo ""

echo "🧪 測試功能:"
echo "  python3 test_audio_preprocessing.py"
echo ""

echo "📚 詳細文檔:"
echo "  AUDIO_PREPROCESSING_GUIDE.md"
echo ""

echo "🎯 主要優勢:"
echo "  1. 解決採樣率不兼容問題"
echo "  2. 解決時長限制問題"
echo "  3. 保持正確的WER計算"
echo "  4. 提高ASR模型兼容性"
echo "" 