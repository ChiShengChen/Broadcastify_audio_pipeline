#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=== Audio Filter Conflict Test Script ==="
echo "This script tests the handling of filter conflicts"
echo ""

# Configuration
AUDIO_DIR="/media/meow/One Touch/ems_call/long_audio_test_dataset"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/media/meow/One Touch/ems_call/filter_conflict_test_${TIMESTAMP}"
PYTHON_EXEC="python3"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Test 1: Audio filtering only (no VAD)"
echo "====================================="
TEST1_DIR="$OUTPUT_DIR/test1_filter_only"
mkdir -p "$TEST1_DIR"

./run_pipeline.sh \
    --input_dir "$AUDIO_DIR" \
    --output_dir "$TEST1_DIR" \
    --ground_truth "/media/meow/One Touch/ems_call/long_audio_test_dataset/long_audio_ground_truth.csv" \
    --use-audio-filtering \
    --filter-highpass-cutoff 300.0 \
    --filter-lowcut 300.0 \
    --filter-highcut 3000.0 \
    --filter-order 5 \
    --filter-enable-wiener

echo ""
echo "Test 2: Enhanced VAD only (no separate filtering)"
echo "================================================="
TEST2_DIR="$OUTPUT_DIR/test2_enhanced_vad_only"
mkdir -p "$TEST2_DIR"

./run_pipeline.sh \
    --input_dir "$AUDIO_DIR" \
    --output_dir "$TEST2_DIR" \
    --ground_truth "/media/meow/One Touch/ems_call/long_audio_test_dataset/long_audio_ground_truth.csv" \
    --use-enhanced-vad \
    --vad-threshold 0.5 \
    --vad-min-speech 0.5 \
    --vad-min-silence 0.3

echo ""
echo "Test 3: Both audio filtering AND enhanced VAD (conflict resolution)"
echo "=================================================================="
TEST3_DIR="$OUTPUT_DIR/test3_both_filters"
mkdir -p "$TEST3_DIR"

./run_pipeline.sh \
    --input_dir "$AUDIO_DIR" \
    --output_dir "$TEST3_DIR" \
    --ground_truth "/media/meow/One Touch/ems_call/long_audio_test_dataset/long_audio_ground_truth.csv" \
    --use-audio-filtering \
    --filter-highpass-cutoff 300.0 \
    --filter-lowcut 300.0 \
    --filter-highcut 3000.0 \
    --filter-order 5 \
    --filter-enable-wiener \
    --use-enhanced-vad \
    --vad-threshold 0.5 \
    --vad-min-speech 0.5 \
    --vad-min-silence 0.3

echo ""
echo "Test 4: Basic VAD + audio filtering (no conflict)"
echo "================================================="
TEST4_DIR="$OUTPUT_DIR/test4_basic_vad_filter"
mkdir -p "$TEST4_DIR"

./run_pipeline.sh \
    --input_dir "$AUDIO_DIR" \
    --output_dir "$TEST4_DIR" \
    --ground_truth "/media/meow/One Touch/ems_call/long_audio_test_dataset/long_audio_ground_truth.csv" \
    --use-audio-filtering \
    --filter-highpass-cutoff 300.0 \
    --filter-lowcut 300.0 \
    --filter-highcut 3000.0 \
    --filter-order 5 \
    --filter-enable-wiener \
    --use-vad \
    --vad-threshold 0.5 \
    --vad-min-speech 0.5 \
    --vad-min-silence 0.3

echo ""
echo "=== Test Summary ==="
echo "All tests completed. Results saved to: $OUTPUT_DIR"
echo ""
echo "Test directories:"
echo "  Test 1 (Filter only): $TEST1_DIR"
echo "  Test 2 (Enhanced VAD only): $TEST2_DIR"
echo "  Test 3 (Both - conflict resolved): $TEST3_DIR"
echo "  Test 4 (Basic VAD + filter): $TEST4_DIR"
echo ""
echo "Expected behavior:"
echo "  - Test 1: Audio filtered once by audio_filter.py"
echo "  - Test 2: Audio filtered once by enhanced_vad_pipeline.py"
echo "  - Test 3: Audio filtered once by audio_filter.py, enhanced VAD skips filtering"
echo "  - Test 4: Audio filtered once by audio_filter.py, basic VAD has no filters"
echo ""
echo "Check the pipeline logs for filter conflict warnings and resolution messages." 