#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=== Audio Filter Test Script ==="
echo "This script demonstrates the new audio filtering functionality"
echo ""

# Configuration
AUDIO_DIR="/media/meow/One Touch/ems_call/long_audio_test_dataset"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/media/meow/One Touch/ems_call/audio_filter_test_${TIMESTAMP}"
PYTHON_EXEC="python3"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Test 1: Basic audio filtering (band-pass + high-pass)"
echo "=================================================="
TEST1_DIR="$OUTPUT_DIR/test1_basic_filtering"
mkdir -p "$TEST1_DIR"

$PYTHON_EXEC audio_filter.py \
    --input_dir "$AUDIO_DIR" \
    --output_dir "$TEST1_DIR" \
    --enable-filters \
    --highpass_cutoff 300.0 \
    --lowcut 300.0 \
    --highcut 3000.0 \
    --filter_order 5 \
    --target_sample_rate 16000

echo ""
echo "Test 2: Audio filtering with Wiener filter"
echo "=========================================="
TEST2_DIR="$OUTPUT_DIR/test2_filtering_with_wiener"
mkdir -p "$TEST2_DIR"

$PYTHON_EXEC audio_filter.py \
    --input_dir "$AUDIO_DIR" \
    --output_dir "$TEST2_DIR" \
    --enable-filters \
    --enable-wiener \
    --highpass_cutoff 300.0 \
    --lowcut 300.0 \
    --highcut 3000.0 \
    --filter_order 5 \
    --target_sample_rate 16000

echo ""
echo "Test 3: Pipeline with audio filtering only (no VAD)"
echo "==================================================="
TEST3_DIR="$OUTPUT_DIR/test3_pipeline_filter_only"
mkdir -p "$TEST3_DIR"

# Run pipeline with audio filtering but no VAD
./run_pipeline.sh \
    --input_dir "$AUDIO_DIR" \
    --output_dir "$TEST3_DIR" \
    --ground_truth "/media/meow/One Touch/ems_call/long_audio_test_dataset/long_audio_ground_truth.csv" \
    --use-audio-filtering \
    --filter-highpass-cutoff 300.0 \
    --filter-lowcut 300.0 \
    --filter-highcut 3000.0 \
    --filter-order 5 \
    --filter-enable-wiener

echo ""
echo "Test 4: Pipeline with audio filtering + VAD"
echo "==========================================="
TEST4_DIR="$OUTPUT_DIR/test4_pipeline_filter_and_vad"
mkdir -p "$TEST4_DIR"

# Run pipeline with both audio filtering and VAD
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
    --use-vad

echo ""
echo "=== Test Summary ==="
echo "All tests completed. Results saved to: $OUTPUT_DIR"
echo ""
echo "Test directories:"
echo "  Test 1 (Basic filtering): $TEST1_DIR"
echo "  Test 2 (Filtering + Wiener): $TEST2_DIR"
echo "  Test 3 (Pipeline filter only): $TEST3_DIR"
echo "  Test 4 (Pipeline filter + VAD): $TEST4_DIR"
echo ""
echo "Usage examples:"
echo "  # Basic audio filtering only"
echo "  python3 audio_filter.py --input_dir /path/to/audio --output_dir /path/to/output --enable-filters"
echo ""
echo "  # Audio filtering with Wiener filter"
echo "  python3 audio_filter.py --input_dir /path/to/audio --output_dir /path/to/output --enable-filters --enable-wiener"
echo ""
echo "  # Pipeline with audio filtering only (no VAD)"
echo "  ./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results --use-audio-filtering"
echo ""
echo "  # Pipeline with audio filtering + VAD"
echo "  ./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results --use-audio-filtering --use-vad" 