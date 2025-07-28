#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=== Testing Multiple Directory Input ==="
echo "This script tests the multiple directory input functionality"
echo ""

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_DIR="/media/meow/One Touch/ems_call/multi_dir_test_${TIMESTAMP}"
PYTHON_EXEC="python3"

# Create test directories
mkdir -p "$TEST_DIR"
mkdir -p "$TEST_DIR/dir1"
mkdir -p "$TEST_DIR/dir2"
mkdir -p "$TEST_DIR/dir3"

echo "Creating test audio files..."

# Create test audio files in different directories
for i in {1..3}; do
    echo "Creating test file in dir$i..."
    $PYTHON_EXEC -c "
import torch
import torchaudio
import os

# Create a simple audio file
sample_rate = 16000
duration = 5.0
t = torch.linspace(0, duration, int(sample_rate * duration))
signal = 0.3 * torch.sin(2 * 3.14159 * 300 * t)  # 300Hz sine wave

# Save to different directories
for i in range(1, 4):
    output_dir = f'$TEST_DIR/dir{i}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'test_audio_{i}.wav')
    torchaudio.save(output_file, signal.unsqueeze(0), sample_rate)
    print(f'Created: {output_file}')
"

done

# Create ground truth file
cat > "$TEST_DIR/ground_truth.csv" << 'EOF'
Filename,transcript
test_audio_1.wav,This is a test audio file number one
test_audio_2.wav,This is a test audio file number two
test_audio_3.wav,This is a test audio file number three
EOF

echo ""
echo "Test 1: Single directory processing"
echo "=================================="

# Test with single directory
$PYTHON_EXEC run_pipeline.sh \
    --input_dir "$TEST_DIR/dir1" \
    --output_dir "$TEST_DIR/results_single" \
    --ground_truth "$TEST_DIR/ground_truth.csv" \
    --no-audio-preprocessing \
    --no-audio-filtering

echo ""
echo "Test 2: Multiple directory processing"
echo "===================================="

# Test with multiple directories
$PYTHON_EXEC run_pipeline.sh \
    --input_dir "$TEST_DIR/dir1 $TEST_DIR/dir2 $TEST_DIR/dir3" \
    --output_dir "$TEST_DIR/results_multi" \
    --ground_truth "$TEST_DIR/ground_truth.csv" \
    --no-audio-preprocessing \
    --no-audio-filtering

echo ""
echo "Test 3: Multiple directory with preprocessing"
echo "============================================"

# Test with multiple directories and preprocessing
$PYTHON_EXEC run_pipeline.sh \
    --input_dir "$TEST_DIR/dir1 $TEST_DIR/dir2" \
    --output_dir "$TEST_DIR/results_multi_preprocessed" \
    --ground_truth "$TEST_DIR/ground_truth.csv" \
    --use-audio-preprocessing \
    --no-audio-filtering

echo ""
echo "Test 4: Multiple directory with filtering"
echo "========================================"

# Test with multiple directories and filtering
$PYTHON_EXEC run_pipeline.sh \
    --input_dir "$TEST_DIR/dir1 $TEST_DIR/dir3" \
    --output_dir "$TEST_DIR/results_multi_filtered" \
    --ground_truth "$TEST_DIR/ground_truth.csv" \
    --no-audio-preprocessing \
    --use-audio-filtering

echo ""
echo "=== Test Summary ==="
echo "All tests completed. Results saved to: $TEST_DIR"
echo ""
echo "Test directories:"
echo "  Single dir: $TEST_DIR/results_single"
echo "  Multi dir: $TEST_DIR/results_multi"
echo "  Multi dir + preprocessing: $TEST_DIR/results_multi_preprocessed"
echo "  Multi dir + filtering: $TEST_DIR/results_multi_filtered"
echo ""
echo "Expected behavior:"
echo "  - Single directory: Process files from dir1 only"
echo "  - Multiple directories: Process files from all directories"
echo "  - Preprocessing: Should handle multiple input directories"
echo "  - Filtering: Should handle multiple input directories"
echo ""
echo "Check the pipeline logs for any errors or warnings." 