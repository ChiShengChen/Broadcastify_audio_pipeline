#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=== Testing Analysis Tool Fix ==="
echo "This script tests the fix for summary file handling in model analysis"
echo ""

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_DIR="/media/meow/One Touch/ems_call/analysis_test_${TIMESTAMP}"
PYTHON_EXEC="python3"

# Create test directory
mkdir -p "$TEST_DIR"

echo "Test 1: Running analysis on recent pipeline results"
echo "=================================================="

# Test with the recent pipeline results
$PYTHON_EXEC tool/analyze_model_files_enhanced.py \
    --transcript_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_025214/merged_segmented_transcripts" \
    --ground_truth_file "/media/meow/One Touch/ems_call/pipeline_results_20250729_025214/preprocessed_ground_truth.csv" \
    --output_file "$TEST_DIR/analysis_results.txt" \
    --error_log_file "$TEST_DIR/error_log.txt"

echo ""
echo "Test 2: Checking if summary files are properly ignored"
echo "======================================================"

# Check the analysis results
if [ -f "$TEST_DIR/analysis_results.txt" ]; then
    echo "Analysis completed successfully"
    echo ""
    echo "Checking for summary file warnings..."
    
    if grep -q "merging_summary.txt" "$TEST_DIR/error_log.txt"; then
        echo "❌ Still detecting summary files as errors"
    else
        echo "✅ Summary files properly ignored"
    fi
    
    echo ""
    echo "Model analysis summary:"
    grep -A 10 "Summary Table:" "$TEST_DIR/analysis_results.txt" || echo "No summary table found"
else
    echo "❌ Analysis failed"
fi

echo ""
echo "Test 3: Creating test files to verify fix"
echo "========================================="

# Create test transcript files
mkdir -p "$TEST_DIR/test_transcripts"

# Create normal transcript files
echo "This is a normal transcript file" > "$TEST_DIR/test_transcripts/wav2vec-xls-r_test_001.txt"
echo "Another normal transcript" > "$TEST_DIR/test_transcripts/canary-1b_test_002.txt"

# Create a summary file (should be ignored)
cat > "$TEST_DIR/test_transcripts/merging_summary.txt" << 'EOF'
============================================================
SEGMENTED TRANSCRIPT MERGING SUMMARY
============================================================
Total original files: 2
Successfully merged: 2
Errors: 0
Models processed: wav2vec-xls-r, canary-1b
EOF

# Create a test ground truth file
cat > "$TEST_DIR/test_ground_truth.csv" << 'EOF'
Filename,transcript
test_001.wav,This is a normal transcript file
test_002.wav,Another normal transcript
EOF

echo "Test files created"
echo ""

echo "Test 4: Running analysis on test files"
echo "======================================"

$PYTHON_EXEC tool/analyze_model_files_enhanced.py \
    --transcript_dir "$TEST_DIR/test_transcripts" \
    --ground_truth_file "$TEST_DIR/test_ground_truth.csv" \
    --output_file "$TEST_DIR/test_analysis.txt" \
    --error_log_file "$TEST_DIR/test_error_log.txt"

echo ""
echo "Test 5: Verifying results"
echo "========================"

if [ -f "$TEST_DIR/test_analysis.txt" ]; then
    echo "✅ Test analysis completed"
    
    # Check if summary file was ignored
    if grep -q "merging_summary.txt" "$TEST_DIR/test_error_log.txt"; then
        echo "❌ Summary file still being processed"
    else
        echo "✅ Summary file properly ignored"
    fi
    
    # Check if normal files were processed
    if grep -q "wav2vec-xls-r" "$TEST_DIR/test_analysis.txt" && grep -q "canary-1b" "$TEST_DIR/test_analysis.txt"; then
        echo "✅ Normal transcript files processed correctly"
    else
        echo "❌ Normal files not processed"
    fi
    
    echo ""
    echo "Test analysis results:"
    cat "$TEST_DIR/test_analysis.txt"
else
    echo "❌ Test analysis failed"
fi

echo ""
echo "=== Test Summary ==="
echo "All tests completed. Results saved to: $TEST_DIR"
echo ""
echo "Files created:"
echo "  - Analysis results: $TEST_DIR/analysis_results.txt"
echo "  - Error log: $TEST_DIR/error_log.txt"
echo "  - Test analysis: $TEST_DIR/test_analysis.txt"
echo "  - Test error log: $TEST_DIR/test_error_log.txt"
echo ""
echo "Expected behavior:"
echo "  - Summary files (merging_summary.txt) should be ignored"
echo "  - Normal transcript files should be processed normally"
echo "  - No warnings about summary files in error logs" 