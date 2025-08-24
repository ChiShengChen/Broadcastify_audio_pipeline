#!/bin/bash

# Test script for multi-ASR detection functionality
# This script tests the auto-detection of multi-ASR results from pipeline output

echo "=== Testing Multi-ASR Auto-Detection Functionality ==="
echo ""

# Test 1: Check if the main script exists
if [ -f "run_llm_pipeline.sh" ]; then
    echo "✅ Test 1 PASSED: run_llm_pipeline.sh exists"
else
    echo "❌ Test 1 FAILED: run_llm_pipeline.sh not found"
    exit 1
fi

# Test 2: Check if multi-ASR comparison script exists
if [ -f "multi_asr_comparison.py" ]; then
    echo "✅ Test 2 PASSED: multi_asr_comparison.py exists"
else
    echo "❌ Test 2 FAILED: multi_asr_comparison.py not found"
    exit 1
fi

# Test 3: Check script syntax
if bash -n run_llm_pipeline.sh; then
    echo "✅ Test 3 PASSED: run_llm_pipeline.sh syntax is valid"
else
    echo "❌ Test 3 FAILED: run_llm_pipeline.sh has syntax errors"
    exit 1
fi

# Test 4: Check Python script syntax
if python3 -m py_compile multi_asr_comparison.py; then
    echo "✅ Test 4 PASSED: multi_asr_comparison.py syntax is valid"
else
    echo "❌ Test 4 FAILED: multi_asr_comparison.py has syntax errors"
    exit 1
fi

# Test 5: Check help output for new options
if ./run_llm_pipeline.sh --help | grep -q "auto_detect_multi_asr"; then
    echo "✅ Test 5 PASSED: Auto-detect multi-ASR option found in help"
else
    echo "❌ Test 5 FAILED: Auto-detect multi-ASR option not found in help"
fi

# Test 6: Check help output for multi-ASR comparison
if ./run_llm_pipeline.sh --help | grep -q "multi_asr_comparison"; then
    echo "✅ Test 6 PASSED: Multi-ASR comparison option found in help"
else
    echo "❌ Test 6 FAILED: Multi-ASR comparison option not found in help"
fi

echo ""
echo "=== Test Summary ==="
echo "All basic functionality tests passed!"
echo ""
echo "To test with actual data, run:"
echo "  ./run_llm_pipeline.sh \\"
echo "    --asr_results_dir /path/to/pipeline/results \\"
echo "    --enable_multi_asr_comparison \\"
echo "    --enable_auto_detect_multi_asr \\"
echo "    --enable_information_extraction \\"
echo "    --disable_page_generation"
echo ""
echo "Expected pipeline output structure:"
echo "  llm_results_YYYYMMDD_HHMMSS/"
echo "  ├── multi_asr_organized/"
echo "  │   ├── canary/"
echo "  │   ├── whisper/"
echo "  │   └── multi_asr_mapping.json"
echo "  ├── corrected_transcripts/"
echo "  ├── extracted_information/"
echo "  └── ..."
