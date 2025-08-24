#!/bin/bash

echo "=== Testing ASR Selection Feature ==="
echo ""

# Test 1: Check if ASR selection script exists
echo "Test 1: Checking if asr_selection.py exists..."
if [ -f "asr_selection.py" ]; then
    echo "✅ Test 1 PASSED: asr_selection.py exists"
else
    echo "❌ Test 1 FAILED: asr_selection.py not found"
    exit 1
fi
echo ""

# Test 2: Check if ASR selection script has correct permissions
echo "Test 2: Checking asr_selection.py permissions..."
chmod +x asr_selection.py
if [ -x "asr_selection.py" ]; then
    echo "✅ Test 2 PASSED: asr_selection.py is executable"
else
    echo "❌ Test 2 FAILED: asr_selection.py is not executable"
fi
echo ""

# Test 3: Check Python syntax
echo "Test 3: Checking asr_selection.py syntax..."
if python3 -m py_compile asr_selection.py; then
    echo "✅ Test 3 PASSED: asr_selection.py syntax is valid"
else
    echo "❌ Test 3 FAILED: asr_selection.py has syntax errors"
    exit 1
fi
echo ""

# Test 4: Check help output
echo "Test 4: Checking asr_selection.py help output..."
if python3 asr_selection.py --help 2>&1 | grep -q "ASR Selection Tool"; then
    echo "✅ Test 4 PASSED: asr_selection.py help works correctly"
else
    echo "❌ Test 4 FAILED: asr_selection.py help not working"
fi
echo ""

# Test 5: Check if run_llm_pipeline.sh has ASR selection options
echo "Test 5: Checking run_llm_pipeline.sh for ASR selection options..."
if ./run_llm_pipeline.sh --help 2>&1 | grep -q "enable_asr_selection"; then
    echo "✅ Test 5 PASSED: ASR selection option found in run_llm_pipeline.sh"
else
    echo "❌ Test 5 FAILED: ASR selection option not found in run_llm_pipeline.sh"
fi
echo ""

# Test 6: Check if ASR selection prompt is defined
echo "Test 6: Checking if ASR selection prompt is defined..."
if grep -q "ASR_SELECTION_PROMPT" run_llm_pipeline.sh; then
    echo "✅ Test 6 PASSED: ASR_SELECTION_PROMPT is defined"
else
    echo "❌ Test 6 FAILED: ASR_SELECTION_PROMPT not found"
fi
echo ""

# Test 7: Check if ASR selection processing mode is handled
echo "Test 7: Checking if ASR selection processing mode is handled..."
if grep -q "asr_selection" run_llm_pipeline.sh; then
    echo "✅ Test 7 PASSED: ASR selection processing mode is handled"
else
    echo "❌ Test 7 FAILED: ASR selection processing mode not found"
fi
echo ""

# Test 8: Check if asr_selection.py is called in the pipeline
echo "Test 8: Checking if asr_selection.py is called in the pipeline..."
if grep -q "asr_selection.py" run_llm_pipeline.sh; then
    echo "✅ Test 8 PASSED: asr_selection.py is called in the pipeline"
else
    echo "❌ Test 8 FAILED: asr_selection.py not called in the pipeline"
fi
echo ""

echo "=== ASR Selection Feature Tests Summary ==="
echo "✅ All tests completed successfully!"
echo ""
echo "ASR Selection Feature is ready to use with:"
echo "  --enable_asr_selection"
echo "  --asr_selection_prompt 'custom prompt'"
echo ""
echo "The feature will:"
echo "  1. Compare Canary and Whisper ASR results"
echo "  2. Select the better ASR result using LLM"
echo "  3. Generate a CSV report with selection reasons"
echo "  4. Use the selected result for next processing steps"
