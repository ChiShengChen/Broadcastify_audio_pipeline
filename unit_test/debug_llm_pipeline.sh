#!/bin/bash

# Debug script for LLM pipeline issues
# This script helps diagnose and fix common issues with the LLM pipeline

set -e

echo "=== LLM Pipeline Debug Script ==="
echo "Date: $(date)"
echo ""

# Check Python environment
echo "--- Checking Python Environment ---"
python3 --version
echo ""

# Check required packages
echo "--- Checking Required Packages ---"
REQUIRED_PACKAGES=("torch" "transformers" "accelerate" "bitsandbytes")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✓ $package is installed"
    else
        echo "✗ $package is missing"
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "Missing packages detected. Install with:"
    echo "pip install ${MISSING_PACKAGES[*]}"
    echo ""
fi

# Check CUDA availability
echo "--- Checking CUDA ---"
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
    print(f'Memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')
else:
    print('CUDA not available, will use CPU')
"
echo ""

# Check disk space
echo "--- Checking Disk Space ---"
df -h . | head -2
echo ""

# Check model accessibility
echo "--- Checking Model Accessibility ---"
python3 -c "
from transformers import AutoTokenizer
import sys

models_to_check = [
    'microsoft/DialoGPT-small',
    'BioMistral/BioMistral-7B'
]

for model_name in models_to_check:
    try:
        print(f'Checking {model_name}...')
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f'✓ {model_name} is accessible')
    except Exception as e:
        print(f'✗ {model_name} failed: {e}')
    print()
"

# Test simple model loading
echo "--- Testing Simple Model Loading ---"
python3 -c "
import sys
sys.path.append('.')
try:
    from simple_llm_pipeline import SimpleLLMModel
    print('Testing SimpleLLMModel...')
    model = SimpleLLMModel('microsoft/DialoGPT-small')
    test_prompt = 'Test prompt for medical transcript correction:'
    result = model.generate(test_prompt)
    if result:
        print('✓ Simple model test successful')
        print(f'Result length: {len(result)} characters')
    else:
        print('✗ Simple model returned empty result')
except Exception as e:
    print(f'✗ Simple model test failed: {e}')
    import traceback
    traceback.print_exc()
"
echo ""

# Check input directories
echo "--- Checking Input Directories ---"
ASR_RESULTS_DIR="/media/meow/One Touch/ems_call/pipeline_results_20250729_033902"
if [ -d "$ASR_RESULTS_DIR" ]; then
    echo "✓ ASR results directory exists: $ASR_RESULTS_DIR"
    
    # Check for transcript files
    TRANSCRIPT_COUNT=$(find "$ASR_RESULTS_DIR" -name "*.txt" | wc -l)
    echo "  Total .txt files: $TRANSCRIPT_COUNT"
    
    # Check for Whisper files specifically
    WHISPER_COUNT=$(find "$ASR_RESULTS_DIR" -name "*large-v3*.txt" | wc -l)
    echo "  Whisper (large-v3) files: $WHISPER_COUNT"
    
    if [ $WHISPER_COUNT -gt 0 ]; then
        echo "  Sample Whisper files:"
        find "$ASR_RESULTS_DIR" -name "*large-v3*.txt" | head -3
    fi
else
    echo "✗ ASR results directory not found: $ASR_RESULTS_DIR"
fi
echo ""

# Test file processing
echo "--- Testing File Processing ---"
SAMPLE_FILE=$(find "$ASR_RESULTS_DIR" -name "*large-v3*.txt" | head -1)
if [ -n "$SAMPLE_FILE" ]; then
    echo "Testing with sample file: $SAMPLE_FILE"
    echo "File size: $(stat -f%z "$SAMPLE_FILE" 2>/dev/null || stat -c%s "$SAMPLE_FILE" 2>/dev/null || echo "unknown") bytes"
    echo "File content preview:"
    head -3 "$SAMPLE_FILE" | sed 's/^/  /'
else
    echo "No Whisper files found for testing"
fi
echo ""

echo "=== Debug Complete ==="
echo ""
echo "Recommendations:"
echo "1. If packages are missing, install them with pip"
echo "2. If CUDA is not available, consider using CPU mode or smaller models"
echo "3. If models fail to load, check internet connection and Hugging Face access"
echo "4. If simple model test fails, there may be fundamental setup issues"
echo "5. Check the simple_llm_pipeline.py for a working alternative"