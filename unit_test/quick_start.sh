#!/bin/bash

# Quick Start Script for Audio Preprocessing
# ========================================

echo "? Audio Preprocessing Quick Start"
echo "=================================="
echo ""

# Check if required files exist
if [ ! -f "audio_preprocessor.py" ]; then
    echo "? Error: audio_preprocessor.py not found"
    echo "Please make sure you're in the correct directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "? Error: python3 not found"
    echo "Please install Python 3.7+"
    exit 1
fi

# Check if required Python packages are available
echo "? Checking Python dependencies..."
python3 -c "
import sys
required_packages = ['numpy', 'soundfile', 'librosa', 'scipy']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f'? {package}')
    except ImportError:
        missing_packages.append(package)
        print(f'? {package} (missing)')

if missing_packages:
    print(f'\n? Missing packages: {', '.join(missing_packages)}')
    print('Please install them with: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('\n? All dependencies are available')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Please install missing dependencies and try again"
    exit 1
fi

# Create test directory
TEST_DIR="./quick_start_test"
OUTPUT_DIR="./quick_start_output"

echo ""
echo "? Creating test environment..."
mkdir -p "$TEST_DIR"
mkdir -p "$OUTPUT_DIR"

# Create a simple test audio file
echo "? Creating test audio file..."
python3 -c "
import numpy as np
import soundfile as sf
import os

# Create a test audio file with various characteristics
test_dir = '$TEST_DIR'

# Create a 2-second audio file at 16kHz
audio = np.random.randn(32000) * 0.1  # 2s at 16kHz
test_path = os.path.join(test_dir, 'test_audio.wav')
sf.write(test_path, audio, 16000)

print(f'? Created test audio: {test_path}')
print(f'  - Duration: 2.0 seconds')
print(f'  - Sample rate: 16000 Hz')
print(f'  - Channels: 1 (mono)')
"

if [ $? -ne 0 ]; then
    echo "? Failed to create test audio file"
    exit 1
fi

echo ""
echo "? Running audio preprocessor test..."

# Run the test
python3 test_audio_preprocessor.py

if [ $? -eq 0 ]; then
    echo ""
    echo "? Test completed successfully!"
    echo ""
    echo "? Next steps:"
    echo "1. Test with your own audio files:"
    echo "   python3 audio_preprocessor.py --input_dir /path/to/audio --output_dir /path/to/output --verbose"
    echo ""
    echo "2. Run the full pipeline with preprocessing:"
    echo "   ./run_preprocessing_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results"
    echo ""
    echo "3. Check the documentation:"
    echo "   cat AUDIO_PREPROCESSING_GUIDE.md"
    echo ""
else
    echo ""
    echo "? Test failed. Please check the error messages above."
    echo ""
    echo "? Troubleshooting:"
    echo "1. Make sure all dependencies are installed"
    echo "2. Check that you have write permissions in the current directory"
    echo "3. Verify that Python 3.7+ is installed"
    echo ""
fi

# Cleanup
echo "? Cleaning up test files..."
rm -rf "$TEST_DIR"
rm -rf "$OUTPUT_DIR"

echo ""
echo "? Quick start completed!"
echo "You can now use the audio preprocessor with your own files." 