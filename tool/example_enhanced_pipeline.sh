#!/bin/bash

# Enhanced Pipeline Usage Examples
# =================================
# This script shows different ways to use the enhanced run_pipeline.sh with VAD

set -e

# Configuration
AUDIO_DIR="/media/meow/One Touch/ems_call/random_samples_1_preprocessed"
OUTPUT_BASE="/media/meow/One Touch/ems_call/pipeline_examples"
GROUND_TRUTH="/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv"

echo "Enhanced ASR Pipeline Examples"
echo "=============================="
echo ""
echo "Available examples:"
echo "1. Original workflow (no VAD)"
echo "2. Basic VAD preprocessing"
echo "3. Enhanced VAD with filters"
echo "4. Custom VAD parameters"
echo ""

# Make sure the enhanced pipeline script is executable
chmod +x ems_call/run_pipeline.sh

# Example 1: Original workflow (backward compatible)
echo "Example 1: Original workflow (no VAD)"
echo "-------------------------------------"
cat << 'EOF'
# Run ASR directly on original files (same as before)
bash ems_call/run_pipeline.sh \
    --input_dir "/media/meow/One Touch/ems_call/random_samples_1_preprocessed" \
    --output_dir "/media/meow/One Touch/ems_call/pipeline_examples/original_workflow" \
    --ground_truth "/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv"

# Output structure:
# original_workflow/
# ├── asr_transcripts/           # ASR results on original files
# │   ├── large-v3_file1.txt
# │   ├── canary-1b_file1.txt
# │   └── ...
# ├── asr_evaluation_results.csv # WER evaluation results
# └── pipeline_summary.txt       # Processing summary
EOF
echo ""

# Example 2: Basic VAD preprocessing
echo "Example 2: Basic VAD preprocessing"
echo "-----------------------------------"
cat << 'EOF'
# Run with basic VAD preprocessing
bash ems_call/run_pipeline.sh \
    --input_dir "/media/meow/One Touch/ems_call/random_samples_1_preprocessed" \
    --output_dir "/media/meow/One Touch/ems_call/pipeline_examples/basic_vad" \
    --ground_truth "/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv" \
    --use-vad

# Output structure:
# basic_vad/
# ├── vad_segments/              # VAD extracted speech segments
# │   ├── file1/                 # Original filename as folder
# │   │   ├── segment_001.wav    # Speech segments
# │   │   ├── segment_002.wav
# │   │   └── file1_vad_metadata.json
# │   ├── file2/
# │   └── vad_processing_summary.json
# ├── asr_transcripts/           # ASR results
# │   ├── temp_segments/         # Individual segment transcripts
# │   └── consolidated/          # Consolidated transcripts by original file
# │       ├── large-v3_file1.txt # Combined transcript for file1
# │       ├── canary-1b_file1.txt
# │       └── ...
# ├── asr_evaluation_results.csv
# └── pipeline_summary.txt
EOF
echo ""

# Example 3: Enhanced VAD with filters
echo "Example 3: Enhanced VAD with filters (recommended for noisy audio)"
echo "-----------------------------------------------------------------"
cat << 'EOF'
# Run with enhanced VAD (includes audio filtering)
bash ems_call/run_pipeline.sh \
    --input_dir "/media/meow/One Touch/ems_call/random_samples_1_preprocessed" \
    --output_dir "/media/meow/One Touch/ems_call/pipeline_examples/enhanced_vad" \
    --ground_truth "/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv" \
    --use-enhanced-vad

# This applies audio filters before VAD:
# - High-pass filter (removes low-frequency noise)
# - Band-pass filter (focuses on speech frequencies 300-3000Hz)
# - Optional Wiener filter for noise reduction
EOF
echo ""

# Example 4: Custom VAD parameters
echo "Example 4: Custom VAD parameters"
echo "--------------------------------"
cat << 'EOF'
# Fine-tune VAD parameters for your specific audio
bash ems_call/run_pipeline.sh \
    --input_dir "/media/meow/One Touch/ems_call/random_samples_1_preprocessed" \
    --output_dir "/media/meow/One Touch/ems_call/pipeline_examples/custom_vad" \
    --ground_truth "/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv" \
    --use-vad \
    --vad-threshold 0.6 \
    --vad-min-speech 1.0 \
    --vad-min-silence 0.5

# Parameter explanations:
# --vad-threshold 0.6      Higher threshold = more conservative speech detection
# --vad-min-speech 1.0     Only keep speech segments >= 1.0 seconds
# --vad-min-silence 0.5    Need >= 0.5 seconds silence to split segments
EOF
echo ""

# Show help
echo "Getting Help:"
echo "-------------"
cat << 'EOF'
# View all available options
bash ems_call/run_pipeline.sh --help

# Key options:
# --input_dir DIR              Input directory with audio files
# --output_dir DIR             Output directory for results  
# --ground_truth FILE          Ground truth CSV file
# --use-vad                    Enable basic VAD preprocessing
# --use-enhanced-vad           Enable enhanced VAD with filters
# --vad-threshold FLOAT        VAD speech threshold (default: 0.5)
# --vad-min-speech FLOAT       Min speech duration (default: 0.5s)
# --vad-min-silence FLOAT      Min silence duration (default: 0.3s)
EOF
echo ""

echo "Quick Test (using first few files):"
echo "-----------------------------------"
echo "# Test with a small subset first"
echo "mkdir -p /tmp/test_audio"
echo "cp \"$AUDIO_DIR\"/*.wav /tmp/test_audio/ 2>/dev/null | head -3"
echo ""
echo "bash ems_call/run_pipeline.sh \\"
echo "    --input_dir /tmp/test_audio \\"
echo "    --output_dir /tmp/test_results \\"
echo "    --use-vad"
echo ""

echo "Performance Expectations:"
echo "-------------------------"
cat << 'EOF'
Processing time comparison (for medical call audio):
- Original ASR:     100% time
- Basic VAD + ASR:  ~40% time (2.5x speedup)
- Enhanced VAD + ASR: ~45% time (2.2x speedup)

The VAD preprocessing extracts only speech segments, significantly reducing
the amount of audio that needs to be transcribed by ASR models.

Enhanced VAD adds audio filtering but is slightly slower due to filter processing.
Use Enhanced VAD for noisy environments or when audio quality is poor.
EOF
echo ""

echo "Troubleshooting:"
echo "---------------"
cat << 'EOF'
If you encounter issues:

1. No speech detected:
   - Lower --vad-threshold (try 0.3 or 0.4)
   - Reduce --vad-min-speech (try 0.2)

2. Too many short segments:
   - Increase --vad-min-speech (try 1.0 or 2.0)
   - Decrease --vad-min-silence (try 0.2)

3. Poor performance on noisy audio:
   - Use --use-enhanced-vad instead of --use-vad
   - Check audio quality and consider preprocessing

4. Memory issues:
   - Process smaller batches of files
   - Use CPU instead of GPU if CUDA memory is limited
EOF

echo ""
echo "For more detailed information, see:"
echo "- ems_call/VAD_README.md - Complete documentation"
echo "- ems_call/example_vad_usage.py - Python usage examples"
echo "- ems_call/test_vad_pipeline.py - Installation test" 