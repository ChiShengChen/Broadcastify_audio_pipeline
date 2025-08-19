#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Audio Preprocessing Pipeline Integration ---
# This script integrates audio preprocessing with the existing ASR pipeline
# to ensure all audio files are compatible with all ASR models.

# --- User Configuration ---
# Directory containing the original .wav files to be processed.
AUDIO_DIR="/media/meow/One Touch/ems_call/long_audio_test_dataset"

# Path to the ground truth CSV file for evaluation.
GROUND_TRUTH_FILE="/media/meow/One Touch/ems_call/long_audio_test_dataset/long_audio_ground_truth.csv"

# Output directory for preprocessing results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./preprocessing_pipeline_results_${TIMESTAMP}"

# Path to save the final evaluation report CSV.
OUTPUT_FILE="$OUTPUT_DIR/asr_evaluation_results.csv"

# Preprocessing options
USE_AUDIO_PREPROCESSING=true    # Enable audio preprocessing for model compatibility
PREPROCESSED_AUDIO_DIR=""       # Will be set automatically if preprocessing is enabled

# Python interpreter to use.
PYTHON_EXEC="python3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            AUDIO_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            OUTPUT_FILE="$OUTPUT_DIR/asr_evaluation_results.csv"
            shift 2
            ;;
        --ground_truth)
            GROUND_TRUTH_FILE="$2"
            shift 2
            ;;
        --use-audio-preprocessing)
            USE_AUDIO_PREPROCESSING=true
            shift
            ;;
        --no-audio-preprocessing)
            USE_AUDIO_PREPROCESSING=false
            shift
            ;;
        -h|--help)
            echo "Audio Preprocessing Pipeline Integration"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input_dir DIR              Input directory with audio files"
            echo "  --output_dir DIR             Output directory for results"  
            echo "  --ground_truth FILE          Ground truth CSV file"
            echo "  --use-audio-preprocessing    Enable audio preprocessing for model compatibility"
            echo "  --no-audio-preprocessing     Disable audio preprocessing"
            echo "  -h, --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  # Basic usage with audio preprocessing"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results"
            echo ""
            echo "  # Without audio preprocessing"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --no-audio-preprocessing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$AUDIO_DIR" ]; then
    echo "Error: --input_dir is required"
    echo "Use -h or --help for usage information"
    exit 1
fi

if [ ! -d "$AUDIO_DIR" ]; then
    echo "Error: Input directory does not exist: $AUDIO_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Display configuration
echo "=== Audio Preprocessing Pipeline Configuration ==="
echo "Input directory: $AUDIO_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Ground truth file: $GROUND_TRUTH_FILE"
echo "Use Audio Preprocessing: $USE_AUDIO_PREPROCESSING"
echo "==============================================="
echo ""

# --- Step 1: Audio Preprocessing for Model Compatibility ---
if [ "$USE_AUDIO_PREPROCESSING" = true ]; then
    echo "--- Step 1: Audio Preprocessing for Model Compatibility ---"
    PREPROCESSED_AUDIO_DIR="$OUTPUT_DIR/preprocessed_audio"
    
    echo "Running audio preprocessor to ensure compatibility with all ASR models..."
    echo "Input: $AUDIO_DIR"
    echo "Output: $PREPROCESSED_AUDIO_DIR"
    
    # Run the audio preprocessor
    $PYTHON_EXEC audio_preprocessor.py \
        --input_dir "$AUDIO_DIR" \
        --output_dir "$PREPROCESSED_AUDIO_DIR" \
        --summary_file "preprocessing_summary.json" \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "Audio preprocessing completed successfully"
        echo "Preprocessed audio saved to: $PREPROCESSED_AUDIO_DIR"
        
        # Check preprocessing summary
        SUMMARY_FILE="$PREPROCESSED_AUDIO_DIR/preprocessing_summary.json"
        if [ -f "$SUMMARY_FILE" ]; then
            echo ""
            echo "Preprocessing Summary:"
            if command -v jq > /dev/null 2>&1; then
                TOTAL_FILES=$(jq -r '.total_files' "$SUMMARY_FILE" 2>/dev/null || echo "N/A")
                echo "  - Total files processed: $TOTAL_FILES"
                
                echo "  - Model statistics:"
                for model in large-v3 canary-1b parakeet-tdt-0.6b-v2 wav2vec-xls-r; do
                    SUCCESS_RATE=$(jq -r ".model_stats.\"$model\".success_rate" "$SUMMARY_FILE" 2>/dev/null || echo "N/A")
                    OUTPUT_FILES=$(jq -r ".model_stats.\"$model\".total_output_files" "$SUMMARY_FILE" 2>/dev/null || echo "N/A")
                    echo "    * $model: $SUCCESS_RATE ($OUTPUT_FILES files)"
                done
            else
                echo "  - Summary available in: $SUMMARY_FILE"
            fi
        fi
    else
        echo "Warning: Audio preprocessing failed, using original files"
        echo "ERROR: Audio preprocessing failed" >> "$OUTPUT_DIR/error.log"
        echo "  Input directory: $AUDIO_DIR" >> "$OUTPUT_DIR/error.log"
        echo "  Output directory: $PREPROCESSED_AUDIO_DIR" >> "$OUTPUT_DIR/error.log"
        echo "  Using original audio files instead" >> "$OUTPUT_DIR/error.log"
        echo "" >> "$OUTPUT_DIR/error.log"
        PREPROCESSED_AUDIO_DIR="$AUDIO_DIR"
    fi
else
    echo "--- Skipping Audio Preprocessing ---"
    PREPROCESSED_AUDIO_DIR="$AUDIO_DIR"
fi
echo ""

# --- Step 2: Run ASR Pipeline with Preprocessed Audio ---
echo "--- Step 2: Running ASR Pipeline with Preprocessed Audio ---"

# Create a modified version of the original pipeline script
MODIFIED_PIPELINE_SCRIPT="$OUTPUT_DIR/run_modified_pipeline.sh"

# Create the modified pipeline script
cat > "$MODIFIED_PIPELINE_SCRIPT" << 'EOF'
#!/bin/bash

# Modified pipeline script that uses preprocessed audio
set -e

# Use the preprocessed audio directory
AUDIO_DIR="$1"
GROUND_TRUTH_FILE="$2"
OUTPUT_DIR="$3"

# Run the original pipeline with preprocessed audio
./run_pipeline.sh \
    --input_dir "$AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ground_truth "$GROUND_TRUTH_FILE" \
    --use-vad \
    --use-long-audio-split \
    --preprocess-ground-truth \
    --use-enhanced-preprocessor
EOF

chmod +x "$MODIFIED_PIPELINE_SCRIPT"

echo "Running ASR pipeline with preprocessed audio..."
echo "Audio directory: $PREPROCESSED_AUDIO_DIR"
echo "Ground truth: $GROUND_TRUTH_FILE"
echo "Output: $OUTPUT_DIR"

# Run the modified pipeline
"$MODIFIED_PIPELINE_SCRIPT" "$PREPROCESSED_AUDIO_DIR" "$GROUND_TRUTH_FILE" "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "ASR pipeline completed successfully"
else
    echo "Warning: ASR pipeline encountered issues"
    echo "ERROR: ASR pipeline failed" >> "$OUTPUT_DIR/error.log"
    echo "  Audio directory: $PREPROCESSED_AUDIO_DIR" >> "$OUTPUT_DIR/error.log"
    echo "  Ground truth: $GROUND_TRUTH_FILE" >> "$OUTPUT_DIR/error.log"
    echo "  Output directory: $OUTPUT_DIR" >> "$OUTPUT_DIR/error.log"
    echo "" >> "$OUTPUT_DIR/error.log"
fi
echo ""

# --- Step 3: Generate Integration Summary ---
echo "--- Step 3: Generating Integration Summary ---"
INTEGRATION_SUMMARY_FILE="$OUTPUT_DIR/integration_summary.txt"

{
    echo "Audio Preprocessing Pipeline Integration Summary"
    echo "=============================================="
    echo "Date: $(date)"
    echo "Input Directory: $AUDIO_DIR"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Ground Truth File: $GROUND_TRUTH_FILE"
    echo ""
    echo "Configuration:"
    echo "  - Audio Preprocessing: $USE_AUDIO_PREPROCESSING"
    if [ "$USE_AUDIO_PREPROCESSING" = true ]; then
        echo "  - Preprocessed Audio Directory: $PREPROCESSED_AUDIO_DIR"
    fi
    echo ""
    
    # Count input files
    INPUT_COUNT=$(find "$AUDIO_DIR" -name "*.wav" -o -name "*.mp3" -o -name "*.m4a" -o -name "*.flac" | wc -l)
    echo "Input Files: $INPUT_COUNT audio files"
    
    # Preprocessing results
    if [ "$USE_AUDIO_PREPROCESSING" = true ] && [ -f "$PREPROCESSED_AUDIO_DIR/preprocessing_summary.json" ]; then
        echo ""
        echo "Audio Preprocessing Results:"
        if command -v jq > /dev/null 2>&1; then
            TOTAL_FILES=$(jq -r '.total_files' "$PREPROCESSED_AUDIO_DIR/preprocessing_summary.json" 2>/dev/null || echo "N/A")
            echo "  - Total files processed: $TOTAL_FILES"
            
            for model in large-v3 canary-1b parakeet-tdt-0.6b-v2 wav2vec-xls-r; do
                SUCCESS_RATE=$(jq -r ".model_stats.\"$model\".success_rate" "$PREPROCESSED_AUDIO_DIR/preprocessing_summary.json" 2>/dev/null || echo "N/A")
                OUTPUT_FILES=$(jq -r ".model_stats.\"$model\".total_output_files" "$PREPROCESSED_AUDIO_DIR/preprocessing_summary.json" 2>/dev/null || echo "N/A")
                echo "  - $model: $SUCCESS_RATE ($OUTPUT_FILES files)"
            done
        else
            echo "  - Preprocessing summary available in: $PREPROCESSED_AUDIO_DIR/preprocessing_summary.json"
        fi
    fi
    
    # ASR results
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "ASR Evaluation Results:"
        echo "  - Evaluation report: $OUTPUT_FILE"
    fi
    
    # Model file analysis results
    MODEL_ANALYSIS_FILE="$OUTPUT_DIR/model_file_analysis.txt"
    if [ -f "$MODEL_ANALYSIS_FILE" ]; then
        echo ""
        echo "Model File Analysis:"
        echo "  - Model file analysis: $MODEL_ANALYSIS_FILE"
    fi
    
    # Error analysis results
    ERROR_LOG_FILE="$OUTPUT_DIR/error_analysis.log"
    if [ -f "$ERROR_LOG_FILE" ]; then
        echo ""
        echo "Error Analysis:"
        echo "  - Error log: $ERROR_LOG_FILE"
        
        # Count errors and warnings
        ERROR_COUNT=$(grep -c "\[ERROR\]" "$ERROR_LOG_FILE" 2>/dev/null || echo "0")
        WARNING_COUNT=$(grep -c "\[WARNING\]" "$ERROR_LOG_FILE" 2>/dev/null || echo "0")
        
        if [ "$ERROR_COUNT" -gt 0 ] || [ "$WARNING_COUNT" -gt 0 ]; then
            echo "  - Errors found: $ERROR_COUNT"
            echo "  - Warnings found: $WARNING_COUNT"
            echo "  - Check error log for detailed analysis"
        else
            echo "  - No errors or warnings detected"
        fi
    fi
    
    echo ""
    echo "All results saved to: $OUTPUT_DIR"
    
} > "$INTEGRATION_SUMMARY_FILE"

echo "Integration summary saved to: $INTEGRATION_SUMMARY_FILE"
echo ""

# Check for errors and determine pipeline status
PIPELINE_SUCCESS=true
ERROR_COUNT=0
WARNING_COUNT=0

# Check error log if it exists
if [ -f "$OUTPUT_DIR/error.log" ]; then
    ERROR_COUNT=$(grep -c "ERROR" "$OUTPUT_DIR/error.log" 2>/dev/null || echo "0")
    WARNING_COUNT=$(grep -c "WARNING" "$OUTPUT_DIR/error.log" 2>/dev/null || echo "0")
    
    # If there are errors, mark pipeline as failed
    if [ "$ERROR_COUNT" -gt 0 ]; then
        PIPELINE_SUCCESS=false
    fi
fi

# Check if critical files exist
if [ ! -f "$OUTPUT_FILE" ]; then
    PIPELINE_SUCCESS=false
fi

# Display final status
if [ "$PIPELINE_SUCCESS" = true ]; then
    echo "=== Audio Preprocessing Pipeline Completed Successfully ==="
    echo ""
    echo "Results structure:"
    if [ "$USE_AUDIO_PREPROCESSING" = true ]; then
        echo "  $OUTPUT_DIR/preprocessed_audio/     # Preprocessed audio files"
    fi
    echo "  $OUTPUT_DIR/long_audio_segments/       # Long audio split segments"
    echo "  $OUTPUT_DIR/vad_segments/              # VAD extracted speech segments"
    echo "  $OUTPUT_DIR/asr_transcripts/           # ASR transcription results"
    echo "  $OUTPUT_DIR/merged_transcripts/        # Merged transcripts for evaluation"
    echo "  $OUTPUT_FILE                           # Evaluation metrics"
    echo "  $MODEL_ANALYSIS_FILE                   # Model file processing analysis"
    echo "  $INTEGRATION_SUMMARY_FILE              # Integration summary"
    echo ""
    echo "Check the integration summary for detailed results: $INTEGRATION_SUMMARY_FILE"
    
    # Show warnings if any
    if [ "$WARNING_COUNT" -gt 0 ]; then
        echo ""
        echo "??  Note: $WARNING_COUNT warnings were detected during processing."
        echo "   Check $OUTPUT_DIR/error.log for details."
    fi
else
    echo "=== Audio Preprocessing Pipeline Completed with Errors ==="
    echo ""
    echo "? Pipeline encountered issues during execution."
    echo ""
    echo "Error Summary:"
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "  - Errors detected: $ERROR_COUNT"
    fi
    if [ "$WARNING_COUNT" -gt 0 ]; then
        echo "  - Warnings detected: $WARNING_COUNT"
    fi
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check the error log: $OUTPUT_DIR/error.log"
    echo "  2. Review the integration summary: $INTEGRATION_SUMMARY_FILE"
    echo "  3. Verify input files and configuration"
    echo "  4. Check system resources (disk space, memory)"
    echo ""
    echo "Available results (may be incomplete):"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  - Output directory: $OUTPUT_DIR"
        if [ -f "$INTEGRATION_SUMMARY_FILE" ]; then
            echo "  - Integration summary: $INTEGRATION_SUMMARY_FILE"
        fi
        if [ -f "$OUTPUT_DIR/error.log" ]; then
            echo "  - Error log: $OUTPUT_DIR/error.log"
        fi
    fi
fi 