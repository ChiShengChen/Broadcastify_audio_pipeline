#!/bin/bash

# Simple LLM Pipeline Script
# This is a simplified version that uses smaller, more reliable models

set -e

echo "=== Simple LLM Pipeline ==="
echo "Date: $(date)"
echo ""

# Configuration
ASR_RESULTS_DIR="/media/meow/One Touch/ems_call/pipeline_results_20250729_033902"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/media/meow/One Touch/ems_call/simple_llm_results"
PYTHON_EXEC="python3"

# Simple model that works on CPU
SIMPLE_MODEL="microsoft/DialoGPT-small"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --asr_results_dir)
            ASR_RESULTS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            SIMPLE_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Simple LLM Pipeline"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --asr_results_dir DIR     Directory containing ASR results"
            echo "  --output_dir DIR          Output directory (default: simple_llm_results)"
            echo "  --model MODEL             Model to use (default: microsoft/DialoGPT-small)"
            echo "  -h, --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate input directory
if [ ! -d "$ASR_RESULTS_DIR" ]; then
    echo "Error: ASR results directory does not exist: $ASR_RESULTS_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  ASR Results: $ASR_RESULTS_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Model: $SIMPLE_MODEL"
echo ""

# Step 1: Filter Whisper results
echo "--- Step 1: Filtering Whisper Results ---"
WHISPER_FILTERED_DIR="$OUTPUT_DIR/whisper_filtered"
mkdir -p "$WHISPER_FILTERED_DIR"

# Find and copy Whisper files
WHISPER_FILES=$(find "$ASR_RESULTS_DIR" -name "*large-v3*.txt")
WHISPER_COUNT=0

for file in $WHISPER_FILES; do
    if [ -f "$file" ]; then
        cp "$file" "$WHISPER_FILTERED_DIR/"
        ((WHISPER_COUNT++))
    fi
done

echo "Filtered $WHISPER_COUNT Whisper files to: $WHISPER_FILTERED_DIR"

if [ $WHISPER_COUNT -eq 0 ]; then
    echo "Warning: No Whisper files found. Checking for any .txt files..."
    
    # Fallback: copy any .txt files
    TXT_FILES=$(find "$ASR_RESULTS_DIR" -name "*.txt")
    TXT_COUNT=0
    
    for file in $TXT_FILES; do
        if [ -f "$file" ]; then
            cp "$file" "$WHISPER_FILTERED_DIR/"
            ((TXT_COUNT++))
        fi
    done
    
    echo "Copied $TXT_COUNT transcript files as fallback"
    
    if [ $TXT_COUNT -eq 0 ]; then
        echo "Error: No transcript files found in $ASR_RESULTS_DIR"
        exit 1
    fi
fi

# Step 2: Medical correction
echo ""
echo "--- Step 2: Medical Term Correction ---"
CORRECTED_DIR="$OUTPUT_DIR/corrected_transcripts"

$PYTHON_EXEC simple_llm_pipeline.py \
    --mode medical_correction \
    --input_dirs "$WHISPER_FILTERED_DIR" \
    --output_dir "$CORRECTED_DIR" \
    --model "$SIMPLE_MODEL" \
    --batch_size 1

if [ $? -eq 0 ]; then
    echo "✓ Medical correction completed"
else
    echo "✗ Medical correction failed"
    exit 1
fi

# Step 3: Emergency page generation
echo ""
echo "--- Step 3: Emergency Page Generation ---"
EMERGENCY_PAGES_DIR="$OUTPUT_DIR/emergency_pages"

$PYTHON_EXEC simple_llm_pipeline.py \
    --mode emergency_page \
    --input_dirs "$CORRECTED_DIR" \
    --output_dir "$EMERGENCY_PAGES_DIR" \
    --model "$SIMPLE_MODEL" \
    --batch_size 1

if [ $? -eq 0 ]; then
    echo "✓ Emergency page generation completed"
else
    echo "✗ Emergency page generation failed"
    exit 1
fi

# Step 4: Generate summary
echo ""
echo "--- Generating Summary ---"
SUMMARY_FILE="$OUTPUT_DIR/simple_pipeline_summary.txt"

{
    echo "Simple LLM Pipeline Summary"
    echo "=========================="
    echo "Date: $(date)"
    echo "ASR Results Directory: $ASR_RESULTS_DIR"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Model Used: $SIMPLE_MODEL"
    echo ""
    echo "Processing Results:"
    
    # Count files in each directory
    if [ -d "$WHISPER_FILTERED_DIR" ]; then
        FILTERED_COUNT=$(find "$WHISPER_FILTERED_DIR" -name "*.txt" | wc -l)
        echo "  - Whisper files processed: $FILTERED_COUNT"
    fi
    
    if [ -d "$CORRECTED_DIR" ]; then
        CORRECTED_COUNT=$(find "$CORRECTED_DIR" -name "*.txt" | wc -l)
        echo "  - Corrected transcripts: $CORRECTED_COUNT"
    fi
    
    if [ -d "$EMERGENCY_PAGES_DIR" ]; then
        PAGE_COUNT=$(find "$EMERGENCY_PAGES_DIR" -name "*.txt" | wc -l)
        echo "  - Emergency pages: $PAGE_COUNT"
    fi
    
    echo ""
    echo "Output Structure:"
    echo "  $OUTPUT_DIR/whisper_filtered/       # Filtered input files"
    echo "  $OUTPUT_DIR/corrected_transcripts/  # Medical term corrected"
    echo "  $OUTPUT_DIR/emergency_pages/        # Generated emergency pages"
    echo "  $OUTPUT_DIR/simple_pipeline_summary.txt  # This summary"
    echo ""
    echo "All results saved to: $OUTPUT_DIR"
    
} > "$SUMMARY_FILE"

echo ""
echo "=== Simple LLM Pipeline Completed ==="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Summary: $SUMMARY_FILE"
echo ""
echo "To view results:"
echo "  ls -la '$OUTPUT_DIR'"
echo "  cat '$SUMMARY_FILE'"