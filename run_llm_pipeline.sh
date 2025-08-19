#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# Disabled to allow pipeline to continue even if some files fail
# set -e

# --- LLM-Enhanced ASR Pipeline Overview ---
# This script extends the basic ASR pipeline with LLM capabilities:
# 1. ASR: Transcribe audio files (using existing pipeline)
# 1.5. WHISPER FILTER: Filter only Whisper results (optional)
# 2. LLM Medical Term Correction: Correct medical terms in ASR results
# 3. LLM Emergency Page Generation: Generate emergency pages from corrected transcripts
# 4. EVALUATION: Compare results against ground truth (optional)

# Example:
# ./run_llm_enhanced_pipeline.sh \
#   --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902" \
#   --output_dir "/media/meow/One Touch/ems_call/llm_results" \
#   --medical_correction_model "BioMistral-7B" \
#   --page_generation_model "BioMistral-7B" \
#   --batch_size 1 \
#   --load_in_8bit \
#   --device "cuda"






# --- User Configuration ---
# Input directory containing ASR results from previous pipeline
ASR_RESULTS_DIR="/media/meow/One Touch/ems_call/pipeline_results_20250814_044143"
# Example: "/media/meow/One Touch/ems_call/pipeline_results_20250729_034836"

# Ground truth file for evaluation (optional)
GROUND_TRUTH_FILE="/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv"
# Example: "/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv"

# Output directory for LLM processing results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR="/media/meow/One Touch/ems_call/llm_results_${TIMESTAMP}"
OUTPUT_DIR=""

# --- LLM Configuration ---
# Available LLM models
AVAILABLE_MODELS=("gpt-oss-20b" "gpt-oss-120b" "BioMistral-7B" "Meditron-7B" "Llama-3-8B-UltraMedica")

# Use local models instead of API calls
USE_LOCAL_MODELS=true

# Model paths (can be overridden with --model_path)
MODEL_PATHS=(
    "gpt-oss-20b:openai/gpt-oss-20b"
    "gpt-oss-120b:openai/gpt-oss-120b"
    "BioMistral-7B:BioMistral/BioMistral-7B"
    "Meditron-7B:epfl-llm/meditron-7b"
    "Llama-3-8B-UltraMedica:/path/to/llama-3-8b-ultramedica"
)

# Default model selections
MEDICAL_CORRECTION_MODEL="gpt-oss-20b"    # Model for medical term correction
PAGE_GENERATION_MODEL="gpt-oss-20b"     # Model for emergency page generation

# --- Feature Switches ---
ENABLE_MEDICAL_CORRECTION=true    # Enable medical term correction
ENABLE_PAGE_GENERATION=true       # Enable emergency page generation
ENABLE_EVALUATION=true            # Enable evaluation of corrected results
ENABLE_WHISPER_FILTER=true        # Enable filtering for Whisper results only

# Device configuration
DEVICE="auto"  # auto, cpu, cuda
LOAD_IN_8BIT=false
LOAD_IN_4BIT=false

# Generation parameters
TEMPERATURE=0.1  # Default temperature for gpt-oss models
MAX_NEW_TOKENS=128  # Default max new tokens for gpt-oss models

# --- Medical Correction Configuration ---
# MEDICAL_CORRECTION_PROMPT="You are a medical transcription specialist. Please correct any medical terms, drug names, anatomical terms, and medical procedures in the following ASR transcript. Maintain the original meaning and context. Only correct obvious medical errors and standardize medical terminology. Return only the corrected transcript without explanations."
# MEDICAL_CORRECTION_PROMPT="You are an expert medical transcription correction system. Your role is to improve noisy, error-prone transcripts generated from EMS radio calls. These transcripts are derived from automatic speech recognition (ASR) and often contain phonetic errors, especially with medication names, clinical terminology, and numerical values.
# Each transcript reflects a real-time communication from EMS personnel to hospital staff, summarizing a patient’s clinical condition, vital signs, and any treatments administered during prehospital care. Use your knowledge of emergency medicine, pharmacology, and EMS protocols to reconstruct the intended meaning of the message as accurately and clearly as possible.
# Guidelines:
# 	1.	Replace misrecognized or phonetically incorrect words and phrases with their most likely intended clinical equivalents.
# 	2.	Express the message in clear, natural language while maintaining the tone and intent of an EMS-to-hospital handoff.
# 	3.	Include all information from the original transcript—ensure your output is complete and continuous.
# 	4.	Use medical abbreviations and shorthand appropriately when they match clinical usage (e.g., “BP” for blood pressure, “ETT” for endotracheal tube).
# 	5.	Apply contextual reasoning to identify and correct drug names, dosages, clinical phrases, and symptoms using common EMS knowledge.
# 	6.	Deliver your output as plain, unstructured text without metadata, formatting, or explanatory notes.
# 	7.	Present the cleaned transcript as a fully corrected version, without gaps, placeholders, or annotations.
# "
MEDICAL_CORRECTION_PROMPT="You are an information extraction model for EMS prearrival radio transcripts in Massachusetts. TASK: Return a single JSON object only. No prose, no code fences, no explanations. SCHEMA (all keys required; values are strings; if unspecified, use \"\"): {\"agency\": \"\", \"unit\": \"\", \"ETA\": \"\", \"age\": \"\", \"sex\": \"\", \"moi\": \"\", \"hr\": \"\", \"rrq\": \"\", \"sbp\": \"\", \"dbp\": \"\", \"end_tidal\": \"\", \"rr\": \"\", \"bgl\": \"\", \"spo2\": \"\", \"o2\": \"\", \"injuries\": \"\", \"ao\": \"\", \"GCS\": \"\", \"LOC\": \"\", \"ac\": \"\", \"treatment\": \"\", \"pregnant\": \"\", \"notes\": \"\"} RULES: Fill fields only with information explicitly stated in the transcript. Do not infer, guess, or normalize beyond obvious medical term corrections. Keep numbers as they are spoken. If multiple possibilities are stated, choose the most explicit; otherwise put \"\". Output must be valid JSON. No trailing commas. OUTPUT FORMAT: A single JSON object exactly matching the SCHEMA keys and order above. TRANSCRIPT:"
# --- Emergency Page Generation Configuration ---
PAGE_GENERATION_PROMPT="You are an emergency medical dispatcher. Based on the following corrected medical transcript, generate a structured emergency page that includes: 1) Patient condition summary, 2) Location details, 3) Required medical resources, 4) Priority level, 5) Key medical information. Format the response as a structured emergency page."

# --- Processing Options ---
BATCH_SIZE=5                      # Number of files to process in parallel
MAX_RETRIES=3                     # Maximum retry attempts for API calls
REQUEST_TIMEOUT=60                # Timeout for API requests in seconds

# Python interpreter to use
# Ensure we're using the correct conda environment with CUDA support
# if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "pytorch_112" ]; then
#     echo "Warning: Not in pytorch_112 environment. Attempting to activate..."
#     source $(conda info --base)/etc/profile.d/conda.sh
#     conda activate pytorch_112
# fi
PYTHON_EXEC="python3"

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
        --ground_truth)
            GROUND_TRUTH_FILE="$2"
            shift 2
            ;;
        --medical_correction_model)
            MEDICAL_CORRECTION_MODEL="$2"
            shift 2
            ;;
        --page_generation_model)
            PAGE_GENERATION_MODEL="$2"
            shift 2
            ;;
        --enable_medical_correction)
            ENABLE_MEDICAL_CORRECTION=true
            shift
            ;;
        --disable_medical_correction)
            ENABLE_MEDICAL_CORRECTION=false
            shift
            ;;
        --enable_page_generation)
            ENABLE_PAGE_GENERATION=true
            shift
            ;;
        --disable_page_generation)
            ENABLE_PAGE_GENERATION=false
            shift
            ;;
        --enable_evaluation)
            ENABLE_EVALUATION=true
            shift
            ;;
        --disable_evaluation)
            ENABLE_EVALUATION=false
            shift
            ;;
        --enable_whisper_filter)
            ENABLE_WHISPER_FILTER=true
            shift
            ;;
        --disable_whisper_filter)
            ENABLE_WHISPER_FILTER=false
            shift
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --load_in_8bit)
            LOAD_IN_8BIT=true
            shift
            ;;
        --load_in_4bit)
            LOAD_IN_4BIT=true
            shift
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --medical_correction_prompt)
            MEDICAL_CORRECTION_PROMPT="$2"
            shift 2
            ;;
        --page_generation_prompt)
            PAGE_GENERATION_PROMPT="$2"
            shift 2
            ;;
        --enable_whisper_filter)
            ENABLE_WHISPER_FILTER=true
            shift
            ;;
        --disable_whisper_filter)
            ENABLE_WHISPER_FILTER=false
            shift
            ;;
        -h|--help)
            echo "LLM-Enhanced ASR Pipeline"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required Options:"
            echo "  --asr_results_dir DIR     Directory containing ASR results from previous pipeline"
            echo ""
            echo "Optional Options:"
            echo "  --output_dir DIR          Output directory for LLM results"
            echo "                           (default: llm_results_YYYYMMDD_HHMMSS)"
            echo "  --ground_truth FILE       Ground truth CSV file for evaluation"
            echo ""
            echo "LLM Model Selection:"
            echo "  --medical_correction_model MODEL  Model for medical term correction"
            echo "                                    Available: gpt-oss-20b, gpt-oss-120b, BioMistral-7B, Meditron-7B, Llama-3-8B-UltraMedica"
            echo "  --page_generation_model MODEL     Model for emergency page generation"
            echo "                                    Available: gpt-oss-20b, gpt-oss-120b, BioMistral-7B, Meditron-7B, Llama-3-8B-UltraMedica"
            echo ""
            echo "Feature Switches:"
            echo "  --enable_medical_correction        Enable medical term correction (default)"
            echo "  --disable_medical_correction       Disable medical term correction"
            echo "  --enable_page_generation           Enable emergency page generation (default)"
            echo "  --disable_page_generation          Disable emergency page generation"
            echo "  --enable_evaluation                Enable evaluation of corrected results (default)"
echo "  --disable_evaluation               Disable evaluation"
echo "  --enable_whisper_filter            Enable filtering for Whisper results only (default)"
echo "  --disable_whisper_filter           Disable Whisper filtering"
            echo "  --enable_whisper_filter            Enable filtering for Whisper results only (default)"
            echo "  --disable_whisper_filter           Disable Whisper filtering"
            echo ""
            echo "LLM Configuration:"
            echo "  --model_path PATH                  Custom model path (optional)"
            echo "  --device DEVICE                    Device to use: auto, cpu, cuda (default: auto)"
            echo "  --load_in_8bit                     Load model in 8-bit quantization"
            echo "  --load_in_4bit                     Load model in 4-bit quantization"
            echo "  --batch_size INT                   Number of files to process in parallel (default: 1 for local models)"
            echo "  --temperature FLOAT                Temperature for generation (default: 0.1 for gpt-oss models)"
            echo "  --max_new_tokens INT               Maximum new tokens to generate (default: 128 for gpt-oss models)"
            echo "  --medical_correction_prompt TEXT   Custom prompt for medical correction"
            echo "  --page_generation_prompt TEXT      Custom prompt for page generation"
            echo ""
            echo "Examples:"
            echo "  # Basic usage with default settings"
            echo "  $0 --asr_results_dir /path/to/asr/results"
            echo ""
            echo "  # Custom models and output directory"
            echo "  $0 --asr_results_dir /path/to/asr/results \\"
            echo "     --output_dir /path/to/output \\"
            echo "     --medical_correction_model BioMistral-7B \\"
            echo "     --page_generation_model Meditron-7B"
            echo ""
            echo "  # Only medical correction, no page generation"
            echo "  $0 --asr_results_dir /path/to/asr/results \\"
            echo "     --disable_page_generation"
            echo ""
            echo "  # Only page generation, no medical correction"
echo "  $0 --asr_results_dir /path/to/asr/results \\"
echo "     --disable_medical_correction"
echo ""
echo "  # Process only Whisper results (default)"
echo "  $0 --asr_results_dir /path/to/asr/results"
echo ""
echo "  # Process all ASR results (disable Whisper filter)"
echo "  $0 --asr_results_dir /path/to/asr/results \\"
echo "     --disable_whisper_filter"
echo ""
echo "  # With evaluation"
echo "  $0 --asr_results_dir /path/to/asr/results \\"
echo "     --ground_truth /path/to/ground_truth.csv"
            echo ""
            echo "  # Process only Whisper results"
            echo "  $0 --asr_results_dir /path/to/asr/results \\"
            echo "     --enable_whisper_filter"
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
if [ -z "$ASR_RESULTS_DIR" ]; then
    echo "Error: --asr_results_dir is required"
    echo "Use -h or --help for usage information"
    exit 1
fi

if [ ! -d "$ASR_RESULTS_DIR" ]; then
    echo "Error: ASR results directory does not exist: $ASR_RESULTS_DIR"
    exit 1
fi

# Validate model selections
validate_model() {
    local model="$1"
    local valid=false
    for available_model in "${AVAILABLE_MODELS[@]}"; do
        if [ "$model" = "$available_model" ]; then
            valid=true
            break
        fi
    done
    if [ "$valid" = false ]; then
        echo "Error: Invalid model '$model'. Available models: ${AVAILABLE_MODELS[*]}"
        exit 1
    fi
}

validate_model "$MEDICAL_CORRECTION_MODEL"
validate_model "$PAGE_GENERATION_MODEL"

# Set default output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    echo "Using default output directory: $OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize error log
ERROR_LOG_FILE="$OUTPUT_DIR/error_analysis.log"
echo "=== LLM-Enhanced Pipeline Error Analysis Log ===" > "$ERROR_LOG_FILE"
echo "Analysis Date: $(date)" >> "$ERROR_LOG_FILE"
echo "Pipeline Output Directory: $OUTPUT_DIR" >> "$ERROR_LOG_FILE"
echo "ASR Results Directory: $ASR_RESULTS_DIR" >> "$ERROR_LOG_FILE"
echo "" >> "$ERROR_LOG_FILE"

# Display configuration
echo "=== LLM-Enhanced ASR Pipeline Configuration ==="
echo "ASR Results Directory: $ASR_RESULTS_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Ground Truth File: $GROUND_TRUTH_FILE"
echo ""
echo "Feature Configuration:"
echo "  - Whisper Filter: $ENABLE_WHISPER_FILTER"
echo "  - Medical Correction: $ENABLE_MEDICAL_CORRECTION"
echo "  - Page Generation: $ENABLE_PAGE_GENERATION"
echo "  - Evaluation: $ENABLE_EVALUATION"
echo ""
echo "LLM Model Configuration:"
echo "  - Medical Correction Model: $MEDICAL_CORRECTION_MODEL"
echo "  - Page Generation Model: $PAGE_GENERATION_MODEL"
echo ""
echo "LLM Configuration:"
echo "  - Use Local Models: $USE_LOCAL_MODELS"
echo "  - Device: $DEVICE"
echo "  - Load in 8-bit: $LOAD_IN_8BIT"
echo "  - Load in 4-bit: $LOAD_IN_4BIT"
echo "  - Temperature: $TEMPERATURE"
echo "  - Max New Tokens: $MAX_NEW_TOKENS"
echo ""
echo "Processing Configuration:"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Max Retries: $MAX_RETRIES"
echo "  - Request Timeout: ${REQUEST_TIMEOUT}s"
echo "==============================================="
echo ""

# --- Step 1: Find ASR Transcripts ---
echo "--- Step 1: Locating ASR Transcripts ---"

# Look for transcripts in various possible locations (prioritize merged results)
TRANSCRIPT_DIRS=()
# Prioritize merged segmented transcripts (complete files after merging segments)
if [ -d "$ASR_RESULTS_DIR/merged_segmented_transcripts" ]; then
    TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR/merged_segmented_transcripts")
# Then merged transcripts (for long audio splits)
elif [ -d "$ASR_RESULTS_DIR/merged_transcripts" ]; then
    TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR/merged_transcripts")
# Finally fall back to raw ASR transcripts (may contain segments)
elif [ -d "$ASR_RESULTS_DIR/asr_transcripts" ]; then
    TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR/asr_transcripts")
fi

# If no specific transcript directory found, check the root
if [ ${#TRANSCRIPT_DIRS[@]} -eq 0 ]; then
    # Check if there are .txt files in the root directory
    if find "$ASR_RESULTS_DIR" -maxdepth 1 -name "*.txt" | grep -q .; then
        TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR")
    fi
fi

if [ ${#TRANSCRIPT_DIRS[@]} -eq 0 ]; then
    echo "Error: No transcript directories found in $ASR_RESULTS_DIR"
    echo "Expected locations:"
    echo "  - $ASR_RESULTS_DIR/asr_transcripts/"
    echo "  - $ASR_RESULTS_DIR/merged_transcripts/"
    echo "  - $ASR_RESULTS_DIR/merged_segmented_transcripts/"
    echo "  - $ASR_RESULTS_DIR/*.txt (root directory)"
    exit 1
fi

echo "Found transcript directories:"
for dir in "${TRANSCRIPT_DIRS[@]}"; do
    echo "  - $dir"
done

# Count total transcript files
TOTAL_TRANSCRIPTS=0
for dir in "${TRANSCRIPT_DIRS[@]}"; do
    COUNT=$(find "$dir" -name "*.txt" | wc -l)
    TOTAL_TRANSCRIPTS=$((TOTAL_TRANSCRIPTS + COUNT))
done

echo "Total transcript files found: $TOTAL_TRANSCRIPTS"
echo ""

# --- Step 1.5: Whisper Filter (Optional) ---
if [ "$ENABLE_WHISPER_FILTER" = true ]; then
    echo "--- Step 1.5: Filtering Whisper Results ---"
    WHISPER_FILTERED_DIR="$OUTPUT_DIR/whisper_filtered"
    mkdir -p "$WHISPER_FILTERED_DIR"
    
    echo "Filtering Whisper (large-v3) results from transcript directories..."
    echo "Input directories: ${TRANSCRIPT_DIRS[*]}"
    echo "Output: $WHISPER_FILTERED_DIR"
    
    # Use the existing filter script
    if [ -f "filter_whisper_results.py" ]; then
        echo "Using existing filter_whisper_results.py script..."
        $PYTHON_EXEC filter_whisper_results.py \
            --input_dir "${TRANSCRIPT_DIRS[0]}" \
            --output_dir "$WHISPER_FILTERED_DIR"
    else
        echo "Creating temporary filter script..."
        # Create a temporary Python script for filtering
        FILTER_SCRIPT="$OUTPUT_DIR/temp_filter_script.py"
        cat > "$FILTER_SCRIPT" << 'EOF'
#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import sys

def filter_whisper_files(input_dirs, output_dir):
    """Filter only Whisper (large-v3) ASR results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    whisper_files = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Warning: Input directory does not exist: {input_dir}")
            continue
        
        # Find all large-v3 files (Whisper results)
        for file_path in input_path.rglob("*.txt"):
            if "large-v3_" in file_path.name:
                whisper_files.append(file_path)
    
    print(f"Found {len(whisper_files)} Whisper (large-v3) files")
    
    # Copy Whisper files to output directory
    for file_path in whisper_files:
        # Create relative path structure
        relative_path = file_path.relative_to(input_path)
        output_file_path = output_path / relative_path
        
        # Create parent directories if needed
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(file_path, output_file_path)
        print(f"Copied: {relative_path}")
    
    print(f"Whisper files copied to: {output_dir}")
    return len(whisper_files)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 script.py <input_dirs> <output_dir>")
        sys.exit(1)
    
    input_dirs = sys.argv[1].split(',')
    output_dir = sys.argv[2]
    
    count = filter_whisper_files(input_dirs, output_dir)
    if count > 0:
        print(f"Successfully filtered {count} Whisper files")
        sys.exit(0)
    else:
        print("No Whisper files found")
        sys.exit(1)
EOF
        
        # Run the filter script
        INPUT_DIRS_STR=$(IFS=','; echo "${TRANSCRIPT_DIRS[*]}")
        $PYTHON_EXEC "$FILTER_SCRIPT" "$INPUT_DIRS_STR" "$WHISPER_FILTERED_DIR"
        
        # Clean up temporary script
        rm -f "$FILTER_SCRIPT"
    fi
    
    WHISPER_FILTER_EXIT_CODE=$?
    
    # Check if whisper filtering produced any output files
    FILTERED_COUNT=$(find "$WHISPER_FILTERED_DIR" -name "*.txt" 2>/dev/null | wc -l)
    
    if [ $WHISPER_FILTER_EXIT_CODE -eq 0 ] || [ $FILTERED_COUNT -gt 0 ]; then
        if [ $WHISPER_FILTER_EXIT_CODE -eq 0 ]; then
            echo "Whisper filtering completed successfully"
        else
            echo "Whisper filtering completed with some issues, but $FILTERED_COUNT files were filtered successfully"
        fi
        echo "Filtered Whisper files saved to: $WHISPER_FILTERED_DIR"
        
        # Update transcript directory for next steps to use filtered results
        TRANSCRIPT_DIRS=("$WHISPER_FILTERED_DIR")
        echo "Filtered transcript files: $FILTERED_COUNT"
    else
        echo "Warning: Whisper filtering failed completely - no output files generated"
        echo "ERROR: Whisper filtering failed completely" >> "$ERROR_LOG_FILE"
        echo "  Input directories: ${TRANSCRIPT_DIRS[*]}" >> "$ERROR_LOG_FILE"
        echo "  Output directory: $WHISPER_FILTERED_DIR" >> "$ERROR_LOG_FILE"
        echo "  Continuing with original transcripts" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
        # Keep using original transcripts for next steps
    fi
else
    echo "--- Skipping Whisper Filter ---"
fi
echo ""


echo ""

# --- Step 2: Medical Term Correction (Optional) ---
if [ "$ENABLE_MEDICAL_CORRECTION" = true ]; then
    echo "--- Step 2: Medical Term Correction ---"
    CORRECTED_TRANSCRIPTS_DIR="$OUTPUT_DIR/corrected_transcripts"
    mkdir -p "$CORRECTED_TRANSCRIPTS_DIR"
    
    echo "Running medical term correction using $MEDICAL_CORRECTION_MODEL..."
    echo "Input directories: ${TRANSCRIPT_DIRS[*]}"
    echo "Output: $CORRECTED_TRANSCRIPTS_DIR"
    
    # Special handling for gpt-oss models
    if [ "$MEDICAL_CORRECTION_MODEL" = "gpt-oss-20b" ] || [ "$MEDICAL_CORRECTION_MODEL" = "gpt-oss-120b" ]; then
        echo "Using specialized gpt-oss handler for $MEDICAL_CORRECTION_MODEL..."
        echo "Temperature: $TEMPERATURE, Max New Tokens: $MAX_NEW_TOKENS"
        
        # Set PyTorch memory allocation config
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        
        # Use the specialized script for gpt-oss models
        if [ "$MEDICAL_CORRECTION_MODEL" = "gpt-oss-20b" ]; then
            SCRIPT_NAME="llm_gpt_oss_20b.py"
        else
            SCRIPT_NAME="llm_gpt_oss_120b.py"
        fi
        
        $PYTHON_EXEC $SCRIPT_NAME \
            "${TRANSCRIPT_DIRS[0]}" \
            "$CORRECTED_TRANSCRIPTS_DIR" \
            "$MEDICAL_CORRECTION_PROMPT" \
            "$TEMPERATURE" \
            "$MAX_NEW_TOKENS" || true
    else
        # Run medical correction with local model for other models
        $PYTHON_EXEC llm_local_models.py \
            --mode medical_correction \
            --input_dirs "${TRANSCRIPT_DIRS[@]}" \
            --output_dir "$CORRECTED_TRANSCRIPTS_DIR" \
            --model "$MEDICAL_CORRECTION_MODEL" \
            --device "$DEVICE" \
            --batch_size "$BATCH_SIZE" \
            --prompt "$MEDICAL_CORRECTION_PROMPT" \
            --error_log "$ERROR_LOG_FILE" \
            $([ "$LOAD_IN_8BIT" = "true" ] && echo "--load_in_8bit") \
            $([ "$LOAD_IN_4BIT" = "true" ] && echo "--load_in_4bit") \
            ${MODEL_PATH:+--model_path "$MODEL_PATH"} || true
    fi
    
    MEDICAL_CORRECTION_EXIT_CODE=$?
    echo "Medical correction exit code: $MEDICAL_CORRECTION_EXIT_CODE"
    echo "DEBUG: Medical correction step completed, continuing to next step..."
    
    # Check if medical correction produced any output files
    CORRECTED_FILE_COUNT=$(find "$CORRECTED_TRANSCRIPTS_DIR" -name "*.txt" 2>/dev/null | wc -l)
    echo "Found $CORRECTED_FILE_COUNT corrected transcript files"
    
    if [ $MEDICAL_CORRECTION_EXIT_CODE -eq 0 ] || [ $CORRECTED_FILE_COUNT -gt 0 ]; then
        if [ $MEDICAL_CORRECTION_EXIT_CODE -eq 0 ]; then
            echo "Medical term correction completed successfully"
        else
            echo "Medical term correction completed with some failures, but $CORRECTED_FILE_COUNT files were processed successfully"
        fi
        echo "Corrected transcripts saved to: $CORRECTED_TRANSCRIPTS_DIR"
        
        # Update transcript directory for next steps to use corrected transcripts
        TRANSCRIPT_DIRS=("$CORRECTED_TRANSCRIPTS_DIR")
    else
        echo "Warning: Medical term correction failed completely - no output files generated"
        echo "ERROR: Medical term correction failed completely" >> "$ERROR_LOG_FILE"
        echo "  Model: $MEDICAL_CORRECTION_MODEL" >> "$ERROR_LOG_FILE"
        echo "  Input directories: ${TRANSCRIPT_DIRS[*]}" >> "$ERROR_LOG_FILE"
        echo "  Output directory: $CORRECTED_TRANSCRIPTS_DIR" >> "$ERROR_LOG_FILE"
        echo "  Continuing with original transcripts" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
        # Keep using original transcripts for next steps
    fi
else
    echo "--- Skipping Medical Term Correction ---"
fi
echo ""

# --- Step 3: Emergency Page Generation (Optional) ---
echo "DEBUG: ENABLE_PAGE_GENERATION = $ENABLE_PAGE_GENERATION"
echo "DEBUG: TRANSCRIPT_DIRS = ${TRANSCRIPT_DIRS[*]}"
if [ "$ENABLE_PAGE_GENERATION" = true ]; then
    echo "--- Step 3: Emergency Page Generation ---"
    EMERGENCY_PAGES_DIR="$OUTPUT_DIR/emergency_pages"
    mkdir -p "$EMERGENCY_PAGES_DIR"
    
    echo "Generating emergency pages using $PAGE_GENERATION_MODEL..."
    echo "Input directories: ${TRANSCRIPT_DIRS[*]}"
    echo "Output: $EMERGENCY_PAGES_DIR"
    
    # Special handling for gpt-oss models
    if [ "$PAGE_GENERATION_MODEL" = "gpt-oss-20b" ] || [ "$PAGE_GENERATION_MODEL" = "gpt-oss-120b" ]; then
        echo "Using specialized gpt-oss handler for $PAGE_GENERATION_MODEL..."
        echo "Temperature: $TEMPERATURE, Max New Tokens: $MAX_NEW_TOKENS"
        
        # Set PyTorch memory allocation config
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        
        # Use the specialized script for gpt-oss models
        if [ "$PAGE_GENERATION_MODEL" = "gpt-oss-20b" ]; then
            SCRIPT_NAME="llm_gpt_oss_20b.py"
        else
            SCRIPT_NAME="llm_gpt_oss_120b.py"
        fi
        
        $PYTHON_EXEC $SCRIPT_NAME \
            "${TRANSCRIPT_DIRS[0]}" \
            "$EMERGENCY_PAGES_DIR" \
            "$PAGE_GENERATION_PROMPT" \
            "$TEMPERATURE" \
            "$MAX_NEW_TOKENS" || true
    else
        # Run emergency page generation with local model for other models
        $PYTHON_EXEC llm_local_models.py \
            --mode emergency_page \
            --input_dirs "${TRANSCRIPT_DIRS[@]}" \
            --output_dir "$EMERGENCY_PAGES_DIR" \
            --model "$PAGE_GENERATION_MODEL" \
            --device "$DEVICE" \
            --batch_size "$BATCH_SIZE" \
            --prompt "$PAGE_GENERATION_PROMPT" \
            --error_log "$ERROR_LOG_FILE" \
            $([ "$LOAD_IN_8BIT" = "true" ] && echo "--load_in_8bit") \
            $([ "$LOAD_IN_4BIT" = "true" ] && echo "--load_in_4bit") \
            ${MODEL_PATH:+--model_path "$MODEL_PATH"} || true
    fi
    
    PAGE_GENERATION_EXIT_CODE=$?
    
    # Check if page generation produced any output files
    PAGE_FILE_COUNT=$(find "$EMERGENCY_PAGES_DIR" -name "*.txt" 2>/dev/null | wc -l)
    
    if [ $PAGE_GENERATION_EXIT_CODE -eq 0 ] || [ $PAGE_FILE_COUNT -gt 0 ]; then
        if [ $PAGE_GENERATION_EXIT_CODE -eq 0 ]; then
            echo "Emergency page generation completed successfully"
        else
            echo "Emergency page generation completed with some failures, but $PAGE_FILE_COUNT pages were generated successfully"
        fi
        echo "Emergency pages saved to: $EMERGENCY_PAGES_DIR"
    else
        echo "Warning: Emergency page generation failed completely - no output files generated"
        echo "ERROR: Emergency page generation failed completely" >> "$ERROR_LOG_FILE"
        echo "  Model: $PAGE_GENERATION_MODEL" >> "$ERROR_LOG_FILE"
        echo "  Input directories: ${TRANSCRIPT_DIRS[*]}" >> "$ERROR_LOG_FILE"
        echo "  Output directory: $EMERGENCY_PAGES_DIR" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
    fi
else
    echo "--- Skipping Emergency Page Generation ---"
fi
echo ""

# --- Step 4: Evaluation (Optional) ---
if [ "$ENABLE_EVALUATION" = true ] && [ -n "$GROUND_TRUTH_FILE" ] && [ -f "$GROUND_TRUTH_FILE" ]; then
    echo "--- Step 4: Evaluation of Corrected Results ---"
    EVALUATION_OUTPUT_FILE="$OUTPUT_DIR/llm_enhanced_evaluation_results.csv"
    
    echo "Evaluating corrected transcripts against ground truth..."
    echo "Ground truth: $GROUND_TRUTH_FILE"
    echo "Transcript directories: ${TRANSCRIPT_DIRS[*]}"
    echo "Output: $EVALUATION_OUTPUT_FILE"
    
    # Run evaluation
    $PYTHON_EXEC evaluate_asr.py \
        --transcript_dirs "${TRANSCRIPT_DIRS[@]}" \
        --ground_truth_file "$GROUND_TRUTH_FILE" \
        --output_file "$EVALUATION_OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully"
        echo "Results saved to: $EVALUATION_OUTPUT_FILE"
    else
        echo "Warning: Evaluation encountered issues"
        echo "ERROR: LLM-enhanced evaluation failed" >> "$ERROR_LOG_FILE"
        echo "  Ground truth file: $GROUND_TRUTH_FILE" >> "$ERROR_LOG_FILE"
        echo "  Transcript directories: ${TRANSCRIPT_DIRS[*]}" >> "$ERROR_LOG_FILE"
        echo "  Output file: $EVALUATION_OUTPUT_FILE" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
    fi
else
    echo "--- Skipping Evaluation ---"
    if [ "$ENABLE_EVALUATION" = false ]; then
        echo "Evaluation disabled by user"
    elif [ -z "$GROUND_TRUTH_FILE" ]; then
        echo "No ground truth file provided"
    elif [ ! -f "$GROUND_TRUTH_FILE" ]; then
        echo "Ground truth file not found: $GROUND_TRUTH_FILE"
    fi
fi
echo ""

# --- Step 5: Generate Summary ---
echo "--- Generating LLM-Enhanced Pipeline Summary ---"
SUMMARY_FILE="$OUTPUT_DIR/llm_enhanced_pipeline_summary.txt"

{
    echo "LLM-Enhanced ASR Pipeline Summary"
    echo "================================="
    echo "Date: $(date)"
    echo "ASR Results Directory: $ASR_RESULTS_DIR"
    echo "Output Directory: $OUTPUT_DIR"
    echo ""
    echo "Configuration:"
    echo "  - Whisper Filter: $ENABLE_WHISPER_FILTER"
    if [ "$ENABLE_WHISPER_FILTER" = true ]; then
        echo "    * Output: $WHISPER_FILTERED_DIR"
    fi
    echo "  - Medical Correction: $ENABLE_MEDICAL_CORRECTION"
    if [ "$ENABLE_MEDICAL_CORRECTION" = true ]; then
        echo "    * Model: $MEDICAL_CORRECTION_MODEL"
        echo "    * Output: $CORRECTED_TRANSCRIPTS_DIR"
    fi
    echo "  - Page Generation: $ENABLE_PAGE_GENERATION"
    if [ "$ENABLE_PAGE_GENERATION" = true ]; then
        echo "    * Model: $PAGE_GENERATION_MODEL"
        echo "    * Output: $EMERGENCY_PAGES_DIR"
    fi
    echo "  - Evaluation: $ENABLE_EVALUATION"
    if [ "$ENABLE_EVALUATION" = true ] && [ -n "$GROUND_TRUTH_FILE" ]; then
        echo "    * Ground Truth: $GROUND_TRUTH_FILE"
        echo "    * Results: $EVALUATION_OUTPUT_FILE"
    fi
    echo ""
    echo "Processing Results:"
    echo "  - Total ASR transcripts: $TOTAL_TRANSCRIPTS"
    
    # Count filtered transcripts
    if [ "$ENABLE_WHISPER_FILTER" = true ] && [ -d "$WHISPER_FILTERED_DIR" ]; then
        FILTERED_COUNT=$(find "$WHISPER_FILTERED_DIR" -name "*.txt" | wc -l)
        echo "  - Whisper filtered transcripts: $FILTERED_COUNT"
    fi
    
    # Count corrected transcripts
    if [ "$ENABLE_MEDICAL_CORRECTION" = true ] && [ -d "$CORRECTED_TRANSCRIPTS_DIR" ]; then
        CORRECTED_COUNT=$(find "$CORRECTED_TRANSCRIPTS_DIR" -name "*.txt" | wc -l)
        echo "  - Corrected transcripts: $CORRECTED_COUNT"
    fi
    
    # Count emergency pages
    if [ "$ENABLE_PAGE_GENERATION" = true ] && [ -d "$EMERGENCY_PAGES_DIR" ]; then
        PAGE_COUNT=$(find "$EMERGENCY_PAGES_DIR" -name "*.txt" | wc -l)
        echo "  - Emergency pages generated: $PAGE_COUNT"
    fi
    
    echo ""
    echo "Output Structure:"
    if [ "$ENABLE_WHISPER_FILTER" = true ]; then
        echo "  $OUTPUT_DIR/whisper_filtered/           # Filtered Whisper transcripts"
    fi
    if [ "$ENABLE_MEDICAL_CORRECTION" = true ]; then
        echo "  $OUTPUT_DIR/corrected_transcripts/     # Medical term corrected transcripts"
    fi
    if [ "$ENABLE_PAGE_GENERATION" = true ]; then
        echo "  $OUTPUT_DIR/emergency_pages/           # Generated emergency pages"
    fi
    if [ "$ENABLE_EVALUATION" = true ] && [ -n "$GROUND_TRUTH_FILE" ]; then
        echo "  $EVALUATION_OUTPUT_FILE                # Evaluation metrics"
    fi
    echo "  $SUMMARY_FILE                             # This summary"
    echo "  $ERROR_LOG_FILE                           # Error analysis"
    echo ""
    echo "All results saved to: $OUTPUT_DIR"
    
} > "$SUMMARY_FILE"

echo "LLM-enhanced pipeline summary saved to: $SUMMARY_FILE"
echo ""

# Check for errors and determine pipeline status
PIPELINE_SUCCESS=true
ERROR_COUNT=0

# Check error log if it exists
if [ -f "$ERROR_LOG_FILE" ]; then
    ERROR_COUNT=$(grep -c "ERROR:" "$ERROR_LOG_FILE" 2>/dev/null || echo "0")
    
    # If there are errors, mark pipeline as failed
    if [ "$ERROR_COUNT" -gt 0 ]; then
        PIPELINE_SUCCESS=false
    fi
fi

# Display final status
if [ "$PIPELINE_SUCCESS" = true ]; then
    echo "=== LLM-Enhanced Pipeline Completed Successfully ==="
    echo ""
    echo "Results structure:"
    if [ "$ENABLE_WHISPER_FILTER" = true ]; then
        echo "  $OUTPUT_DIR/whisper_filtered/           # Filtered Whisper transcripts"
    fi
    if [ "$ENABLE_MEDICAL_CORRECTION" = true ]; then
        echo "  $OUTPUT_DIR/corrected_transcripts/     # Medical term corrected transcripts"
    fi
    if [ "$ENABLE_PAGE_GENERATION" = true ]; then
        echo "  $OUTPUT_DIR/emergency_pages/           # Generated emergency pages"
    fi
    if [ "$ENABLE_EVALUATION" = true ] && [ -n "$GROUND_TRUTH_FILE" ]; then
        echo "  $EVALUATION_OUTPUT_FILE                # Evaluation metrics"
    fi
    echo "  $SUMMARY_FILE                             # Detailed summary"
    echo ""
    echo "Check the summary file for detailed results: $SUMMARY_FILE"
else
    echo "=== LLM-Enhanced Pipeline Completed with Errors ==="
    echo ""
    echo "❌ Pipeline encountered issues during execution."
    echo ""
    echo "Error Summary:"
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "  - Errors detected: $ERROR_COUNT"
    fi
    
    # Count failed files in error log
    if [ -f "$ERROR_LOG_FILE" ]; then
        FAILED_FILES=$(grep -c "FAILED FILE:" "$ERROR_LOG_FILE" 2>/dev/null || echo "0")
        FAILED_FILES=${FAILED_FILES:-0}
        if [ "$FAILED_FILES" -gt 0 ]; then
            echo "  - Failed files: $FAILED_FILES"
            echo ""
            echo "Failed files breakdown:"
            # Show breakdown by error type
            if grep -q "Empty or unreadable transcript" "$ERROR_LOG_FILE" 2>/dev/null; then
                EMPTY_FILES=$(grep -c "Empty or unreadable transcript" "$ERROR_LOG_FILE" 2>/dev/null || echo "0")
                echo "    - Empty/unreadable files: $EMPTY_FILES"
            fi
            if grep -q "Model correction failed\|LLM correction failed" "$ERROR_LOG_FILE" 2>/dev/null; then
                MODEL_FAILURES=$(grep -c "Model correction failed\|LLM correction failed" "$ERROR_LOG_FILE" 2>/dev/null || echo "0")
                echo "    - Model processing failures: $MODEL_FAILURES"
            fi
            if grep -q "Failed to save" "$ERROR_LOG_FILE" 2>/dev/null; then
                SAVE_FAILURES=$(grep -c "Failed to save" "$ERROR_LOG_FILE" 2>/dev/null || echo "0")
                echo "    - File save failures: $SAVE_FAILURES"
            fi
        fi
    fi
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check the error log: $ERROR_LOG_FILE"
    echo "  2. Review the pipeline summary: $SUMMARY_FILE"
    echo "  3. Verify LLM model availability and API endpoints"
    echo "  4. Check network connectivity for API calls"
    echo ""
    echo "Available results (may be incomplete):"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  - Output directory: $OUTPUT_DIR"
        if [ -f "$SUMMARY_FILE" ]; then
            echo "  - Pipeline summary: $SUMMARY_FILE"
        fi
        if [ -f "$ERROR_LOG_FILE" ]; then
            echo "  - Error analysis: $ERROR_LOG_FILE"
        fi
    fi
fi 