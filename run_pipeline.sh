#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Enhanced Workflow Overview ---
# This script automates an enhanced ASR evaluation process with optional VAD:
# 1. OPTIONAL VAD: Extract speech segments from audio files
# 2. ASR: Transcribe audio (original files or VAD segments)  
# 3. EVALUATION: Compare transcripts against ground truth

# --- User Configuration ---
# Directory containing the .wav files to be processed.
# AUDIO_DIR="/media/meow/One Touch/ems_call/random_samples_1_preprocessed"
AUDIO_DIR="/media/meow/One Touch/ems_call/long_audio_test_dataset"


# Path to the ground truth CSV file for evaluation.
# Must contain 'Filename' and 'transcript' columns.
# GROUND_TRUTH_FILE="/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv"
GROUND_TRUTH_FILE="/media/meow/One Touch/ems_call/long_audio_test_dataset/long_audio_ground_truth.csv"


# Output directory for all processing results (with timestamp)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/media/meow/One Touch/ems_call/pipeline_results_${TIMESTAMP}"

# Path to save the final evaluation report CSV.
OUTPUT_FILE="$OUTPUT_DIR/asr_evaluation_results.csv"

USE_VAD=true                    # Enable VAD preprocessing
USE_LONG_AUDIO_SPLIT=true      # Enable long audio splitting to prevent OOM
MAX_SEGMENT_DURATION=120.0      # Maximum segment duration in seconds (2 minutes)


#### DO NOT CHANGE THESE OPTIONS ####
# Processing options
VAD_SPEECH_THRESHOLD=0.5        # VAD speech detection threshold
VAD_MIN_SPEECH_DURATION=0.5     # Minimum speech segment duration
VAD_MIN_SILENCE_DURATION=0.3    # Minimum silence between segments

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
        --use-vad)
            USE_VAD=true
            shift
            ;;
        --use-enhanced-vad)
            USE_VAD=true
            USE_ENHANCED_VAD=false
            shift
            ;;
        --vad-threshold)
            VAD_SPEECH_THRESHOLD="$2"
            shift 2
            ;;
        --vad-min-speech)
            VAD_MIN_SPEECH_DURATION="$2"
            shift 2
            ;;
        --vad-min-silence)
            VAD_MIN_SILENCE_DURATION="$2"
            shift 2
            ;;
        --use-long-audio-split)
            USE_LONG_AUDIO_SPLIT=true
            shift
            ;;
        --max-segment-duration)
            MAX_SEGMENT_DURATION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Enhanced ASR Pipeline with Optional VAD"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input_dir DIR              Input directory with audio files"
            echo "  --output_dir DIR             Output directory for results"  
            echo "  --ground_truth FILE          Ground truth CSV file"
            echo "  --use-vad                    Enable basic VAD preprocessing"
            echo "  --use-enhanced-vad           Enable enhanced VAD with filters"
            echo "  --vad-threshold FLOAT        VAD speech threshold (default: 0.5)"
            echo "  --vad-min-speech FLOAT       Min speech duration (default: 0.5s)"
            echo "  --vad-min-silence FLOAT      Min silence duration (default: 0.3s)"
            echo "  --use-long-audio-split       Enable long audio splitting to prevent OOM"
            echo "  --max-segment-duration FLOAT Max segment duration in seconds (default: 120.0)"
            echo "  -h, --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  # Original workflow (no VAD)"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results"
            echo ""
            echo "  # With basic VAD"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --use-vad"
            echo ""
            echo "  # With enhanced VAD"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --use-enhanced-vad"
            echo ""
            echo "  # With long audio splitting to prevent OOM"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --use-long-audio-split"
            echo ""
            echo "  # With custom segment duration"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --use-long-audio-split --max-segment-duration 90"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done
USE_ENHANCED_VAD=false  
# Create output directory
mkdir -p "$OUTPUT_DIR"
# --------------------

# Display configuration
echo "=== Enhanced ASR Pipeline Configuration ==="
echo "Input directory: $AUDIO_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Ground truth file: $GROUND_TRUTH_FILE"
echo "Use VAD: $USE_VAD"
if [ "$USE_VAD" = true ]; then
    echo "Enhanced VAD: $USE_ENHANCED_VAD"
    echo "VAD Parameters:"
    echo "  - Speech threshold: $VAD_SPEECH_THRESHOLD"
    echo "  - Min speech duration: ${VAD_MIN_SPEECH_DURATION}s"
    echo "  - Min silence duration: ${VAD_MIN_SILENCE_DURATION}s"
fi
echo "Use Long Audio Split: $USE_LONG_AUDIO_SPLIT"
if [ "$USE_LONG_AUDIO_SPLIT" = true ]; then
    echo "Long Audio Split Parameters:"
    echo "  - Max segment duration: ${MAX_SEGMENT_DURATION}s"
fi
echo "==============================================="
echo ""

# --- Step 1: Install Dependencies ---
# echo "--- Installing required Python libraries ---"
# if [ "$USE_ENHANCED_VAD" = true ]; then
#     $PYTHON_EXEC -m pip install --quiet pandas jiwer torch transformers torchaudio "nemo_toolkit[asr]" openai-whisper tqdm scipy numpy pathlib2 soundfile pydub
# else
#     $PYTHON_EXEC -m pip install --quiet pandas jiwer torch transformers torchaudio "nemo_toolkit[asr]" openai-whisper tqdm pathlib2 soundfile pydub
# fi
# echo "Dependencies installation complete"
# echo ""

# --- Step 2: Long Audio Splitting (Optional) ---
if [ "$USE_LONG_AUDIO_SPLIT" = true ]; then
    echo "--- Step 2: Long Audio Splitting ---"
    LONG_AUDIO_OUTPUT_DIR="$OUTPUT_DIR/long_audio_segments"
    
    echo "Running Long Audio Splitter to prevent OOM issues..."
    $PYTHON_EXEC long_audio_splitter.py \
        --input_dir "$AUDIO_DIR" \
        --output_dir "$LONG_AUDIO_OUTPUT_DIR" \
        --max_duration "$MAX_SEGMENT_DURATION" \
        --speech_threshold "$VAD_SPEECH_THRESHOLD" \
        --min_speech_duration "$VAD_MIN_SPEECH_DURATION" \
        --min_silence_duration "$VAD_MIN_SILENCE_DURATION"
    
    echo "Long audio splitting completed"
    echo "Split segments saved to: $LONG_AUDIO_OUTPUT_DIR"
    echo ""
    
    # Set input directory for next steps to split segments
    PROCESSING_INPUT_DIR="$LONG_AUDIO_OUTPUT_DIR"
else
    PROCESSING_INPUT_DIR="$AUDIO_DIR"
fi

# --- Step 3: VAD Processing (Optional) ---
if [ "$USE_VAD" = true ]; then
    echo "--- Step 3: VAD Processing ---"
    VAD_OUTPUT_DIR="$OUTPUT_DIR/vad_segments"
    
    if [ "$USE_ENHANCED_VAD" = true ]; then
        echo "Running Enhanced VAD with audio filters..."
        $PYTHON_EXEC enhanced_vad_pipeline.py \
            --input_dir "$AUDIO_DIR" \
            --output_dir "$VAD_OUTPUT_DIR" \
            --speech_threshold "$VAD_SPEECH_THRESHOLD" \
            --min_speech_duration "$VAD_MIN_SPEECH_DURATION" \
            --min_silence_duration "$VAD_MIN_SILENCE_DURATION"
    else
        echo "Running Basic VAD..."
        $PYTHON_EXEC vad_pipeline.py \
            --input_dir "$AUDIO_DIR" \
            --output_dir "$VAD_OUTPUT_DIR" \
            --speech_threshold "$VAD_SPEECH_THRESHOLD" \
            --min_speech_duration "$VAD_MIN_SPEECH_DURATION" \
            --min_silence_duration "$VAD_MIN_SILENCE_DURATION"
    fi
    
    echo "VAD processing completed"
    echo "Speech segments saved to: $VAD_OUTPUT_DIR"
    echo ""
    
    # Set ASR input directory to VAD output
    ASR_INPUT_DIR="$VAD_OUTPUT_DIR"
else
    echo "--- Skipping VAD (processing original files) ---"
    ASR_INPUT_DIR="$PROCESSING_INPUT_DIR"
fi

# --- Step 4: ASR Processing ---
echo "--- Step 4: ASR Transcription ---"
ASR_OUTPUT_DIR="$OUTPUT_DIR/asr_transcripts"
mkdir -p "$ASR_OUTPUT_DIR"

if [ "$USE_VAD" = true ]; then
    echo "Running ASR on VAD processed files from: $ASR_INPUT_DIR"
    
    # Find all VAD processed files (concatenated segments)
    vad_files=()
    while IFS= read -r -d '' file; do
        vad_files+=("$file")
    done < <(find "$ASR_INPUT_DIR" -name "*_vad.wav" -print0)
    
    if [ ${#vad_files[@]} -eq 0 ]; then
        echo "No VAD processed files found (looking for *_vad.wav files)"
        echo "Checking for individual segments..."
        
        # Fallback: look for individual segment files (legacy mode)
        segment_files=()
        while IFS= read -r -d '' file; do
            segment_files+=("$file")
        done < <(find "$ASR_INPUT_DIR" -name "segment_*.wav" -print0)
        
        if [ ${#segment_files[@]} -eq 0 ]; then
            echo "Error: No VAD output files found in $ASR_INPUT_DIR"
            echo "Expected either *_vad.wav (concatenated) or segment_*.wav files"
            exit 1
        else
            echo "Found ${#segment_files[@]} individual segment files (using legacy processing)"
            # Use the old segment-based processing
            python3 -c "
import os
import glob
import subprocess
import sys
from pathlib import Path

asr_input_dir = '$ASR_INPUT_DIR'
asr_output_dir = '$ASR_OUTPUT_DIR'
python_exec = '$PYTHON_EXEC'

# Find all segment files from VAD output
segment_files = []
for root, dirs, files in os.walk(asr_input_dir):
    for file in files:
        if file.startswith('segment_') and file.endswith('.wav'):
            segment_files.append(os.path.join(root, file))

if not segment_files:
    print('No VAD segments found for ASR processing')
    sys.exit(1)

print(f'Found {len(segment_files)} VAD segments to transcribe')

# Create temporary directory with all segments for batch processing
temp_dir = os.path.join(asr_output_dir, 'temp_segments')
os.makedirs(temp_dir, exist_ok=True)

# Copy segments to temp directory with organized names
file_mapping = {}
for segment_path in segment_files:
    # Extract original filename and segment info
    relative_path = os.path.relpath(segment_path, asr_input_dir)
    parts = relative_path.split(os.sep)
    if len(parts) >= 2:
        original_name = parts[0]  # Original filename (without extension)
        segment_name = parts[1]   # segment_XXX.wav
        
        # Create new name: original_name_segment_XXX.wav
        new_name = f'{original_name}_{segment_name}'
        temp_path = os.path.join(temp_dir, new_name)
        
        # Copy file
        import shutil
        shutil.copy2(segment_path, temp_path)
        
        # Store mapping for later consolidation
        if original_name not in file_mapping:
            file_mapping[original_name] = []
        file_mapping[original_name].append((new_name, segment_name))

print(f'Prepared {len(os.listdir(temp_dir))} segments for ASR processing')

# Run ASR on all segments
print('Running ASR transcription...')
result = subprocess.run([python_exec, 'run_all_asrs.py', temp_dir], 
                       capture_output=True, text=True)
if result.returncode != 0:
    print(f'ASR processing failed: {result.stderr}')
    sys.exit(1)

print('ASR transcription completed')

# Consolidate transcripts by original file
print('Consolidating transcripts by original file...')
consolidated_dir = os.path.join(asr_output_dir, 'consolidated')
os.makedirs(consolidated_dir, exist_ok=True)

# Get available models by checking transcript files
available_models = set()
for file in os.listdir(temp_dir):
    if file.endswith('.txt'):
        model_name = file.split('_')[0]
        available_models.add(model_name)

print(f'Found models: {sorted(available_models)}')

# Consolidate transcripts for each model and original file
for original_name, segments in file_mapping.items():
    for model in available_models:
        # Collect all transcript segments for this original file and model
        transcript_parts = []
        
        for new_name, segment_name in sorted(segments, key=lambda x: x[1]):
            transcript_file = os.path.join(temp_dir, f'{model}_{new_name.replace(\".wav\", \".txt\")}')
            if os.path.exists(transcript_file):
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        transcript_parts.append(content)
        
        # Save consolidated transcript
        if transcript_parts:
            consolidated_text = ' '.join(transcript_parts)
            output_file = os.path.join(consolidated_dir, f'{model}_{original_name}.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(consolidated_text)

print(f'Consolidated transcripts saved to: {consolidated_dir}')
print(f'Processed {len(file_mapping)} original files with {len(available_models)} models')
"
            # Set transcript directory for evaluation
            TRANSCRIPT_DIR="$ASR_OUTPUT_DIR/consolidated"
        fi
    else
        echo "Found ${#vad_files[@]} VAD processed files (concatenated segments)"
        
        # Create a temporary directory and copy VAD processed files for ASR
        temp_vad_dir="$ASR_OUTPUT_DIR/temp_vad_files"
        mkdir -p "$temp_vad_dir"
        
        # Copy VAD processed files to temp directory with simplified names
        declare -A filename_mapping
        for vad_file in "${vad_files[@]}"; do
            # Extract base filename (remove _vad.wav suffix)
            base_name=$(basename "$vad_file" "_vad.wav")
            
            # Copy to temp directory with original base name
            temp_file="$temp_vad_dir/${base_name}.wav"
            cp "$vad_file" "$temp_file"
            
            # Store mapping for later reference
            filename_mapping["$base_name"]="$vad_file"
            echo "  - Prepared ${base_name}.wav for ASR"
        done
        
        # Run ASR on VAD processed files
        echo "Running ASR transcription on concatenated VAD files..."
        $PYTHON_EXEC run_all_asrs.py "$temp_vad_dir"
        
        # Move transcripts to final location with proper naming
        echo "Organizing transcripts..."
        for base_name in "${!filename_mapping[@]}"; do
            # Find all transcript files for this base name
            for transcript in "$temp_vad_dir"/*.txt; do
                if [[ $(basename "$transcript") == *"${base_name}.txt" ]]; then
                    # Move to final location
                    mv "$transcript" "$ASR_OUTPUT_DIR/"
                    echo "  - Saved transcript: $(basename "$transcript")"
                fi
            done
        done
        
        # Clean up temporary directory
        rm -rf "$temp_vad_dir"
        
        # Set transcript directory for evaluation
        TRANSCRIPT_DIR="$ASR_OUTPUT_DIR"
        
        echo "VAD-based ASR processing completed"
        echo "Note: Transcripts are from concatenated speech segments with original timing preserved in VAD metadata"
    fi
    
else
    echo "Running ASR on original files from: $ASR_INPUT_DIR"
    # Copy original files to ASR output for processing
    cp "$ASR_INPUT_DIR"/*.wav "$ASR_OUTPUT_DIR/" 2>/dev/null || true
    
    # Run ASR on original files
    $PYTHON_EXEC run_all_asrs.py "$ASR_OUTPUT_DIR"
    
    # Set transcript directory for evaluation
    TRANSCRIPT_DIR="$ASR_OUTPUT_DIR"
fi

echo "ASR transcription completed"
echo "Transcripts saved to: $TRANSCRIPT_DIR"
echo ""

# --- Step 5: Merge Split Transcripts (if long audio splitting was used) ---
if [ "$USE_LONG_AUDIO_SPLIT" = true ]; then
    echo "--- Step 5: Merging Split Transcripts ---"
    MERGED_TRANSCRIPTS_DIR="$OUTPUT_DIR/merged_transcripts"
    
    echo "Merging split transcripts for WER calculation..."
    $PYTHON_EXEC merge_split_transcripts.py \
        --input_dir "$TRANSCRIPT_DIR" \
        --output_dir "$MERGED_TRANSCRIPTS_DIR" \
        --metadata_dir "$LONG_AUDIO_OUTPUT_DIR"
    
    echo "Transcript merging completed"
    echo "Merged transcripts saved to: $MERGED_TRANSCRIPTS_DIR"
    echo ""
    
    # Set transcript directory for evaluation to merged transcripts
    TRANSCRIPT_DIR="$MERGED_TRANSCRIPTS_DIR"
fi

# --- Step 6: Evaluation ---
echo "--- Step 6: Evaluating ASR Results ---"
if [[ -f "$GROUND_TRUTH_FILE" && -d "$TRANSCRIPT_DIR" ]]; then
    $PYTHON_EXEC evaluate_asr.py \
        --transcript_dirs "$TRANSCRIPT_DIR" \
        --ground_truth_file "$GROUND_TRUTH_FILE" \
        --output_file "$OUTPUT_FILE"
    
    echo "Evaluation completed successfully"
    echo "Results saved to: $OUTPUT_FILE"
else
    echo "Skipping evaluation - missing ground truth file or transcript directory"
fi
echo ""

# --- Step 6.5: Model File Analysis ---
echo "--- Step 6.5: Analyzing Model File Processing ---"
MODEL_ANALYSIS_FILE="$OUTPUT_DIR/model_file_analysis.txt"

if [[ -d "$TRANSCRIPT_DIR" ]]; then
    echo "Running detailed model file analysis..."
    $PYTHON_EXEC analyze_model_files.py \
        --transcript_dir "$TRANSCRIPT_DIR" \
        --ground_truth_file "$GROUND_TRUTH_FILE" \
        --output_file "$MODEL_ANALYSIS_FILE"
    
    echo "Model file analysis completed"
    echo "Detailed analysis saved to: $MODEL_ANALYSIS_FILE"
else
    echo "Skipping model file analysis - transcript directory not found"
fi
echo ""

# --- Step 7: Generate Summary ---
echo "--- Generating Pipeline Summary ---"
SUMMARY_FILE="$OUTPUT_DIR/pipeline_summary.txt"

{
    echo "Enhanced ASR Pipeline Summary"
    echo "============================="
    echo "Date: $(date)"
    echo "Input Directory: $AUDIO_DIR" 
    echo "Output Directory: $OUTPUT_DIR"
    echo ""
    echo "Configuration:"
    echo "  - VAD Preprocessing: $USE_VAD"
    echo "  - Enhanced VAD: $USE_ENHANCED_VAD"
    if [ "$USE_VAD" = true ]; then
        echo "  - VAD Parameters:"
        echo "    * Speech threshold: $VAD_SPEECH_THRESHOLD"
        echo "    * Min speech duration: ${VAD_MIN_SPEECH_DURATION}s"
        echo "    * Min silence duration: ${VAD_MIN_SILENCE_DURATION}s"
    fi
    echo "  - Long Audio Split: $USE_LONG_AUDIO_SPLIT"
    if [ "$USE_LONG_AUDIO_SPLIT" = true ]; then
        echo "  - Long Audio Split Parameters:"
        echo "    * Max segment duration: ${MAX_SEGMENT_DURATION}s"
    fi
    echo ""
    
    # Count input files
    INPUT_COUNT=$(find "$AUDIO_DIR" -name "*.wav" | wc -l)
    echo "Input Files: $INPUT_COUNT audio files"
    
    # Long audio split results
    if [ "$USE_LONG_AUDIO_SPLIT" = true ] && [ -f "$OUTPUT_DIR/long_audio_segments/processing_summary.json" ]; then
        echo ""
        echo "Long Audio Split Results:"
        if command -v jq > /dev/null 2>&1; then
            TOTAL_FILES=$(jq -r '.total_files' "$OUTPUT_DIR/long_audio_segments/processing_summary.json" 2>/dev/null || echo "N/A")
            SPLIT_FILES=$(jq -r '.split_files' "$OUTPUT_DIR/long_audio_segments/processing_summary.json" 2>/dev/null || echo "N/A")
            UNSPLIT_FILES=$(jq -r '.unsplit_files' "$OUTPUT_DIR/long_audio_segments/processing_summary.json" 2>/dev/null || echo "N/A")
            echo "  - Total files processed: $TOTAL_FILES"
            echo "  - Files split: $SPLIT_FILES"
            echo "  - Files unchanged: $UNSPLIT_FILES"
        else
            echo "  - Long audio split summary available in: $OUTPUT_DIR/long_audio_segments/processing_summary.json"
        fi
    fi
    
    # VAD results
    if [ "$USE_VAD" = true ] && [ -f "$OUTPUT_DIR/vad_segments/vad_processing_summary.json" ]; then
        echo ""
        echo "VAD Results:"
        if command -v jq > /dev/null 2>&1; then
            SUCCESSFUL=$(jq -r '.successful' "$OUTPUT_DIR/vad_segments/vad_processing_summary.json" 2>/dev/null || echo "N/A")
            TOTAL_SPEECH=$(jq -r '.total_speech_duration' "$OUTPUT_DIR/vad_segments/vad_processing_summary.json" 2>/dev/null || echo "N/A")
            SPEECH_RATIO=$(jq -r '.overall_speech_ratio' "$OUTPUT_DIR/vad_segments/vad_processing_summary.json" 2>/dev/null || echo "N/A")
            echo "  - Files processed: $SUCCESSFUL"
            echo "  - Total speech duration: ${TOTAL_SPEECH}s"
            if [ "$SPEECH_RATIO" != "N/A" ]; then
                SPEECH_PERCENT=$(echo "$SPEECH_RATIO * 100" | bc -l 2>/dev/null | cut -d. -f1 2>/dev/null || echo "N/A")
                echo "  - Speech ratio: ${SPEECH_PERCENT}%"
            fi
        else
            echo "  - VAD summary available in: $OUTPUT_DIR/vad_segments/vad_processing_summary.json"
        fi
    fi
    
    # ASR results
    if [ -d "$TRANSCRIPT_DIR" ]; then
        echo ""
        echo "ASR Results:"
        TRANSCRIPT_COUNT=$(find "$TRANSCRIPT_DIR" -name "*.txt" | wc -l)
        echo "  - Transcript files: $TRANSCRIPT_COUNT"
        echo "  - Transcripts directory: $TRANSCRIPT_DIR"
        
        # Show merged transcripts info if applicable
        if [ "$USE_LONG_AUDIO_SPLIT" = true ] && [ -d "$OUTPUT_DIR/merged_transcripts" ]; then
            MERGED_COUNT=$(find "$OUTPUT_DIR/merged_transcripts" -name "*.txt" | wc -l)
            echo "  - Merged transcript files: $MERGED_COUNT"
        fi
    fi
    
    # Evaluation results
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "Evaluation Results:"
        echo "  - Evaluation report: $OUTPUT_FILE"
    fi
    
    # Model file analysis results
    if [ -f "$MODEL_ANALYSIS_FILE" ]; then
        echo ""
        echo "Model File Analysis:"
        echo "  - Model file analysis: $MODEL_ANALYSIS_FILE"
    fi
    
    echo ""
    echo "All results saved to: $OUTPUT_DIR"
    
} > "$SUMMARY_FILE"

echo "Pipeline summary saved to: $SUMMARY_FILE"
echo ""
echo "=== Pipeline Completed Successfully ==="
echo ""
echo "Results structure:"
if [ "$USE_LONG_AUDIO_SPLIT" = true ]; then
    echo "  $OUTPUT_DIR/long_audio_segments/   # Long audio split segments"
fi
if [ "$USE_VAD" = true ]; then
    echo "  $OUTPUT_DIR/vad_segments/          # VAD extracted speech segments"
fi
echo "  $OUTPUT_DIR/asr_transcripts/       # ASR transcription results"
if [ "$USE_LONG_AUDIO_SPLIT" = true ]; then
    echo "  $OUTPUT_DIR/merged_transcripts/    # Merged transcripts for evaluation"
fi
echo "  $OUTPUT_FILE         # Evaluation metrics"
echo "  $MODEL_ANALYSIS_FILE # Model file processing analysis"
echo "  $SUMMARY_FILE        # Detailed summary"
echo ""
echo "Check the summary file for detailed results: $SUMMARY_FILE" 