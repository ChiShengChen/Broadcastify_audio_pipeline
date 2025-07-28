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

USE_VAD=false                    # Enable VAD preprocessing
USE_LONG_AUDIO_SPLIT=false      # Enable long audio splitting to prevent OOM
MAX_SEGMENT_DURATION=120.0      # Maximum segment duration in seconds (2 minutes)

# Audio preprocessing options
USE_AUDIO_PREPROCESSING=true   # Enable audio preprocessing (upsampling and segmentation)
TARGET_SAMPLE_RATE=16000        # Target sample rate for upsampling
AUDIO_MAX_DURATION=60.0         # Maximum audio segment duration in seconds
AUDIO_OVERLAP_DURATION=1.0      # Overlap between audio segments in seconds
AUDIO_MIN_SEGMENT_DURATION=5.0  # Minimum audio segment duration in seconds


#### DO NOT CHANGE THESE OPTIONS ####
# Processing options
VAD_SPEECH_THRESHOLD=0.5        # VAD speech detection threshold
VAD_MIN_SPEECH_DURATION=0.5     # Minimum speech segment duration
VAD_MIN_SILENCE_DURATION=0.3    # Minimum silence between segments

# Ground truth preprocessing options
PREPROCESS_GROUND_TRUTH=true    # Enable ground truth preprocessing for better ASR matching
PREPROCESS_MODE="conservative"  # Preprocessing mode: conservative (minimal) or aggressive (extensive)
PREPROCESSED_GROUND_TRUTH_FILE=""  # Will be set automatically if preprocessing is enabled

# Enhanced preprocessing options
USE_ENHANCED_PREPROCESSOR=false  # Use enhanced preprocessor with comprehensive text normalization
ENHANCED_PREPROCESSOR_MODE="conservative"  # Enhanced preprocessor mode: conservative or aggressive


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
        --preprocess-ground-truth)
            PREPROCESS_GROUND_TRUTH=true
            shift
            ;;
        --no-preprocess-ground-truth)
            PREPROCESS_GROUND_TRUTH=false
            shift
            ;;
        --preprocess-mode)
            PREPROCESS_MODE="$2"
            shift 2
            ;;
        --use-enhanced-preprocessor)
            USE_ENHANCED_PREPROCESSOR=true
            shift
            ;;
        --no-enhanced-preprocessor)
            USE_ENHANCED_PREPROCESSOR=false
            shift
            ;;
        --enhanced-preprocessor-mode)
            ENHANCED_PREPROCESSOR_MODE="$2"
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
        --target-sample-rate)
            TARGET_SAMPLE_RATE="$2"
            shift 2
            ;;
        --audio-max-duration)
            AUDIO_MAX_DURATION="$2"
            shift 2
            ;;
        --audio-overlap-duration)
            AUDIO_OVERLAP_DURATION="$2"
            shift 2
            ;;
        --audio-min-segment-duration)
            AUDIO_MIN_SEGMENT_DURATION="$2"
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
            echo "  --preprocess-ground-truth    Enable ground truth preprocessing for better ASR matching"
            echo "  --no-preprocess-ground-truth Disable ground truth preprocessing"
            echo "  --preprocess-mode MODE       Preprocessing mode: conservative or aggressive (default: conservative)"
            echo "  --use-enhanced-preprocessor  Use enhanced preprocessor with comprehensive text normalization"
            echo "  --no-enhanced-preprocessor   Disable enhanced preprocessor (use basic preprocessor)"
            echo "  --enhanced-preprocessor-mode MODE Enhanced preprocessor mode: conservative or aggressive (default: conservative)"
            echo "  --use-audio-preprocessing    Enable audio preprocessing (upsampling and segmentation)"
            echo "  --no-audio-preprocessing     Disable audio preprocessing"
            echo "  --target-sample-rate INT     Target sample rate for upsampling (default: 16000)"
            echo "  --audio-max-duration FLOAT   Maximum audio segment duration in seconds (default: 60.0)"
            echo "  --audio-overlap-duration FLOAT Overlap between audio segments in seconds (default: 1.0)"
            echo "  --audio-min-segment-duration FLOAT Minimum audio segment duration in seconds (default: 5.0)"
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
            echo ""
            echo "  # With ground truth preprocessing"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --preprocess-ground-truth"
            echo ""
            echo "  # Without ground truth preprocessing"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --no-preprocess-ground-truth"
            echo ""
            echo "  # With aggressive preprocessing"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --preprocess-mode aggressive"
            echo ""
            echo "  # With enhanced preprocessor"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --use-enhanced-preprocessor"
            echo ""
            echo "  # With enhanced preprocessor in aggressive mode"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --use-enhanced-preprocessor --enhanced-preprocessor-mode aggressive"
            echo ""
            echo "  # With audio preprocessing (upsampling and segmentation)"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --use-audio-preprocessing"
            echo ""
            echo "  # With custom audio preprocessing parameters"
            echo "  $0 --input_dir /path/to/audio --output_dir /path/to/results --use-audio-preprocessing --target-sample-rate 16000 --audio-max-duration 60"
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

# --- Step 1: Audio Preprocessing (Optional) ---
if [ "$USE_AUDIO_PREPROCESSING" = true ]; then
    echo "--- Step 1: Audio Preprocessing (Upsampling and Segmentation) ---"
    AUDIO_PREPROCESSED_DIR="$OUTPUT_DIR/preprocessed_audio"
    
    echo "Running audio preprocessing..."
    echo "Input: $AUDIO_DIR"
    echo "Output: $AUDIO_PREPROCESSED_DIR"
    echo "Target sample rate: ${TARGET_SAMPLE_RATE}Hz"
    echo "Max duration: ${AUDIO_MAX_DURATION}s"
    echo "Overlap duration: ${AUDIO_OVERLAP_DURATION}s"
    echo "Min segment duration: ${AUDIO_MIN_SEGMENT_DURATION}s"
    
    $PYTHON_EXEC audio_preprocessor.py \
        --input_dir "$AUDIO_DIR" \
        --output_dir "$AUDIO_PREPROCESSED_DIR" \
        --target_sample_rate "$TARGET_SAMPLE_RATE" \
        --max_duration "$AUDIO_MAX_DURATION" \
        --overlap_duration "$AUDIO_OVERLAP_DURATION" \
        --min_segment_duration "$AUDIO_MIN_SEGMENT_DURATION" \
        --preserve_structure
    
    if [ $? -eq 0 ]; then
        echo "Audio preprocessing completed successfully"
        echo "Preprocessed audio saved to: $AUDIO_PREPROCESSED_DIR"
        # Set input directory for next steps to preprocessed audio
        PROCESSING_INPUT_DIR="$AUDIO_PREPROCESSED_DIR"
        AUDIO_PREPROCESSING_METADATA="$AUDIO_PREPROCESSED_DIR/processing_metadata.json"
    else
        echo "Warning: Audio preprocessing failed, using original files"
        echo "ERROR: Audio preprocessing failed" >> "$ERROR_LOG_FILE"
        echo "  Input directory: $AUDIO_DIR" >> "$ERROR_LOG_FILE"
        echo "  Output directory: $AUDIO_PREPROCESSED_DIR" >> "$ERROR_LOG_FILE"
        echo "  Using original audio files instead" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
        PROCESSING_INPUT_DIR="$AUDIO_DIR"
        AUDIO_PREPROCESSING_METADATA=""
    fi
else
    echo "--- Skipping Audio Preprocessing ---"
    PROCESSING_INPUT_DIR="$AUDIO_DIR"
    AUDIO_PREPROCESSING_METADATA=""
fi
echo ""

# --- Step 2: Ground Truth Preprocessing (Optional) ---
if [ "$PREPROCESS_GROUND_TRUTH" = true ]; then
    echo "--- Step 2: Ground Truth Preprocessing ---"
    
    # Set the preprocessed ground truth file path
    PREPROCESSED_GROUND_TRUTH_FILE="$OUTPUT_DIR/preprocessed_ground_truth.csv"
    
    echo "Preprocessing ground truth for better ASR matching..."
    echo "Input: $GROUND_TRUTH_FILE"
    echo "Output: $PREPROCESSED_GROUND_TRUTH_FILE"
    
    # Run the preprocessing script
    if [ "$USE_ENHANCED_PREPROCESSOR" = true ]; then
        echo "Using enhanced preprocessor with comprehensive text normalization..."
        $PYTHON_EXEC tool/enhanced_ground_truth_preprocessor.py \
            --input_file "$GROUND_TRUTH_FILE" \
            --output_file "$PREPROCESSED_GROUND_TRUTH_FILE" \
            --mode "$ENHANCED_PREPROCESSOR_MODE"
    else
        echo "Using basic preprocessor..."
        $PYTHON_EXEC tool/smart_preprocess_ground_truth.py \
            --input_file "$GROUND_TRUTH_FILE" \
            --output_file "$PREPROCESSED_GROUND_TRUTH_FILE" \
            --mode "$PREPROCESS_MODE"
    fi
    
    if [ $? -eq 0 ]; then
        echo "Ground truth preprocessing completed successfully"
        echo "Using preprocessed ground truth: $PREPROCESSED_GROUND_TRUTH_FILE"
        # Update the ground truth file path for evaluation
        EVALUATION_GROUND_TRUTH_FILE="$PREPROCESSED_GROUND_TRUTH_FILE"
    else
        echo "Warning: Ground truth preprocessing failed, using original file"
        echo "ERROR: Ground truth preprocessing failed" >> "$ERROR_LOG_FILE"
        echo "  Input file: $GROUND_TRUTH_FILE" >> "$ERROR_LOG_FILE"
        echo "  Output file: $PREPROCESSED_GROUND_TRUTH_FILE" >> "$ERROR_LOG_FILE"
        echo "  Mode: $PREPROCESS_MODE" >> "$ERROR_LOG_FILE"
        echo "  Using original ground truth file instead" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
        EVALUATION_GROUND_TRUTH_FILE="$GROUND_TRUTH_FILE"
    fi
else
    echo "--- Skipping Ground Truth Preprocessing ---"
    EVALUATION_GROUND_TRUTH_FILE="$GROUND_TRUTH_FILE"
fi
echo ""

# --------------------

# Display configuration
echo "=== Enhanced ASR Pipeline Configuration ==="
echo "Input directory: $AUDIO_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Ground truth file: $GROUND_TRUTH_FILE"
echo "Use Audio Preprocessing: $USE_AUDIO_PREPROCESSING"
if [ "$USE_AUDIO_PREPROCESSING" = true ]; then
    echo "Audio Preprocessing Parameters:"
    echo "  - Target sample rate: ${TARGET_SAMPLE_RATE}Hz"
    echo "  - Max duration: ${AUDIO_MAX_DURATION}s"
    echo "  - Overlap duration: ${AUDIO_OVERLAP_DURATION}s"
    echo "  - Min segment duration: ${AUDIO_MIN_SEGMENT_DURATION}s"
fi
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
echo "Preprocess Ground Truth: $PREPROCESS_GROUND_TRUTH"
if [ "$PREPROCESS_GROUND_TRUTH" = true ]; then
    echo "Use Enhanced Preprocessor: $USE_ENHANCED_PREPROCESSOR"
    if [ "$USE_ENHANCED_PREPROCESSOR" = true ]; then
        echo "Enhanced Preprocessor Mode: $ENHANCED_PREPROCESSOR_MODE"
    else
        echo "Basic Preprocessor Mode: $PREPROCESS_MODE"
    fi
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

# --- Step 3: Long Audio Splitting (Optional) ---
if [ "$USE_LONG_AUDIO_SPLIT" = true ]; then
    echo "--- Step 3: Long Audio Splitting ---"
    LONG_AUDIO_OUTPUT_DIR="$OUTPUT_DIR/long_audio_segments"
    
    echo "Running Long Audio Splitter to prevent OOM issues..."
    $PYTHON_EXEC long_audio_splitter.py \
        --input_dir "$PROCESSING_INPUT_DIR" \
        --output_dir "$LONG_AUDIO_OUTPUT_DIR" \
        --max_duration "$MAX_SEGMENT_DURATION" \
        --speech_threshold "$VAD_SPEECH_THRESHOLD" \
        --min_speech_duration "$VAD_MIN_SPEECH_DURATION" \
        --min_silence_duration "$VAD_MIN_SILENCE_DURATION"
    
    echo "Long audio splitting completed"
    echo "Split segments saved to: $LONG_AUDIO_OUTPUT_DIR"
    echo ""
    
    # Set input directory for next steps to split segments
    ASR_INPUT_DIR="$LONG_AUDIO_OUTPUT_DIR"
else
    ASR_INPUT_DIR="$PROCESSING_INPUT_DIR"
fi

# --- Step 4: VAD Processing (Optional) ---
if [ "$USE_VAD" = true ]; then
    echo "--- Step 4: VAD Processing ---"
    VAD_OUTPUT_DIR="$OUTPUT_DIR/vad_segments"
    
    if [ "$USE_ENHANCED_VAD" = true ]; then
        echo "Running Enhanced VAD with audio filters..."
        $PYTHON_EXEC enhanced_vad_pipeline.py \
            --input_dir "$ASR_INPUT_DIR" \
            --output_dir "$VAD_OUTPUT_DIR" \
            --speech_threshold "$VAD_SPEECH_THRESHOLD" \
            --min_speech_duration "$VAD_MIN_SPEECH_DURATION" \
            --min_silence_duration "$VAD_MIN_SILENCE_DURATION"
    else
        echo "Running Basic VAD..."
        $PYTHON_EXEC vad_pipeline.py \
            --input_dir "$ASR_INPUT_DIR" \
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
    # ASR_INPUT_DIR is already set from previous step
fi

# --- Step 5: ASR Processing ---
echo "--- Step 5: ASR Transcription ---"
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
    
    if [ $? -ne 0 ]; then
        echo "Warning: ASR processing encountered issues"
        echo "ERROR: ASR processing failed for VAD files" >> "$ERROR_LOG_FILE"
        echo "  VAD directory: $temp_vad_dir" >> "$ERROR_LOG_FILE"
        echo "  Number of files: ${#vad_files[@]}" >> "$ERROR_LOG_FILE"
        echo "  Check ASR logs for details" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
    fi
        
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

# --- Step 6: Merge Segmented Transcripts (if audio preprocessing was used) ---
if [ "$USE_AUDIO_PREPROCESSING" = true ] && [ -n "$AUDIO_PREPROCESSING_METADATA" ] && [ -f "$AUDIO_PREPROCESSING_METADATA" ]; then
    echo "--- Step 6: Merging Segmented Transcripts ---"
    MERGED_SEGMENTED_TRANSCRIPTS_DIR="$OUTPUT_DIR/merged_segmented_transcripts"
    
    echo "Merging segmented transcripts for WER calculation..."
    $PYTHON_EXEC merge_segmented_transcripts.py \
        --input_dir "$TRANSCRIPT_DIR" \
        --output_dir "$MERGED_SEGMENTED_TRANSCRIPTS_DIR" \
        --metadata_file "$AUDIO_PREPROCESSING_METADATA"
    
    if [ $? -eq 0 ]; then
        echo "Segmented transcript merging completed"
        echo "Merged transcripts saved to: $MERGED_SEGMENTED_TRANSCRIPTS_DIR"
        echo ""
        
        # Set transcript directory for evaluation to merged segmented transcripts
        TRANSCRIPT_DIR="$MERGED_SEGMENTED_TRANSCRIPTS_DIR"
    else
        echo "Warning: Segmented transcript merging failed, using original transcripts"
        echo "ERROR: Segmented transcript merging failed" >> "$ERROR_LOG_FILE"
        echo "  Input directory: $TRANSCRIPT_DIR" >> "$ERROR_LOG_FILE"
        echo "  Metadata file: $AUDIO_PREPROCESSING_METADATA" >> "$ERROR_LOG_FILE"
        echo "  Using original transcripts instead" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
    fi
fi

# --- Step 7: Merge Split Transcripts (if long audio splitting was used) ---
if [ "$USE_LONG_AUDIO_SPLIT" = true ]; then
    echo "--- Step 7: Merging Split Transcripts ---"
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

# --- Step 8: Evaluation ---
echo "--- Step 8: Evaluating ASR Results ---"
if [[ -f "$EVALUATION_GROUND_TRUTH_FILE" && -d "$TRANSCRIPT_DIR" ]]; then
    $PYTHON_EXEC evaluate_asr.py \
        --transcript_dirs "$TRANSCRIPT_DIR" \
        --ground_truth_file "$EVALUATION_GROUND_TRUTH_FILE" \
        --output_file "$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully"
        echo "Results saved to: $OUTPUT_FILE"
    else
        echo "Warning: Evaluation encountered issues"
        echo "ERROR: ASR evaluation failed" >> "$ERROR_LOG_FILE"
        echo "  Ground truth file: $EVALUATION_GROUND_TRUTH_FILE" >> "$ERROR_LOG_FILE"
        echo "  Transcript directory: $TRANSCRIPT_DIR" >> "$ERROR_LOG_FILE"
        echo "  Output file: $OUTPUT_FILE" >> "$ERROR_LOG_FILE"
        echo "  Check evaluation logs for details" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
    fi
else
    echo "Skipping evaluation - missing ground truth file or transcript directory"
    echo "ERROR: Cannot perform evaluation - missing required files" >> "$ERROR_LOG_FILE"
    if [[ ! -f "$EVALUATION_GROUND_TRUTH_FILE" ]]; then
        echo "  Missing ground truth file: $EVALUATION_GROUND_TRUTH_FILE" >> "$ERROR_LOG_FILE"
    fi
    if [[ ! -d "$TRANSCRIPT_DIR" ]]; then
        echo "  Missing transcript directory: $TRANSCRIPT_DIR" >> "$ERROR_LOG_FILE"
    fi
    echo "" >> "$ERROR_LOG_FILE"
fi
echo ""

# --- Step 8.5: Model File Analysis with Error Logging ---
echo "--- Step 8.5: Analyzing Model File Processing with Error Logging ---"
MODEL_ANALYSIS_FILE="$OUTPUT_DIR/model_file_analysis.txt"
ERROR_LOG_FILE="$OUTPUT_DIR/error_analysis.log"

# Initialize error log
echo "=== Error Analysis Log ===" > "$ERROR_LOG_FILE"
echo "Analysis Date: $(date)" >> "$ERROR_LOG_FILE"
echo "Pipeline Output Directory: $OUTPUT_DIR" >> "$ERROR_LOG_FILE"
echo "" >> "$ERROR_LOG_FILE"

if [[ -d "$TRANSCRIPT_DIR" ]]; then
    echo "Running detailed model file analysis with error logging..."
    
    # Run analysis with error logging
    $PYTHON_EXEC tool/analyze_model_files_enhanced.py \
        --transcript_dir "$TRANSCRIPT_DIR" \
        --ground_truth_file "$EVALUATION_GROUND_TRUTH_FILE" \
        --output_file "$MODEL_ANALYSIS_FILE" \
        --error_log_file "$ERROR_LOG_FILE" \
        --pipeline_output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Model file analysis completed successfully"
        echo "Detailed analysis saved to: $MODEL_ANALYSIS_FILE"
        echo "Error analysis saved to: $ERROR_LOG_FILE"
    else
        echo "Warning: Model file analysis encountered issues, but continued"
        echo "Check error log: $ERROR_LOG_FILE"
    fi
else
    echo "Skipping model file analysis - transcript directory not found"
    echo "ERROR: Transcript directory not found: $TRANSCRIPT_DIR" >> "$ERROR_LOG_FILE"
    echo "This may indicate a failure in previous pipeline steps" >> "$ERROR_LOG_FILE"
fi
echo ""

# --- Step 9: Generate Summary ---
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
    echo "  - Audio Preprocessing: $USE_AUDIO_PREPROCESSING"
    if [ "$USE_AUDIO_PREPROCESSING" = true ]; then
        echo "  - Audio Preprocessing Parameters:"
        echo "    * Target sample rate: ${TARGET_SAMPLE_RATE}Hz"
        echo "    * Max duration: ${AUDIO_MAX_DURATION}s"
        echo "    * Overlap duration: ${AUDIO_OVERLAP_DURATION}s"
        echo "    * Min segment duration: ${AUDIO_MIN_SEGMENT_DURATION}s"
    fi
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
    echo "  - Ground Truth Preprocessing: $PREPROCESS_GROUND_TRUTH"
    if [ "$PREPROCESS_GROUND_TRUTH" = true ]; then
        echo "  - Ground Truth Preprocessing:"
        echo "    * Input file: $GROUND_TRUTH_FILE"
        echo "    * Processed file: $PREPROCESSED_GROUND_TRUTH_FILE"
        if [ "$USE_ENHANCED_PREPROCESSOR" = true ]; then
            echo "    * Preprocessor: Enhanced"
            echo "    * Mode: $ENHANCED_PREPROCESSOR_MODE"
        else
            echo "    * Preprocessor: Basic"
            echo "    * Mode: $PREPROCESS_MODE"
        fi
    fi
    echo ""
    
    # Count input files
    INPUT_COUNT=$(find "$AUDIO_DIR" -name "*.wav" | wc -l)
    echo "Input Files: $INPUT_COUNT audio files"
    
    # Audio preprocessing results
    if [ "$USE_AUDIO_PREPROCESSING" = true ] && [ -f "$OUTPUT_DIR/preprocessed_audio/processing_metadata.json" ]; then
        echo ""
        echo "Audio Preprocessing Results:"
        if command -v jq > /dev/null 2>&1; then
            TOTAL_FILES=$(jq -r '.total_files' "$OUTPUT_DIR/preprocessed_audio/processing_metadata.json" 2>/dev/null || echo "N/A")
            PROCESSED_FILES=$(jq -r '.processed_files' "$OUTPUT_DIR/preprocessed_audio/processing_metadata.json" 2>/dev/null || echo "N/A")
            TOTAL_SEGMENTS=$(jq -r '.total_segments' "$OUTPUT_DIR/preprocessed_audio/processing_metadata.json" 2>/dev/null || echo "N/A")
            UPSAMPLED_FILES=$(jq -r '.upsampled_files' "$OUTPUT_DIR/preprocessed_audio/processing_metadata.json" 2>/dev/null || echo "N/A")
            SPLIT_FILES=$(jq -r '.split_files' "$OUTPUT_DIR/preprocessed_audio/processing_metadata.json" 2>/dev/null || echo "N/A")
            echo "  - Total files processed: $TOTAL_FILES"
            echo "  - Successfully processed: $PROCESSED_FILES"
            echo "  - Total segments created: $TOTAL_SEGMENTS"
            echo "  - Files upsampled: $UPSAMPLED_FILES"
            echo "  - Files split: $SPLIT_FILES"
        else
            echo "  - Audio preprocessing summary available in: $OUTPUT_DIR/preprocessed_audio/processing_metadata.json"
        fi
    fi
    
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
    
    # Error analysis results
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
    
} > "$SUMMARY_FILE"

echo "Pipeline summary saved to: $SUMMARY_FILE"
echo ""

# Check for errors and determine pipeline status
PIPELINE_SUCCESS=true
ERROR_COUNT=0
WARNING_COUNT=0

# Check error log if it exists
if [ -f "$ERROR_LOG_FILE" ]; then
    ERROR_COUNT=$(grep -c "\[ERROR\]" "$ERROR_LOG_FILE" 2>/dev/null || echo "0")
    WARNING_COUNT=$(grep -c "\[WARNING\]" "$ERROR_LOG_FILE" 2>/dev/null || echo "0")
    
    # If there are errors, mark pipeline as failed
    if [ "$ERROR_COUNT" -gt 0 ]; then
        PIPELINE_SUCCESS=false
    fi
fi

# Check if critical files exist
if [ ! -f "$OUTPUT_FILE" ]; then
    PIPELINE_SUCCESS=false
fi

if [ ! -d "$TRANSCRIPT_DIR" ]; then
    PIPELINE_SUCCESS=false
fi

# Display final status
if [ "$PIPELINE_SUCCESS" = true ]; then
    echo "=== Pipeline Completed Successfully ==="
    echo ""
    echo "Results structure:"
    if [ "$USE_AUDIO_PREPROCESSING" = true ]; then
        echo "  $OUTPUT_DIR/preprocessed_audio/     # Audio preprocessing results"
    fi
    if [ "$USE_LONG_AUDIO_SPLIT" = true ]; then
        echo "  $OUTPUT_DIR/long_audio_segments/   # Long audio split segments"
    fi
    if [ "$USE_VAD" = true ]; then
        echo "  $OUTPUT_DIR/vad_segments/          # VAD extracted speech segments"
    fi
    echo "  $OUTPUT_DIR/asr_transcripts/       # ASR transcription results"
    if [ "$USE_AUDIO_PREPROCESSING" = true ]; then
        echo "  $OUTPUT_DIR/merged_segmented_transcripts/ # Merged segmented transcripts"
    fi
    if [ "$USE_LONG_AUDIO_SPLIT" = true ]; then
        echo "  $OUTPUT_DIR/merged_transcripts/    # Merged transcripts for evaluation"
    fi
    echo "  $OUTPUT_FILE         # Evaluation metrics"
    echo "  $MODEL_ANALYSIS_FILE # Model file processing analysis"
    echo "  $SUMMARY_FILE        # Detailed summary"
    echo ""
    echo "Check the summary file for detailed results: $SUMMARY_FILE"
    
    # Show warnings if any
    if [ "$WARNING_COUNT" -gt 0 ]; then
        echo ""
        echo "⚠️  Note: $WARNING_COUNT warnings were detected during processing."
        echo "   Check $ERROR_LOG_FILE for details."
    fi
else
    echo "=== Pipeline Completed with Errors ==="
    echo ""
    echo "❌ Pipeline encountered issues during execution."
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
    echo "  1. Check the error log: $ERROR_LOG_FILE"
    echo "  2. Review the pipeline summary: $SUMMARY_FILE"
    echo "  3. Verify input files and configuration"
    echo "  4. Check system resources (disk space, memory)"
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