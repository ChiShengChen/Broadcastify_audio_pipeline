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
ASR_RESULTS_DIR="/media/meow/One Touch/ems_call/pipeline_results_20250823_095857"
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
MEDICAL_CORRECTION_MODEL="BioMistral-7B"    # Model for medical term correction
PAGE_GENERATION_MODEL="BioMistral-7B"     # Model for emergency page generation
EXTRACTION_MODEL="BioMistral-7B"          # Model for information extraction

# --- Feature Switches ---
ENABLE_MEDICAL_CORRECTION=true    # Enable medical term correction
ENABLE_PAGE_GENERATION=false       # Enable emergency page generation
ENABLE_EVALUATION=true            # Enable evaluation of corrected results

ENABLE_WHISPER_FILTER=false        # Enable filtering for Whisper results only
ENABLE_MULTI_ASR_COMPARISON=false # Enable comparison and merge of multiple ASR results (Canary + Whisper)
ENABLE_ASR_SELECTION=true        # Enable ASR selection mode (choose better ASR result)
ENABLE_INFORMATION_EXTRACTION=true # Enable information extraction step

AUTO_DETECT_MULTI_ASR=true        # Automatically detect and use multiple ASR results from pipeline output

# Device configuration
DEVICE="auto"  # auto, cpu, cuda
LOAD_IN_8BIT=false
LOAD_IN_4BIT=false

# Generation parameters
TEMPERATURE=0.1  # Default temperature for gpt-oss models
MAX_NEW_TOKENS=128  # Default max new tokens for gpt-oss models

# --- Model Compatibility Configuration ---
# 添加模型兼容性配置
GPT_OSS_120B_COMPATIBLE=false  # 默认禁用 120b 以避免兼容性问题
GPT_OSS_20B_COMPATIBLE=true    # 20b 模型通常更稳定

# --- Medical Correction Configuration ---
MEDICAL_CORRECTION_PROMPT="You are an expert medical transcription correction system. Your role is to improve noisy, error-prone transcripts generated from EMS radio calls. These transcripts are derived from automatic speech recognition (ASR) and often contain phonetic errors, especially with medication names, clinical terminology, and numerical values.

Each transcript reflects a real-time communication from EMS personnel to hospital staff, summarizing a patient's clinical condition, vital signs, and any treatments administered during prehospital care. Use your knowledge of emergency medicine, pharmacology, and EMS protocols to reconstruct the intended meaning of the message as accurately and clearly as possible.

Guidelines:
1. Replace misrecognized or phonetically incorrect words and phrases with their most likely intended clinical equivalents.
2. Express the message in clear, natural language while maintaining the tone and intent of an EMS-to-hospital handoff.
3. Include all information from the original transcript—ensure your output is complete and continuous.
4. Use medical abbreviations and shorthand appropriately when they match clinical usage (e.g., "BP" for blood pressure, "ETT" for endotracheal tube).
5. Apply contextual reasoning to identify and correct drug names, dosages, clinical phrases, and symptoms using common EMS knowledge.
6. Deliver your output as plain, unstructured text without metadata, formatting, or explanatory notes.
7. Present the cleaned transcript as a fully corrected version, without gaps, placeholders, or annotations."

# --- Multi-ASR Comparison Configuration ---
MULTI_ASR_COMPARISON_PROMPT="You are an expert medical transcription specialist. You have been provided with multiple ASR transcriptions of the same EMS radio call from different systems (Canary and Whisper). Your task is to compare these transcriptions and provide the best possible corrected version by combining the strengths of each system.

COMPARISON GUIDELINES:
1. Analyze both transcriptions for accuracy, completeness, and medical terminology
2. Identify which transcription is more accurate for different parts of the message
3. Combine the best elements from both transcriptions
4. Correct any obvious medical terminology errors
5. Maintain the original meaning and context
6. Provide a single, coherent, corrected transcript

Return only the corrected transcript without explanations or metadata.

CANARY TRANSCRIPT:
{canary_transcript}

WHISPER TRANSCRIPT:
{whisper_transcript}

BEST COMBINED AND CORRECTED TRANSCRIPT:"

# --- ASR Selection Configuration ---
ASR_SELECTION_PROMPT="You are an expert medical transcription specialist evaluating two ASR transcriptions of the same EMS radio call. Your task is to determine which transcription is better and provide a brief explanation.

EVALUATION CRITERIA:
1. Accuracy: Which transcription more accurately captures the spoken words?
2. Completeness: Which transcription includes more complete information?
3. Medical Terminology: Which transcription has better medical term recognition?
4. Clarity: Which transcription is clearer and more readable?
5. Context: Which transcription better maintains the EMS communication context?

OUTPUT FORMAT:
Return a JSON object with the following structure:
{
  \"selected_asr\": \"canary\" or \"whisper\",
  \"reason\": \"brief explanation of why this ASR was selected\",
  \"accuracy_score\": 1-10,
  \"completeness_score\": 1-10,
  \"medical_terminology_score\": 1-10
}

CANARY TRANSCRIPT:
{canary_transcript}

WHISPER TRANSCRIPT:
{whisper_transcript}

EVALUATION RESULT:"

# --- Information Extraction Configuration ---
INFORMATION_EXTRACTION_PROMPT="You are an information extraction model for EMS prearrival radio transcripts in Massachusetts. TASK: Return a single JSON object only. No prose, no code fences, no explanations. SCHEMA (all keys required; values are strings; if unspecified, use \"\"): {\"agency\": \"\", \"unit\": \"\", \"ETA\": \"\", \"age\": \"\", \"sex\": \"\", \"moi\": \"\", \"hr\": \"\", \"rrq\": \"\", \"sbp\": \"\", \"dbp\": \"\", \"end_tidal\": \"\", \"rr\": \"\", \"bgl\": \"\", \"spo2\": \"\", \"o2\": \"\", \"injuries\": \"\", \"ao\": \"\", \"GCS\": \"\", \"LOC\": \"\", \"ac\": \"\", \"treatment\": \"\", \"pregnant\": \"\", \"notes\": \"\"} RULES: Fill fields only with information explicitly stated in the transcript. Do not infer, guess, or normalize beyond obvious medical term corrections. Keep numbers as they are spoken. If multiple possibilities are stated, choose the most explicit; otherwise put \"\". Output must be valid JSON. No trailing commas. OUTPUT FORMAT: A single JSON object exactly matching the SCHEMA keys and order above. TRANSCRIPT:"



# --- Emergency Page Generation Configuration ---
PAGE_GENERATION_PROMPT="You are an expert emergency medical dispatcher. You have been provided with extracted medical information from an EMS prearrival radio call in JSON format. Your task is to generate a comprehensive, structured emergency page that includes all critical information for hospital staff.

EXTRACTED INFORMATION:
{extracted_json}

ADDITIONAL CONTEXT:
- This is a prearrival notification from EMS to hospital
- The information should be formatted for immediate clinical use
- Include priority level assessment based on vital signs and condition
- Highlight any critical interventions or treatments already provided

Generate a structured emergency page that includes:
1) Patient demographics and ETA
2) Mechanism of injury or chief complaint
3) Vital signs and clinical status
4) Treatments provided
5) Priority level and required resources
6) Additional clinical notes

Format the response as a clear, structured emergency page suitable for hospital handoff."

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

# Function to clear GPU cache and optimize memory before running LLM models
clear_gpu_cache() {
    echo "Clearing GPU cache and optimizing memory before loading model..."
    
    # Create a temporary Python script for comprehensive GPU memory management
    local temp_script="$OUTPUT_DIR/clear_gpu_cache.py"
    cat > "$temp_script" << 'EOF'
#!/usr/bin/env python3
import torch
import gc
import os
import subprocess
import time

def kill_gpu_processes():
    """Kill other GPU processes to free memory"""
    try:
        # Get GPU processes
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            processes = result.stdout.strip().split('\n')
            current_pid = os.getpid()
            
            print("Current GPU processes:")
            for process in processes:
                if process.strip():
                    parts = process.split(', ')
                    if len(parts) >= 3:
                        pid, name, memory = parts[0], parts[1], parts[2]
                        print(f"  PID {pid}: {name} using {memory}MB")
                        
                        # Don't kill our own process or critical system processes
                        if (int(pid) != current_pid and 
                            'python' in name.lower() and 
                            'jupyter' not in name.lower() and
                            'vscode' not in name.lower()):
                            try:
                                print(f"  Attempting to kill process {pid} ({name})")
                                os.kill(int(pid), 9)  # SIGKILL
                                time.sleep(1)
                            except:
                                print(f"  Could not kill process {pid}")
        else:
            print("No GPU processes found or nvidia-smi not available")
            
    except Exception as e:
        print(f"Warning: Failed to check/kill GPU processes: {e}")

def clear_gpu_cache():
    """Comprehensive GPU memory clearing and optimization"""
    try:
        # Kill other GPU processes first
        kill_gpu_processes()
        
        # Clear PyTorch cache multiple times
        if torch.cuda.is_available():
            print("Performing comprehensive GPU cache clearing...")
            
            # Multiple rounds of cache clearing
            for i in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.5)
            
            print("Cleared PyTorch CUDA cache (multiple rounds)")
            
            # Get detailed memory info
            if torch.cuda.device_count() > 0:
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    free = total - reserved
                    print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free, {total:.2f}GB total")
                    
                    # Reset memory stats
                    torch.cuda.reset_peak_memory_stats(i)
                    torch.cuda.reset_accumulated_memory_stats(i)
                    
            # Additional PyTorch memory management
            torch.cuda.ipc_collect()
            
        else:
            print("CUDA not available, skipping GPU cache clearing")
        
        # Aggressive garbage collection
        for i in range(3):
            gc.collect()
            time.sleep(0.1)
        print("Performed aggressive garbage collection")
        
        # Set memory optimization environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        print("Set PYTORCH_CUDA_ALLOC_CONF for memory optimization")
        
        # Additional CUDA environment optimizations
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error reporting
        print("Set additional CUDA optimization flags")
        
    except Exception as e:
        print(f"Warning: Failed to clear GPU cache: {e}")

if __name__ == "__main__":
    clear_gpu_cache()
EOF
    
    # Run the comprehensive cache clearing script
    $PYTHON_EXEC "$temp_script"
    
    # Clean up temporary script
    rm -f "$temp_script"
    
    # Additional system-level memory management
    echo "Performing additional system memory optimization..."
    
    # Force system cache clearing (if available)
    if command -v sync >/dev/null 2>&1; then
        sync
        echo "Synced system caches"
    fi
    
    # Wait a moment for memory to be freed
    sleep 2
    
    echo "GPU cache clearing and memory optimization completed"
    echo ""
}

# Function to automatically detect and organize multi-ASR results
detect_multi_asr_results() {
    local asr_results_dir="$1"
    local output_dir="$2"
    
    echo "Auto-detecting multi-ASR results from: $asr_results_dir"
    
    # Create organized multi-ASR directory
    local multi_asr_dir="$output_dir/multi_asr_organized"
    mkdir -p "$multi_asr_dir"
    
    # Look for transcript directories in priority order
    local transcript_dirs=()
    
    # Priority 1: merged_segmented_transcripts (complete files after merging segments)
    if [ -d "$asr_results_dir/merged_segmented_transcripts" ]; then
        transcript_dirs+=("$asr_results_dir/merged_segmented_transcripts")
    fi
    
    # Priority 2: merged_transcripts (for long audio splits)
    if [ -d "$asr_results_dir/merged_transcripts" ]; then
        transcript_dirs+=("$asr_results_dir/merged_transcripts")
    fi
    
    # Priority 3: asr_transcripts (raw ASR results)
    if [ -d "$asr_results_dir/asr_transcripts" ]; then
        transcript_dirs+=("$asr_results_dir/asr_transcripts")
    fi
    
    # If no specific transcript directory found, check the root
    if [ ${#transcript_dirs[@]} -eq 0 ]; then
        if find "$asr_results_dir" -maxdepth 1 -name "*.txt" | grep -q .; then
            transcript_dirs+=("$asr_results_dir")
        fi
    fi
    
    if [ ${#transcript_dirs[@]} -eq 0 ]; then
        echo "ERROR: No transcript directories found in $asr_results_dir"
        return 1
    fi
    
    # Use the highest priority directory
    local source_dir="${transcript_dirs[0]}"
    echo "Using transcript directory: $source_dir"
    
    # Find all transcript files and organize by model
    local canary_files=()
    local whisper_files=()
    local other_files=()
    
    while IFS= read -r -d '' file; do
        local filename=$(basename "$file")
        
        # Check for Canary model files
        if [[ "$filename" == canary* ]] || [[ "$filename" == *"canary"* ]]; then
            canary_files+=("$file")
        # Check for Whisper model files
        elif [[ "$filename" == large-v3* ]] || [[ "$filename" == *"whisper"* ]] || [[ "$filename" == *"large-v3"* ]]; then
            whisper_files+=("$file")
        else
            other_files+=("$file")
        fi
    done < <(find "$source_dir" -name "*.txt" -print0)
    
    echo "Found ${#canary_files[@]} Canary files, ${#whisper_files[@]} Whisper files, ${#other_files[@]} other files"
    
    # Create organized structure
    if [ ${#canary_files[@]} -gt 0 ]; then
        mkdir -p "$multi_asr_dir/canary"
        for file in "${canary_files[@]}"; do
            local base_name=$(basename "$file")
            cp "$file" "$multi_asr_dir/canary/$base_name"
        done
        echo "Organized Canary files in: $multi_asr_dir/canary"
    fi
    
    if [ ${#whisper_files[@]} -gt 0 ]; then
        mkdir -p "$multi_asr_dir/whisper"
        for file in "${whisper_files[@]}"; do
            local base_name=$(basename "$file")
            cp "$file" "$multi_asr_dir/whisper/$base_name"
        done
        echo "Organized Whisper files in: $multi_asr_dir/whisper"
    fi
    
    # Create a mapping file for multi-ASR comparison
    if [ ${#canary_files[@]} -gt 0 ] && [ ${#whisper_files[@]} -gt 0 ]; then
        echo "Creating multi-ASR mapping for comparison..."
        
        # Create a Python script to generate the mapping
        local mapping_script="$output_dir/create_multi_asr_mapping.py"
        cat > "$mapping_script" << 'EOF'
#!/usr/bin/env python3
import os
import json
from pathlib import Path

def create_multi_asr_mapping(canary_dir, whisper_dir, output_dir):
    """Create mapping between Canary and Whisper files for multi-ASR comparison"""
    
    canary_files = {}
    whisper_files = {}
    
    # Collect Canary files
    if os.path.exists(canary_dir):
        for file_path in Path(canary_dir).glob("*.txt"):
            # Extract base name (remove model prefix)
            filename = file_path.name
            if filename.startswith("canary-1b_"):
                base_name = filename[10:]  # Remove "canary-1b_" prefix
            elif filename.startswith("canary_"):
                base_name = filename[8:]   # Remove "canary_" prefix
            else:
                base_name = filename
            
            canary_files[base_name] = str(file_path)
    
    # Collect Whisper files
    if os.path.exists(whisper_dir):
        for file_path in Path(whisper_dir).glob("*.txt"):
            # Extract base name (remove model prefix)
            filename = file_path.name
            if filename.startswith("large-v3_"):
                base_name = filename[9:]   # Remove "large-v3_" prefix
            elif filename.startswith("whisper_"):
                base_name = filename[8:]   # Remove "whisper_" prefix
            else:
                base_name = filename
            
            whisper_files[base_name] = str(file_path)
    
    # Create mapping for files that exist in both
    mapping = {}
    for base_name in set(canary_files.keys()) & set(whisper_files.keys()):
        mapping[base_name] = {
            "canary": canary_files[base_name],
            "whisper": whisper_files[base_name]
        }
    
    # Save mapping
    mapping_file = os.path.join(output_dir, "multi_asr_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Created mapping for {len(mapping)} file pairs")
    print(f"Mapping saved to: {mapping_file}")
    
    return mapping_file

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python3 script.py <canary_dir> <whisper_dir> <output_dir>")
        sys.exit(1)
    
    canary_dir = sys.argv[1]
    whisper_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    create_multi_asr_mapping(canary_dir, whisper_dir, output_dir)
EOF
        
        # Run the mapping script
        $PYTHON_EXEC "$mapping_script" "$multi_asr_dir/canary" "$multi_asr_dir/whisper" "$multi_asr_dir"
        
        # Clean up script
        rm -f "$mapping_script"
        
        echo "Multi-ASR organization completed successfully"
        echo "Organized files available in: $multi_asr_dir"
        return 0
    else
        echo "WARNING: Not enough ASR models found for multi-ASR comparison"
        echo "  Canary files: ${#canary_files[@]}"
        echo "  Whisper files: ${#whisper_files[@]}"
        echo "  Need at least one file from each model for comparison"
        return 1
    fi
}

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
        --extraction_model)
            EXTRACTION_MODEL="$2"
            shift 2
            ;;

        --enable_multi_asr_comparison)
            ENABLE_MULTI_ASR_COMPARISON=true
            shift
            ;;
        --disable_multi_asr_comparison)
            ENABLE_MULTI_ASR_COMPARISON=false
            shift
            ;;
        --enable_information_extraction)
            ENABLE_INFORMATION_EXTRACTION=true
            shift
            ;;
        --disable_information_extraction)
            ENABLE_INFORMATION_EXTRACTION=false
            shift
            ;;

        --enable_auto_detect_multi_asr)
            AUTO_DETECT_MULTI_ASR=true
            shift
            ;;
        --disable_auto_detect_multi_asr)
            AUTO_DETECT_MULTI_ASR=false
            shift
            ;;
        --multi_asr_comparison_prompt)
            MULTI_ASR_COMPARISON_PROMPT="$2"
            shift 2
            ;;
        --asr_selection_prompt)
            ASR_SELECTION_PROMPT="$2"
            shift 2
            ;;
        --information_extraction_prompt)
            INFORMATION_EXTRACTION_PROMPT="$2"
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
        --enable_gpt_oss_120b)
            GPT_OSS_120B_COMPATIBLE=true
            shift
            ;;
        --disable_gpt_oss_120b)
            GPT_OSS_120B_COMPATIBLE=t
            shift
            ;;
        --enable_asr_selection)
            ENABLE_ASR_SELECTION=true
            shift
            ;;
        --disable_asr_selection)
            ENABLE_ASR_SELECTION=false
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
echo "  --extraction_model MODEL          Model for information extraction"
echo "                                    Available: gpt-oss-20b, gpt-oss-120b, BioMistral-7B, Meditron-7B, Llama-3-8B-UltraMedica"

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
echo "  --enable_multi_asr_comparison      Enable comparison of multiple ASR results (Canary + Whisper)"
echo "  --disable_multi_asr_comparison     Disable multi-ASR comparison (default)"
echo "  --enable_asr_selection            Enable ASR selection mode (choose better ASR result)"
echo "  --disable_asr_selection           Disable ASR selection mode (default)"
echo "  --enable_information_extraction    Enable information extraction step (default)"
echo "  --disable_information_extraction   Disable information extraction"

echo "  --enable_auto_detect_multi_asr    Enable automatic detection of multi-ASR results (default)"
echo "  --disable_auto_detect_multi_asr   Disable automatic detection of multi-ASR results"
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
echo "  --multi_asr_comparison_prompt TEXT Custom prompt for multi-ASR comparison"
echo "  --information_extraction_prompt TEXT Custom prompt for information extraction"

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
echo ""
echo "  # Multi-ASR comparison (Canary + Whisper)"
echo "  $0 --asr_results_dir /path/to/asr/results \\"
echo "     --enable_multi_asr_comparison"
echo ""
echo "  # Auto-detect and compare multiple ASR results"
echo "  $0 --asr_results_dir /path/to/pipeline/results \\"
echo "     --enable_multi_asr_comparison \\"
echo "     --enable_auto_detect_multi_asr"
echo ""
echo "  # Information extraction only"
echo "  $0 --asr_results_dir /path/to/asr/results \\"
echo "     --disable_page_generation \\"
echo "     --enable_information_extraction"
echo ""
echo "  # Enhanced processing with extracted data"
echo "  $0 --asr_results_dir /path/to/asr/results \\"
echo "     --enable_information_extraction \\"

echo "     --disable_page_generation"
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
    
    # Check for known compatibility issues
    if [ "$model" = "gpt-oss-120b" ] && [ "$GPT_OSS_120B_COMPATIBLE" = false ]; then
        echo "Warning: gpt-oss-120b has known compatibility issues with 'NoneType' object has no attribute 'to_dict' error"
        echo "Consider using gpt-oss-20b instead, or set GPT_OSS_120B_COMPATIBLE=true if you want to proceed"
        echo "This error typically occurs due to:"
        echo "  1. Transformers library version incompatibility"
        echo "  2. Model configuration loading issues"
        echo "  3. Memory allocation problems"
        echo ""
        read -p "Do you want to continue with gpt-oss-120b anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Switching to gpt-oss-20b for better compatibility..."
            if [ "$1" = "$MEDICAL_CORRECTION_MODEL" ]; then
                MEDICAL_CORRECTION_MODEL="gpt-oss-20b"
            fi
            if [ "$1" = "$PAGE_GENERATION_MODEL" ]; then
                PAGE_GENERATION_MODEL="gpt-oss-20b"
            fi
        fi
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
  echo "  - Multi-ASR Comparison: $ENABLE_MULTI_ASR_COMPARISON"
  echo "  - ASR Selection: $ENABLE_ASR_SELECTION"
if [ "$ENABLE_MULTI_ASR_COMPARISON" = true ]; then
    echo "    * Auto-detect Multi-ASR: $AUTO_DETECT_MULTI_ASR"
fi
echo "  - Information Extraction: $ENABLE_INFORMATION_EXTRACTION"
echo "  - Page Generation: $ENABLE_PAGE_GENERATION"
echo "  - Evaluation: $ENABLE_EVALUATION"
echo ""
echo "LLM Model Configuration:"
echo "  - Medical Correction Model: $MEDICAL_CORRECTION_MODEL"
echo "  - Extraction Model: $EXTRACTION_MODEL"
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
# Priority 1: asr_transcripts (complete files after merging segments)
if [ -d "$ASR_RESULTS_DIR/asr_transcripts" ]; then
    TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR/asr_transcripts")
# Priority 2: merged_segmented_transcripts (complete files after merging segments)
elif [ -d "$ASR_RESULTS_DIR/merged_segmented_transcripts" ]; then
    TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR/merged_segmented_transcripts")
# Priority 3: merged_transcripts (for long audio splits)
elif [ -d "$ASR_RESULTS_DIR/merged_transcripts" ]; then
    TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR/merged_transcripts")
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
    
    # Check if multi-ASR comparison is enabled
    if [ "$ENABLE_MULTI_ASR_COMPARISON" = true ]; then
        echo "Running multi-ASR comparison (Canary + Whisper) using $MEDICAL_CORRECTION_MODEL..."
        echo "This will compare Canary and Whisper ASR results and provide the best combined version."
    else
    echo "Running medical term correction using $MEDICAL_CORRECTION_MODEL..."
    fi
    echo "Input directories: ${TRANSCRIPT_DIRS[*]}"
    echo "Output: $CORRECTED_TRANSCRIPTS_DIR"
    
    # Choose prompt and processing mode based on whether multi-ASR comparison is enabled
    if [ "$ENABLE_MULTI_ASR_COMPARISON" = true ]; then
        CURRENT_PROMPT="$MULTI_ASR_COMPARISON_PROMPT"
        PROCESSING_MODE="multi_asr_comparison"
        
        # Check if we have organized multi-ASR results
        if [ -d "$OUTPUT_DIR/multi_asr_organized" ] && [ -f "$OUTPUT_DIR/multi_asr_organized/multi_asr_mapping.json" ]; then
            echo "Using organized multi-ASR results for comparison..."
            MULTI_ASR_INPUT_DIR="$OUTPUT_DIR/multi_asr_organized"
        else
            echo "No organized multi-ASR results found, using standard processing..."
            MULTI_ASR_INPUT_DIR=""
        fi
        
        # Auto-detect multi-ASR results if enabled
        if [ "$AUTO_DETECT_MULTI_ASR" = true ]; then
            echo "Auto-detecting multi-ASR results from pipeline output..."
            
            # Try to detect and organize multi-ASR results
            if detect_multi_asr_results "$ASR_RESULTS_DIR" "$OUTPUT_DIR"; then
                echo "Multi-ASR detection successful, using organized results"
                # Update transcript directories to use organized multi-ASR results
                TRANSCRIPT_DIRS=("$OUTPUT_DIR/multi_asr_organized")
                MULTI_ASR_INPUT_DIR="$OUTPUT_DIR/multi_asr_organized"
            else
                echo "Multi-ASR detection failed, falling back to original input"
                echo "WARNING: Multi-ASR detection failed" >> "$ERROR_LOG_FILE"
                echo "  ASR results directory: $ASR_RESULTS_DIR" >> "$ERROR_LOG_FILE"
                echo "  Using original transcript directories" >> "$ERROR_LOG_FILE"
                echo "" >> "$ERROR_LOG_FILE"
            fi
        fi
    elif [ "$ENABLE_ASR_SELECTION" = true ]; then
        echo "Using ASR Selection mode..."
        CURRENT_PROMPT="$ASR_SELECTION_PROMPT"
        PROCESSING_MODE="asr_selection"
        
        # For ASR selection, preserve the original transcript directory
        # (since the script expects files to be in root directory, not in canary/whisper subdirs)
        ORIGINAL_TRANSCRIPT_DIR="${TRANSCRIPT_DIRS[0]}"
        echo "Preserving original transcript directory for ASR selection: $ORIGINAL_TRANSCRIPT_DIR"
        
        # Auto-detect multi-ASR results if enabled (for organization but not for input)
        if [ "$AUTO_DETECT_MULTI_ASR" = true ]; then
            echo "Auto-detecting multi-ASR results from pipeline output..."
            
            # Try to detect and organize multi-ASR results
            if detect_multi_asr_results "$ASR_RESULTS_DIR" "$OUTPUT_DIR"; then
                echo "Multi-ASR detection successful, organized results created"
                echo "Note: For ASR selection, using original flat directory structure"
            else
                echo "Multi-ASR detection failed, falling back to original input"
                echo "WARNING: Multi-ASR detection failed" >> "$ERROR_LOG_FILE"
                echo "  ASR results directory: $ASR_RESULTS_DIR" >> "$ERROR_LOG_FILE"
                echo "  Using original transcript directories" >> "$ERROR_LOG_FILE"
                echo "" >> "$ERROR_LOG_FILE"
            fi
        fi
        
        # For ASR selection, always use original transcript directory
        # (since the script expects files to be in root directory, not in canary/whisper subdirs)
        echo "Using original transcript directory for ASR selection..."
        MULTI_ASR_INPUT_DIR="$ORIGINAL_TRANSCRIPT_DIR"
    else
        CURRENT_PROMPT="$MEDICAL_CORRECTION_PROMPT"
        PROCESSING_MODE="medical_correction"
        MULTI_ASR_INPUT_DIR=""
    fi
    
    # Clear GPU cache before loading model
    clear_gpu_cache
    
    # Special handling for gpt-oss models
    if [ "$MEDICAL_CORRECTION_MODEL" = "gpt-oss-20b" ] || [ "$MEDICAL_CORRECTION_MODEL" = "gpt-oss-120b" ]; then
        echo "Using specialized gpt-oss handler for $MEDICAL_CORRECTION_MODEL..."
        echo "Temperature: $TEMPERATURE, Max New Tokens: $MAX_NEW_TOKENS"
        
        # Set aggressive PyTorch memory allocation config for large models
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
        export CUDA_LAUNCH_BLOCKING=1  # Better error reporting
        
        # Additional memory optimization for gpt-oss models
        echo "Setting memory optimization for gpt-oss model..."
        export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable memory caching
        export OMP_NUM_THREADS=1  # Reduce CPU thread usage
        
        # Use the optimized script for gpt-oss models
        if [ "$MEDICAL_CORRECTION_MODEL" = "gpt-oss-20b" ]; then
            SCRIPT_NAME="llm_gpt_oss_20b_optimized.py"
            # Fallback to original if optimized version doesn't exist
            if [ ! -f "$SCRIPT_NAME" ]; then
                SCRIPT_NAME="llm_gpt_oss_20b.py"
            fi
        else
            SCRIPT_NAME="llm_gpt_oss_120b.py"
        fi
        
        # Check if we should use specialized multi-ASR processing
        if [ "$PROCESSING_MODE" = "multi_asr_comparison" ] && [ -n "$MULTI_ASR_INPUT_DIR" ]; then
            echo "Using specialized multi-ASR comparison script..."
            
            # Use the specialized multi-ASR comparison script
            $PYTHON_EXEC multi_asr_comparison.py \
                --input_dir "$MULTI_ASR_INPUT_DIR" \
                --output_dir "$CORRECTED_TRANSCRIPTS_DIR" \
                --model "$MEDICAL_CORRECTION_MODEL" \
                --device "$DEVICE" \
                --prompt "$CURRENT_PROMPT" \
                --batch_size "$BATCH_SIZE" \
                $([ "$LOAD_IN_8BIT" = "true" ] && echo "--load_in_8bit") \
                $([ "$LOAD_IN_4BIT" = "true" ] && echo "--load_in_4bit") || true

        elif [ "$PROCESSING_MODE" = "asr_selection" ]; then
            echo "Using ASR selection script with standard input..."
            
            # Use the specialized ASR selection script with standard input
            $PYTHON_EXEC asr_selection.py \
                --input_dir "$MULTI_ASR_INPUT_DIR" \
                --output_dir "$CORRECTED_TRANSCRIPTS_DIR" \
                --model "$MEDICAL_CORRECTION_MODEL" \
                --device "$DEVICE" \
                --prompt "$CURRENT_PROMPT" \
                --batch_size "$BATCH_SIZE" \
                $([ "$LOAD_IN_8BIT" = "true" ] && echo "--load_in_8bit") \
                $([ "$LOAD_IN_4BIT" = "true" ] && echo "--load_in_4bit") || true
        else
            # Check if script exists
            if [ ! -f "$SCRIPT_NAME" ]; then
                echo "ERROR: Script $SCRIPT_NAME not found!"
                echo "ERROR: Script $SCRIPT_NAME not found!" >> "$ERROR_LOG_FILE"
                echo "  Expected script: $SCRIPT_NAME" >> "$ERROR_LOG_FILE"
                echo "  Current directory: $(pwd)" >> "$ERROR_LOG_FILE"
                echo "  Available scripts:" >> "$ERROR_LOG_FILE"
                ls -la *.py 2>/dev/null | head -10 >> "$ERROR_LOG_FILE" || echo "  No .py files found" >> "$ERROR_LOG_FILE"
                echo "" >> "$ERROR_LOG_FILE"
                echo "Falling back to llm_local_models.py..."
                
                # Fallback to local models script
                $PYTHON_EXEC llm_local_models.py \
                        --mode "$PROCESSING_MODE" \
                    --input_dirs "${TRANSCRIPT_DIRS[@]}" \
                    --output_dir "$CORRECTED_TRANSCRIPTS_DIR" \
                    --model "$MEDICAL_CORRECTION_MODEL" \
                    --device "$DEVICE" \
                    --batch_size "$BATCH_SIZE" \
                        --prompt "$CURRENT_PROMPT" \
                    --error_log "$ERROR_LOG_FILE" \
                    $([ "$LOAD_IN_8BIT" = "true" ] && echo "--load_in_8bit") \
                    $([ "$LOAD_IN_4BIT" = "true" ] && echo "--load_in_4bit") \
                    ${MODEL_PATH:+--model_path "$MODEL_PATH"} || true
            else
                echo "Running $SCRIPT_NAME..."
                $PYTHON_EXEC $SCRIPT_NAME \
                    "${TRANSCRIPT_DIRS[0]}" \
                    "$CORRECTED_TRANSCRIPTS_DIR" \
                        "$CURRENT_PROMPT" \
                    "$TEMPERATURE" \
                    "$MAX_NEW_TOKENS" || true
            fi
        fi
    else
        # Run medical correction with local model for other models
        # Special handling for ASR selection mode
        if [ "$PROCESSING_MODE" = "asr_selection" ]; then
            echo "Using ASR selection script for non-gpt-oss model..."
            
            # Use the specialized ASR selection script with original transcript directory
            # (since the script expects files to be in root directory, not in canary/whisper subdirs)
            $PYTHON_EXEC asr_selection.py \
                --input_dir "$MULTI_ASR_INPUT_DIR" \
                --output_dir "$CORRECTED_TRANSCRIPTS_DIR" \
                --model "$MEDICAL_CORRECTION_MODEL" \
                --device "$DEVICE" \
                --prompt "$CURRENT_PROMPT" \
                --batch_size "$BATCH_SIZE" \
                $([ "$LOAD_IN_8BIT" = "true" ] && echo "--load_in_8bit") \
                $([ "$LOAD_IN_4BIT" = "true" ] && echo "--load_in_4bit") || true
        else
            $PYTHON_EXEC llm_local_models.py \
                --mode "$PROCESSING_MODE" \
                --input_dirs "${TRANSCRIPT_DIRS[@]}" \
                --output_dir "$CORRECTED_TRANSCRIPTS_DIR" \
                --model "$MEDICAL_CORRECTION_MODEL" \
                --device "$DEVICE" \
                --batch_size "$BATCH_SIZE" \
                --prompt "$CURRENT_PROMPT" \
                --error_log "$ERROR_LOG_FILE" \
                $([ "$LOAD_IN_8BIT" = "true" ] && echo "--load_in_8bit") \
                $([ "$LOAD_IN_4BIT" = "true" ] && echo "--load_in_4bit") \
                ${MODEL_PATH:+--model_path "$MODEL_PATH"} || true
        fi
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

# --- Step 3: Information Extraction (Optional) ---
if [ "$ENABLE_INFORMATION_EXTRACTION" = true ]; then
    echo "--- Step 3: Information Extraction ---"
    EXTRACTION_DIR="$OUTPUT_DIR/extracted_information"
    mkdir -p "$EXTRACTION_DIR"
    
    echo "Extracting structured information using $EXTRACTION_MODEL..."
    echo "Input directories: ${TRANSCRIPT_DIRS[*]}"
    echo "Output: $EXTRACTION_DIR"
    
    # Clear GPU cache before loading model
    clear_gpu_cache
    
    # Special handling for gpt-oss models
    if [ "$EXTRACTION_MODEL" = "gpt-oss-20b" ] || [ "$EXTRACTION_MODEL" = "gpt-oss-120b" ]; then
        echo "Using specialized gpt-oss handler for $EXTRACTION_MODEL..."
        echo "Temperature: $TEMPERATURE, Max New Tokens: $MAX_NEW_TOKENS"
        
        # Set aggressive PyTorch memory allocation config for large models
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
        export CUDA_LAUNCH_BLOCKING=1  # Better error reporting
        
        # Additional memory optimization for gpt-oss models
        echo "Setting memory optimization for gpt-oss model..."
        export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable memory caching
        export OMP_NUM_THREADS=1  # Reduce CPU thread usage
        
        # Use the optimized script for gpt-oss models
        if [ "$EXTRACTION_MODEL" = "gpt-oss-20b" ]; then
            SCRIPT_NAME="llm_gpt_oss_20b_optimized.py"
            # Fallback to original if optimized version doesn't exist
            if [ ! -f "$SCRIPT_NAME" ]; then
                SCRIPT_NAME="llm_gpt_oss_20b.py"
            fi
        else
            SCRIPT_NAME="llm_gpt_oss_120b.py"
        fi
        
        # Check if script exists
        if [ ! -f "$SCRIPT_NAME" ]; then
            echo "ERROR: Script $SCRIPT_NAME not found!"
            echo "ERROR: Script $SCRIPT_NAME not found!" >> "$ERROR_LOG_FILE"
            echo "  Expected script: $SCRIPT_NAME" >> "$ERROR_LOG_FILE"
            echo "  Current directory: $(pwd)" >> "$ERROR_LOG_FILE"
            echo "  Available scripts:" >> "$ERROR_LOG_FILE"
            ls -la *.py 2>/dev/null | head -10 >> "$ERROR_LOG_FILE" || echo "  No .py files found" >> "$ERROR_LOG_FILE"
            echo "" >> "$ERROR_LOG_FILE"
            echo "Falling back to llm_local_models.py..."
            
            # Fallback to local models script
            $PYTHON_EXEC llm_local_models.py \
                --mode information_extraction \
                --input_dirs "${TRANSCRIPT_DIRS[@]}" \
                --output_dir "$EXTRACTION_DIR" \
                --model "$EXTRACTION_MODEL" \
                --device "$DEVICE" \
                --batch_size "$BATCH_SIZE" \
                --prompt "$INFORMATION_EXTRACTION_PROMPT" \
                --error_log "$ERROR_LOG_FILE" \
                $([ "$LOAD_IN_8BIT" = "true" ] && echo "--load_in_8bit") \
                $([ "$LOAD_IN_4BIT" = "true" ] && echo "--load_in_4bit") \
                ${MODEL_PATH:+--model_path "$MODEL_PATH"} || true
        else
            echo "Running $SCRIPT_NAME..."
            $PYTHON_EXEC $SCRIPT_NAME \
                "${TRANSCRIPT_DIRS[0]}" \
                "$EXTRACTION_DIR" \
                "$INFORMATION_EXTRACTION_PROMPT" \
                "$TEMPERATURE" \
                "$MAX_NEW_TOKENS" || true
        fi
    else
        # Run information extraction with specialized script for other models
        $PYTHON_EXEC information_extraction.py \
            --input_dir "${TRANSCRIPT_DIRS[0]}" \
            --output_dir "$EXTRACTION_DIR" \
            --model "$EXTRACTION_MODEL" \
            --device "$DEVICE" \
            --batch_size "$BATCH_SIZE" \
            --prompt "$INFORMATION_EXTRACTION_PROMPT" \
            $([ "$LOAD_IN_8BIT" = "true" ] && echo "--load_in_8bit") \
            $([ "$LOAD_IN_4BIT" = "true" ] && echo "--load_in_4bit") || true
    fi
    
    EXTRACTION_EXIT_CODE=$?
    
    # Check if extraction produced any output files
    EXTRACTION_FILE_COUNT=$(find "$EXTRACTION_DIR" -name "*.txt" 2>/dev/null | wc -l)
    echo "Found $EXTRACTION_FILE_COUNT extracted information files"
    
    if [ $EXTRACTION_EXIT_CODE -eq 0 ] || [ $EXTRACTION_FILE_COUNT -gt 0 ]; then
        if [ $EXTRACTION_EXIT_CODE -eq 0 ]; then
            echo "Information extraction completed successfully"
        else
            echo "Information extraction completed with some failures, but $EXTRACTION_FILE_COUNT files were processed successfully"
        fi
        echo "Extracted information saved to: $EXTRACTION_DIR"
        
        # Update transcript directory for next steps to use extracted information
        TRANSCRIPT_DIRS=("$EXTRACTION_DIR")
    else
        echo "Warning: Information extraction failed completely - no output files generated"
        echo "ERROR: Information extraction failed completely" >> "$ERROR_LOG_FILE"
        echo "  Model: $EXTRACTION_MODEL" >> "$ERROR_LOG_FILE"
        echo "  Input directories: ${TRANSCRIPT_DIRS[*]}" >> "$ERROR_LOG_FILE"
        echo "  Output directory: $EXTRACTION_DIR" >> "$ERROR_LOG_FILE"
        echo "  Continuing with previous transcripts" >> "$ERROR_LOG_FILE"
        echo "" >> "$ERROR_LOG_FILE"
        # Keep using previous transcripts for next steps
    fi
else
    echo "--- Skipping Information Extraction ---"
fi
echo ""



# --- Step 5: Emergency Page Generation (Optional) ---
echo "DEBUG: ENABLE_PAGE_GENERATION = $ENABLE_PAGE_GENERATION"
echo "DEBUG: TRANSCRIPT_DIRS = ${TRANSCRIPT_DIRS[*]}"
if [ "$ENABLE_PAGE_GENERATION" = true ]; then
    echo "--- Step 5: Emergency Page Generation ---"
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
            # Use fixed version for 120b to avoid NoneType errors
            SCRIPT_NAME="llm_gpt_oss_120b_fixed.py"
            if [ ! -f "$SCRIPT_NAME" ]; then
                SCRIPT_NAME="llm_gpt_oss_120b.py"  # Fallback to original
            fi
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

# --- Step 6: Evaluation (Optional) ---
if [ "$ENABLE_EVALUATION" = true ] && [ -n "$GROUND_TRUTH_FILE" ] && [ -f "$GROUND_TRUTH_FILE" ]; then
    echo "--- Step 6: Evaluation of Corrected Results ---"
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
    echo "    * Multi-ASR Comparison: $ENABLE_MULTI_ASR_COMPARISON"
    echo "    * ASR Selection: $ENABLE_ASR_SELECTION"
    if [ "$ENABLE_MULTI_ASR_COMPARISON" = true ]; then
        echo "    * Auto-detect Multi-ASR: $AUTO_DETECT_MULTI_ASR"
    fi
        echo "    * Output: $CORRECTED_TRANSCRIPTS_DIR"
    fi
echo "  - Information Extraction: $ENABLE_INFORMATION_EXTRACTION"
if [ "$ENABLE_INFORMATION_EXTRACTION" = true ]; then
    echo "    * Model: $EXTRACTION_MODEL"
    echo "    * Output: $EXTRACTION_DIR"
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

# Count extracted information
if [ "$ENABLE_INFORMATION_EXTRACTION" = true ] && [ -d "$EXTRACTION_DIR" ]; then
    EXTRACTION_COUNT=$(find "$EXTRACTION_DIR" -name "*.txt" | wc -l)
    echo "  - Extracted information files: $EXTRACTION_COUNT"
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
if [ "$ENABLE_MULTI_ASR_COMPARISON" = true ] && [ "$AUTO_DETECT_MULTI_ASR" = true ]; then
    echo "  $OUTPUT_DIR/multi_asr_organized/       # Organized multi-ASR results"
fi
if [ "$ENABLE_INFORMATION_EXTRACTION" = true ]; then
    echo "  $OUTPUT_DIR/extracted_information/     # Extracted structured information"
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
    ERROR_COUNT=${ERROR_COUNT:-0}
    
    # If there are errors, mark pipeline as failed
    if [ "$ERROR_COUNT" -gt 0 ]; then
        PIPELINE_SUCCESS=false
    fi
else
    ERROR_COUNT=0
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
if [ "$ENABLE_MULTI_ASR_COMPARISON" = true ] && [ "$AUTO_DETECT_MULTI_ASR" = true ]; then
    echo "  $OUTPUT_DIR/multi_asr_organized/       # Organized multi-ASR results"
fi
if [ "$ENABLE_INFORMATION_EXTRACTION" = true ]; then
    echo "  $OUTPUT_DIR/extracted_information/     # Extracted structured information"
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