# EMS Call ASR and LLM-Enhanced Pipeline

A comprehensive two-stage pipeline for emergency medical service (EMS) call analysis, combining Automatic Speech Recognition (ASR) evaluation with Large Language Model (LLM) enhancement for medical term correction and emergency page generation.

## 📋 Overview

This project provides a complete two-stage processing system:

1. **Stage 1: ASR Pipeline** (`run_pipeline.sh`) - Transcribes audio files using multiple ASR models with optional preprocessing
2. **Stage 2: LLM Enhancement** (`run_llm_pipeline.sh`) - Enhances ASR transcripts with medical term correction and emergency page generation

## ⚠️ Important Configuration Requirements

### HuggingFace Authentication (Required for Certain Models)

**🔐 Meditron-7B Model Access**

The `Meditron-7B` model is a **gated repository** on HuggingFace and requires authentication and access approval:

#### Step 1: HuggingFace Login
```bash
# Login to HuggingFace (use your access token)
huggingface-cli login
# or
hf auth login
```

#### Step 2: Request Model Access
1. Visit: https://huggingface.co/epfl-llm/meditron-7b
2. Click **"Request access"** button
3. Fill out the access request form
4. Wait for approval (may take several days)

<!-- #### Alternative: Use Non-Gated Models
If you need immediate access, use these models instead:
- **BioMistral-7B** ⭐ (Recommended) - No authentication required
- **gpt-oss-20b** - No authentication required  
- **gpt-oss-120b** - No authentication required

```bash
# Use BioMistral-7B instead of Meditron-7B
./run_llm_pipeline.sh \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B"
``` -->

**💡 Note**: All models are automatically downloaded from HuggingFace Hub on first use - no manual pre-download required.

---

### Before Running the Pipeline

**You MUST configure the following parameters in the script files before execution:**

#### Stage 1: ASR Pipeline Configuration (`run_pipeline.sh`)

Edit the following variables in `run_pipeline.sh` (lines 17-38):

```bash
# 1. Set your audio input directory(ies)
AUDIO_DIR=("/path/to/your/audio/files")
# Example: AUDIO_DIR=("/media/meow/One Touch/ems_call/audio_samples")

# 2. Set your ground truth CSV file path
GROUND_TRUTH_FILE="/path/to/your/ground_truth.csv"
# Example: GROUND_TRUTH_FILE="/media/meow/One Touch/ems_call/annotations/ground_truth.csv"

# 3. Enable/disable VAD preprocessing (optional)
USE_VAD=false  # Set to true if you want VAD preprocessing
# Example: USE_VAD=true
```

**Required CSV Format for Ground Truth:**
```csv
Filename,transcript
audio_file1.wav,"This is the transcript for file 1"
audio_file2.wav,"This is the transcript for file 2"
```

#### Stage 2: LLM Enhancement Configuration (`run_llm_pipeline.sh`)

Edit the following variables in `run_llm_pipeline.sh` (lines 31-49):

```bash
# 1. Set ASR results directory (output from Stage 1)
ASR_RESULTS_DIR="/path/to/asr/pipeline/results"
# Example: ASR_RESULTS_DIR="/media/meow/One Touch/ems_call/pipeline_results_20240101_120000"

# 2. Set ground truth file (same as Stage 1, optional for evaluation)
GROUND_TRUTH_FILE="/path/to/your/ground_truth.csv"
# Example: GROUND_TRUTH_FILE="/media/meow/One Touch/ems_call/annotations/ground_truth.csv"

# 3. Choose medical correction model
MEDICAL_CORRECTION_MODEL="BioMistral-7B"  # Recommended
# Options: "BioMistral-7B", "Meditron-7B", "Llama-3-8B-UltraMedica", "gpt-oss-20b"

# 4. Choose emergency page generation model  
PAGE_GENERATION_MODEL="BioMistral-7B"     # Recommended
# Options: "BioMistral-7B", "Meditron-7B", "Llama-3-8B-UltraMedica", "gpt-oss-20b"

# 5. Customize prompts (optional) - lines 75-78
MEDICAL_CORRECTION_PROMPT="You are a medical transcription specialist. Please correct any medical terms, drug names, anatomical terms, and medical procedures in the following ASR transcript. Maintain the original meaning and context. Only correct obvious medical errors and standardize medical terminology. Return only the corrected transcript without explanations."

PAGE_GENERATION_PROMPT="You are an emergency medical dispatcher. Based on the following corrected medical transcript, generate a structured emergency page that includes: 1) Patient condition summary, 2) Location details, 3) Required medical resources, 4) Priority level, 5) Key medical information. Format the response as a structured emergency page."
```

### Configuration Steps

#### Step 1: Prepare Your Data
```bash
# 1. Place your .wav audio files in a directory
mkdir -p /path/to/your/audio/files
# Copy your .wav files here

# 2. Create a ground truth CSV file with the required format
# Filename,transcript
# file1.wav,"Medical transcript content here"
# file2.wav,"Another medical transcript"
```

#### Step 2: Configure Stage 1 (ASR Pipeline)
```bash
# Edit run_pipeline.sh
nano run_pipeline.sh

# Update these lines (around lines 17-38):
AUDIO_DIR=("/your/actual/audio/directory")
GROUND_TRUTH_FILE="/your/actual/ground_truth.csv"
USE_VAD=true  # or false, depending on your needs
```

#### Step 3: Run Stage 1
```bash
./run_pipeline.sh
# This will create a results directory like: pipeline_results_YYYYMMDD_HHMMSS/
```

#### Step 4: Configure Stage 2 (LLM Enhancement)
```bash
# Edit run_llm_pipeline.sh  
nano run_llm_pipeline.sh

# Update these lines:
GROUND_TRUTH_FILE="/your/actual/ground_truth.csv"
MEDICAL_CORRECTION_MODEL="BioMistral-7B"
PAGE_GENERATION_MODEL="BioMistral-7B"

# Optional: Customize prompts (around lines 75-78):
MEDICAL_CORRECTION_PROMPT="Your custom medical correction prompt..."
PAGE_GENERATION_PROMPT="Your custom emergency page generation prompt..."
```

#### Step 5: Run Stage 2
```bash
./run_llm_pipeline.sh
# This will create an LLM results directory like: llm_results_YYYYMMDD_HHMMSS/
```

### Alternative: Command-Line Configuration

Instead of editing the script files, you can override the default settings using command-line parameters:

#### Stage 1 Command-Line Override
```bash
./run_pipeline.sh \
    --input_dir "/your/audio/directory" \
    --ground_truth "/your/ground_truth.csv" \
    --output_dir "/your/output/directory" \
    --use-vad  # optional
```

#### Stage 2 Command-Line Override
```bash
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/pipeline_results" \
    --ground_truth "/your/ground_truth.csv" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit \
    --device "cuda"

# With custom prompts
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/pipeline_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --medical_correction_prompt "Your custom medical correction instructions..." \
    --page_generation_prompt "Your custom emergency page generation instructions..." \
    --load_in_8bit \
    --device "cuda"
```

### Quick Validation

Before running the full pipeline, validate your configuration:

```bash
# Check if your audio directory exists and contains .wav files
ls -la /your/audio/directory/*.wav

# Check if your ground truth file exists and has the correct format
head -5 /your/ground_truth.csv

# Verify the CSV has the required columns
head -1 /your/ground_truth.csv | grep -q "Filename,transcript" && echo "✓ CSV format correct" || echo "✗ CSV format incorrect"
```

## 🤖 Available Models and Configuration

### Stage 1: ASR Models (`run_pipeline.sh`)

The ASR pipeline supports multiple state-of-the-art speech recognition models:

#### Supported ASR Models
| Model | Framework | Description | Performance | Configuration |
|-------|-----------|-------------|-------------|---------------|
| **Whisper Large-v3** | OpenAI Whisper | Multilingual speech recognition | High accuracy, robust | `large-v3` |
| **Wav2Vec2** | HuggingFace Transformers | Facebook's self-supervised ASR | Good for English | `facebook/wav2vec2-base-960h` |
| **Canary-1B** | NVIDIA NeMo | Multilingual ASR with punctuation | Enterprise-grade | `nvidia/canary-1b` |
| **Parakeet CTC-0.6B** | NVIDIA NeMo | Streaming ASR model | Low latency | `nvidia/parakeet-ctc-0.6b` |

#### ASR Model Configuration in `run_pipeline.sh`

##### Default Configuration (All Models)
By default, all ASR models are executed by the pipeline:

```bash
# Run all ASR models (default behavior)
./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results

# All models will generate transcripts with prefixes:
# - large-v3_filename.txt      (Whisper Large-v3)
# - wav2vec-xls-r_filename.txt (Wav2Vec2)
# - canary-1b_filename.txt     (Canary-1B)  
# - parakeet-tdt-0.6b-v2_filename.txt (Parakeet)
```

##### Selective Model Execution
You can now choose specific ASR models to run using the `--asr-models` parameter:

```bash
# Run only Whisper Large-v3 (fastest, most accurate)
./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results --asr-models "large-v3"

# Run multiple specific models
./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results --asr-models "large-v3 wav2vec-xls-r"

# Run only NeMo models (enterprise-grade)
./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results --asr-models "canary-1b parakeet-tdt-0.6b-v2"

# Combine with other options
./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results --use-vad --asr-models "large-v3"
```

##### Available Model Identifiers
| Model Name | Identifier | Best For |
|------------|------------|----------|
| **Whisper Large-v3** | `large-v3` | General purpose, multilingual |
| **Wav2Vec2** | `wav2vec-xls-r` | English audio, clean recordings |
| **Canary-1B** | `canary-1b` | Enterprise use, punctuation |
| **Parakeet CTC-0.6B** | `parakeet-tdt-0.6b-v2` | Streaming, low latency |

##### Configuration in Script
You can also set the default models in the script configuration:

```bash
# Edit run_pipeline.sh
ASR_MODELS="large-v3"  # Run only Whisper by default
# or
ASR_MODELS="large-v3 canary-1b"  # Run multiple models by default
# or
ASR_MODELS=""  # Run all models (default behavior)
```

#### ASR Model Requirements
- **Whisper**: `pip install openai-whisper`
- **Transformers**: `pip install transformers torch torchaudio`
- **NeMo Models**: `pip install nemo_toolkit[asr]`
- **GPU**: CUDA-compatible GPU recommended for faster processing

### Stage 2: LLM Models (`run_llm_pipeline.sh`)

The LLM pipeline supports specialized medical language models for enhancement:

#### Medical Correction Models
| Model | Size | Specialization | Access | Memory (FP16) | Memory (8-bit) | Memory (4-bit) |
|-------|------|----------------|--------|---------------|----------------|----------------|
| **BioMistral-7B** ⭐ | 7B | Medical domain | ✅ Public | ~14GB | ~4GB | ~2GB |
| **Meditron-7B** | 7B | Medical literature | 🔐 **Gated** | ~14GB | ~4GB | ~2GB |
| **Llama-3-8B-UltraMedica** | 8B | Medical fine-tuned | ✅ Public | ~16GB | ~4.5GB | ~2.5GB |
| **gpt-oss-20b** | 20B | General purpose | ✅ Public | ~40GB | ~12GB | ~6GB |
| **gpt-oss-120b** | 120B | Large-scale reasoning | ✅ Public | ~240GB | ~70GB | ~35GB |

⭐ **Recommended**: BioMistral-7B offers the best balance of medical accuracy and efficiency.

#### Emergency Page Generation Models
The same models are used for both medical correction and emergency page generation, with specialized prompts:

- **Medical Correction**: Corrects medical terminology, drug names, anatomical terms
- **Emergency Page Generation**: Creates structured emergency reports with patient condition, location, resources, priority

#### LLM Model Configuration in `run_llm_pipeline.sh`

```bash
# Basic configuration with recommended models
./run_llm_pipeline.sh \
    --asr_results_dir /path/to/asr_results \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B"

# Advanced configuration with quantization
./run_llm_pipeline.sh \
    --asr_results_dir /path/to/asr_results \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "Meditron-7B" \
    --load_in_8bit \
    --device "cuda" \
    --batch_size 1

# Memory-optimized configuration
./run_llm_pipeline.sh \
    --asr_results_dir /path/to/asr_results \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_4bit \
    --device "cuda"

# Use different models for different tasks
./run_llm_pipeline.sh \
    --asr_results_dir /path/to/asr_results \
    --medical_correction_model "Meditron-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit
```

#### Model Selection Guidelines

**For Medical Correction:**
- **BioMistral-7B** ⭐: Best overall medical accuracy and terminology (immediate access)
- **Meditron-7B**: Strong medical literature understanding (**requires HuggingFace authentication**)
- **Llama-3-8B-UltraMedica**: Advanced medical reasoning (requires more memory)
- **gpt-oss-20b**: Good general medical capabilities (immediate access)

**For Emergency Page Generation:**
- **BioMistral-7B** ⭐: Excellent structured medical reporting (immediate access)
- **Meditron-7B**: Good clinical documentation (**requires HuggingFace authentication**)
- **gpt-oss-20b**: Best general language capabilities (high memory requirement, immediate access)

**🚨 Authentication Note**: Meditron-7B requires HuggingFace login and access approval. For immediate usage, use BioMistral-7B which offers similar medical performance without authentication requirements.

#### Quantization Options

| Setting | Memory Usage | Speed | Quality | Use Case |
|---------|-------------|--------|---------|----------|
| **No quantization** | 100% | Baseline | Highest | High-end GPUs (24GB+) |
| **8-bit (`--load_in_8bit`)** | ~25% | 1.5-2x faster | Very High | Most GPUs (8GB+) |
| **4-bit (`--load_in_4bit`)** | ~12% | 2-4x faster | High | Low-memory GPUs (4GB+) |

#### LLM Model Requirements
- **Base**: `pip install torch transformers accelerate`
- **Quantization**: `pip install bitsandbytes>=0.41.0`
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **Models are downloaded automatically** from HuggingFace Hub on first use

### Prompt Configuration

#### Default Prompts

The pipeline includes optimized default prompts for medical EMS call processing:

**Medical Correction Prompt (Default):**
```
You are a medical transcription specialist. Please correct any medical terms, drug names, anatomical terms, and medical procedures in the following ASR transcript. Maintain the original meaning and context. Only correct obvious medical errors and standardize medical terminology. Return only the corrected transcript without explanations.
```

**Emergency Page Generation Prompt (Default):**
```
You are an emergency medical dispatcher. Based on the following corrected medical transcript, generate a structured emergency page that includes: 1) Patient condition summary, 2) Location details, 3) Required medical resources, 4) Priority level, 5) Key medical information. Format the response as a structured emergency page.
```

#### Custom Prompt Configuration

You can customize prompts in three ways:

**1. Edit Script Configuration (lines 75-78 in `run_llm_pipeline.sh`):**
```bash
# Edit the default prompts directly in the script
MEDICAL_CORRECTION_PROMPT="Your custom medical correction instructions..."
PAGE_GENERATION_PROMPT="Your custom emergency page instructions..."
```

**2. Command-Line Parameters:**
```bash
./run_llm_pipeline.sh \
    --medical_correction_prompt "Your custom medical prompt..." \
    --page_generation_prompt "Your custom page generation prompt..."
```

**3. Specialized Prompt Examples:**

**For Cardiac Emergencies:**
```bash
--medical_correction_prompt "Focus on cardiac terminology: arrhythmias, medications (beta-blockers, ACE inhibitors), procedures (CPR, defibrillation), and cardiac anatomy. Correct drug dosages and timing."

--page_generation_prompt "CARDIAC EMERGENCY REPORT: RHYTHM STATUS, CARDIAC MEDICATIONS GIVEN, CPR DURATION, DEFIBRILLATION ATTEMPTS, CARDIAC HISTORY, TRANSPORT DESTINATION (PCI-capable facility)."
```

**For Trauma Cases:**
```bash
--medical_correction_prompt "Specialize in trauma terminology: injury mechanisms, anatomical locations, Glasgow Coma Scale, vital signs, trauma interventions. Preserve mechanism of injury details."

--page_generation_prompt "TRAUMA ALERT: MECHANISM OF INJURY, INJURIES IDENTIFIED, VITAL SIGNS, GCS, INTERVENTIONS PERFORMED, TRAUMA CENTER CRITERIA MET, TRANSPORT PRIORITY."
```

**For Pediatric Emergencies:**
```bash
--medical_correction_prompt "Focus on pediatric medical terms: age-appropriate vital signs, pediatric drug dosages, developmental considerations, family dynamics."

--page_generation_prompt "PEDIATRIC EMERGENCY: AGE/WEIGHT, PEDIATRIC VITAL SIGNS, PARENT/GUARDIAN PRESENT, PEDIATRIC-SPECIFIC INTERVENTIONS, PEDIATRIC FACILITY REQUIREMENTS."
```

#### Prompt Best Practices

1. **Be Specific**: Include domain-specific terminology requirements
2. **Maintain Context**: Preserve important contextual information
3. **Format Requirements**: Specify desired output structure
4. **Medical Accuracy**: Emphasize accuracy for medical terminology
5. **EMS Protocols**: Include standard EMS reporting requirements

### Model Configuration Files

#### ASR Models (`run_all_asrs.py`)
```python
# Available ASR models configuration
MODELS = {
    'wav2vec-xls-r': {
        'path': 'facebook/wav2vec2-base-960h',
        'framework': 'transformers'
    },
    'canary-1b': {
        'path': 'nvidia/canary-1b',
        'framework': 'nemo'
    },
    'parakeet-tdt-0.6b-v2': {
        'path': 'nvidia/parakeet-ctc-0.6b',
        'framework': 'nemo'
    },
    'large-v3': {
        'path': 'large-v3',
        'framework': 'whisper'
    }
}

# Model selection usage:
# python3 run_all_asrs.py /path/to/audio                    # Run all models
# python3 run_all_asrs.py /path/to/audio --models large-v3  # Run only Whisper
# python3 run_all_asrs.py /path/to/audio --models "large-v3 canary-1b"  # Run multiple
```

#### Pipeline Configuration (`run_pipeline.sh`)
```bash
# ASR Model Selection Configuration
ASR_MODELS="large-v3"  # Default to Whisper Large-v3 only
# ASR_MODELS="large-v3 canary-1b"  # Multiple models
# ASR_MODELS=""  # All models (original behavior)

# Available model identifiers:
# - wav2vec-xls-r: Facebook Wav2Vec2 (good for English)
# - canary-1b: NVIDIA Canary (multilingual with punctuation)
# - parakeet-tdt-0.6b-v2: NVIDIA Parakeet (streaming, low latency)
# - large-v3: OpenAI Whisper Large-v3 (general purpose, most accurate)
```

#### LLM Models (`run_llm_pipeline.sh`)
```bash
AVAILABLE_MODELS=("gpt-oss-20b" "BioMistral-7B" "Meditron-7B" "Llama-3-8B-UltraMedica")

MODEL_PATHS=(
    "BioMistral-7B:BioMistral/BioMistral-7B"
    "Meditron-7B:epfl-llm/meditron-7b" 
    "Llama-3-8B-UltraMedica:/path/to/llama-3-8b-ultramedica"
    "gpt-oss-20b:/path/to/gpt-oss-20b"
)
```

## 🚀 Quick Start

### Two-Stage Pipeline Execution

```bash
# Stage 1: ASR Processing
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/asr_results \
    --ground_truth /path/to/ground_truth.csv

# Stage 2: LLM Enhancement
./run_llm_pipeline.sh \
    --asr_results_dir /path/to/asr_results \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit \
    --device "cuda"
```

### Complete Workflow Examples

#### Full Pipeline with All Models
```bash
# Stage 1: ASR with VAD and preprocessing (all models)
./run_pipeline.sh \
    --input_dir "/media/meow/One Touch/ems_call/audio_files" \
    --output_dir "/media/meow/One Touch/ems_call/asr_results_$(date +%Y%m%d_%H%M%S)" \
    --ground_truth "/media/meow/One Touch/ems_call/ground_truth.csv" \
    --use-vad \
    --use-long-audio-split \
    --preprocess-ground-truth

# Stage 2: LLM enhancement with quantization
./run_llm_pipeline.sh \
    --asr_results_dir "/media/meow/One Touch/ems_call/asr_results_20240101_120000" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --ground_truth "/media/meow/One Touch/ems_call/ground_truth.csv" \
    --load_in_8bit \
    --device "cuda" \
    --batch_size 1
```

#### Fast Pipeline with Selected Models
```bash
# Stage 1: ASR with only Whisper Large-v3 (fastest)
./run_pipeline.sh \
    --input_dir "/media/meow/One Touch/ems_call/audio_files" \
    --output_dir "/media/meow/One Touch/ems_call/asr_results_$(date +%Y%m%d_%H%M%S)" \
    --ground_truth "/media/meow/One Touch/ems_call/ground_truth.csv" \
    --asr-models "large-v3" \
    --use-vad \
    --preprocess-ground-truth

# Stage 2: LLM enhancement
./run_llm_pipeline.sh \
    --asr_results_dir "/media/meow/One Touch/ems_call/asr_results_20240101_120000" \
    --medical_correction_model "BioMistral-7B" \
    --load_in_8bit
```

#### Enterprise Pipeline with NeMo Models
```bash
# Stage 1: ASR with enterprise-grade NeMo models
./run_pipeline.sh \
    --input_dir "/media/meow/One Touch/ems_call/audio_files" \
    --output_dir "/media/meow/One Touch/ems_call/asr_results_$(date +%Y%m%d_%H%M%S)" \
    --ground_truth "/media/meow/One Touch/ems_call/ground_truth.csv" \
    --asr-models "canary-1b parakeet-tdt-0.6b-v2" \
    --use-enhanced-vad \
    --preprocess-ground-truth

# Stage 2: LLM enhancement with medical specialization
./run_llm_pipeline.sh \
    --asr_results_dir "/media/meow/One Touch/ems_call/asr_results_20240101_120000" \
    --medical_correction_model "Meditron-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit
```

## 🏗️ Data Flow Architecture

```
Audio Files (.wav)
       ↓
┌─────────────────────────────────────────┐
│           STAGE 1: ASR PIPELINE         │
│            (run_pipeline.sh)            │
├─────────────────────────────────────────┤
│ 1. Audio Preprocessing (Optional)       │
│    • Upsampling & Segmentation         │
│    • Audio Filtering                   │
│ 2. VAD Processing (Optional)           │
│    • Speech Segment Extraction         │
│ 3. Long Audio Splitting (Optional)     │
│    • Prevent OOM Issues               │
│ 4. ASR Transcription                   │
│    • Multiple Models (Whisper, etc.)   │
│ 5. Transcript Merging                  │
│ 6. ASR Evaluation                      │
└─────────────────────────────────────────┘
       ↓
ASR Results Directory
├── asr_transcripts/
│   ├── large-v3_file1.txt
│   ├── large-v3_file2.txt
│   └── [other_model]_file.txt
├── merged_transcripts/
├── asr_evaluation_results.csv
└── pipeline_summary.txt
       ↓
┌─────────────────────────────────────────┐
│          STAGE 2: LLM PIPELINE          │
│           (run_llm_pipeline.sh)         │
├─────────────────────────────────────────┤
│ 1. Whisper Filtering (Optional)        │
│    • Extract Whisper Results Only      │
│ 2. Medical Term Correction             │
│    • LLM-based Medical Enhancement     │
│ 3. Emergency Page Generation           │
│    • Structured Emergency Reports      │
│ 4. Enhanced Evaluation                 │
└─────────────────────────────────────────┘
       ↓
LLM Results Directory
├── whisper_filtered/
├── corrected_transcripts/
├── emergency_pages/
├── llm_enhanced_evaluation_results.csv
├── error_analysis.log
└── llm_enhanced_pipeline_summary.txt
```

## 🔧 Stage 1: ASR Pipeline (`run_pipeline.sh`)

### Input Requirements
- **Audio Files**: `.wav` format audio files
- **Ground Truth**: CSV file with `Filename` and `transcript` columns
- **Configuration**: Processing parameters

### Key Features
- **Multi-model ASR**: Whisper Large-v3, Wav2Vec2, Parakeet, Canary-1B
- **VAD Preprocessing**: Optional voice activity detection
- **Long Audio Processing**: Automatic segmentation for large files
- **Audio Enhancement**: Filtering, upsampling, noise reduction
- **Ground Truth Preprocessing**: Intelligent text normalization

### Output Structure
```
pipeline_results_YYYYMMDD_HHMMSS/
├── preprocessed_audio/          # Audio preprocessing results
├── filtered_audio/              # Audio filtering results  
├── vad_segments/               # VAD extracted speech segments
├── long_audio_segments/        # Long audio split segments
├── asr_transcripts/            # Raw ASR transcription results
│   ├── large-v3_file1.txt      # Whisper Large-v3 results
│   ├── large-v3_file2.txt
│   ├── wav2vec2_file1.txt      # Wav2Vec2 results
│   ├── parakeet_file1.txt      # Parakeet results
│   └── canary-1b_file1.txt     # Canary-1B results
├── merged_transcripts/         # Merged transcripts for evaluation
├── asr_evaluation_results.csv  # ASR performance metrics
├── model_file_analysis.txt     # Model processing analysis
├── error_analysis.log          # Error tracking and analysis
└── pipeline_summary.txt        # Complete processing summary
```

### Usage Examples

#### Basic ASR Processing
```bash
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv
```

#### Advanced Processing with VAD and Filtering
```bash
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv \
    --use-vad \
    --use-long-audio-split \
    --max-segment-duration 120 \
    --preprocess-ground-truth \
    --use-enhanced-preprocessor \
    --enhanced-preprocessor-mode aggressive
```

## 🔧 Stage 2: LLM Pipeline (`run_llm_pipeline.sh`)

### Input Requirements
- **ASR Results Directory**: Output from Stage 1 (`run_pipeline.sh`)
- **Ground Truth**: Same CSV file used in Stage 1 (optional, for evaluation)
- **LLM Configuration**: Model selection and quantization settings

### Key Features
- **Medical Term Correction**: LLM-based medical terminology enhancement
- **Emergency Page Generation**: Structured emergency report creation
- **Multiple LLM Models**: BioMistral-7B, Meditron-7B, Llama-3-8B-UltraMedica
- **Model Quantization**: 4-bit and 8-bit quantization for memory efficiency
- **Customizable Prompts**: Configurable prompts for both medical correction and emergency page generation
- **Error Tracking**: Detailed logging of failed files and processing issues

### LLM Models and Quantization

#### Available Models
- **BioMistral-7B**: Medical domain specialized model (recommended)
- **Meditron-7B**: Medical language model
- **Llama-3-8B-UltraMedica**: Medical fine-tuned Llama model
- **gpt-oss-20b**: General purpose large model

#### Quantization Options
```bash
# No quantization (highest quality, most memory)
./run_llm_pipeline.sh --asr_results_dir /path/to/asr

# 8-bit quantization (recommended balance)
./run_llm_pipeline.sh --asr_results_dir /path/to/asr --load_in_8bit

# 4-bit quantization (maximum memory savings)
./run_llm_pipeline.sh --asr_results_dir /path/to/asr --load_in_4bit
```

#### Memory Requirements
| Configuration | GPU Memory | Performance | Quality |
|---------------|------------|-------------|---------|
| **No Quantization** | ~14GB | Baseline | Highest |
| **8-bit Quantization** | ~4GB | 1.5-2x faster | Very High |
| **4-bit Quantization** | ~2GB | 2-4x faster | High |

### Output Structure
```
llm_results_YYYYMMDD_HHMMSS/
├── whisper_filtered/                    # Filtered Whisper transcripts only
│   ├── large-v3_file1.txt
│   ├── large-v3_file2.txt
│   └── large-v3_file3.txt
├── corrected_transcripts/               # Medical term corrected transcripts
│   ├── large-v3_file1.txt              # Enhanced medical terminology
│   ├── large-v3_file2.txt
│   ├── large-v3_file3.txt
│   └── local_medical_correction_summary.json
├── emergency_pages/                     # Generated emergency pages
│   ├── large-v3_file1_emergency_page.txt
│   ├── large-v3_file2_emergency_page.txt
│   ├── large-v3_file3_emergency_page.txt
│   └── local_emergency_page_summary.json
├── llm_enhanced_evaluation_results.csv # Enhanced evaluation metrics
├── error_analysis.log                  # Detailed error tracking
└── llm_enhanced_pipeline_summary.txt   # Processing summary
```

### Usage Examples

#### Basic LLM Enhancement
```bash
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B"
```

#### Medical Correction Only
```bash
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --disable_page_generation \
    --load_in_8bit \
    --device "cuda"
```

#### Emergency Page Generation Only
```bash
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --page_generation_model "BioMistral-7B" \
    --disable_medical_correction \
    --load_in_4bit \
    --device "cuda"
```

#### Full Pipeline with Evaluation
```bash
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --ground_truth "/path/to/ground_truth.csv" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit \
    --device "cuda" \
    --batch_size 1
```

#### Custom Prompts Configuration
```bash
# Using custom prompts for specialized medical domains
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --medical_correction_prompt "You are a specialized emergency medicine transcriptionist. Focus on correcting cardiac, respiratory, and trauma-related medical terminology. Preserve all timestamps and speaker identifications. Return only the corrected transcript." \
    --page_generation_prompt "Generate a structured EMS dispatch report with: PRIORITY LEVEL, CHIEF COMPLAINT, PATIENT STATUS, LOCATION, RESOURCES REQUESTED, SPECIAL CONSIDERATIONS. Use clear medical terminology and standard EMS protocols." \
    --load_in_8bit \
    --device "cuda"

# Prompts for different medical specialties
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "Meditron-7B" \
    --medical_correction_prompt "Focus on pediatric emergency terminology. Correct age-specific medical terms, dosages, and procedures. Maintain family/guardian context." \
    --page_generation_model "BioMistral-7B" \
    --page_generation_prompt "Generate pediatric emergency page: AGE/WEIGHT, CHIEF COMPLAINT, VITAL SIGNS, PARENT/GUARDIAN INFO, PEDIATRIC RESOURCES NEEDED, TRANSPORT PRIORITY." \
    --load_in_4bit
```

## 📊 File Flow and Data Processing

### Stage 1 → Stage 2 Data Flow

1. **ASR Transcripts Generation** (Stage 1)
   ```
   Audio Files → ASR Models → Raw Transcripts
   ```

2. **Whisper Filtering** (Stage 2 - Optional)
   ```
   All ASR Transcripts → Filter → Whisper-only Transcripts
   ```

3. **Medical Term Correction** (Stage 2)
   ```
   Raw/Filtered Transcripts → LLM Processing → Corrected Transcripts
   ```

4. **Emergency Page Generation** (Stage 2)
   ```
   Corrected Transcripts → LLM Processing → Emergency Pages
   ```

### Key File Types and Formats

#### Input Files
- **Audio**: `.wav` format, preferably 16kHz sampling rate
- **Ground Truth**: CSV with columns `Filename`, `transcript`
- **Configuration**: Command-line parameters

#### Intermediate Files
- **ASR Transcripts**: `.txt` files named `[model]_[filename].txt`
- **Processing Metadata**: JSON files with processing statistics
- **Error Logs**: Detailed error tracking and analysis

#### Output Files
- **Evaluation Results**: CSV files with WER, MER, WIL metrics
- **Corrected Transcripts**: Enhanced medical terminology
- **Emergency Pages**: Structured emergency reports
- **Summary Reports**: Processing statistics and results

### Error Handling and Logging

Both pipelines include comprehensive error handling:

#### Stage 1 Error Types
- Audio file processing failures
- ASR model execution errors
- VAD processing issues
- File I/O problems

#### Stage 2 Error Types
- Empty or unreadable transcript files
- LLM model loading failures
- GPU memory issues
- Model processing failures

#### Error Analysis Features
```bash
# View error summary
cat /path/to/results/error_analysis.log

# Count failed files
grep -c "FAILED FILE:" /path/to/results/error_analysis.log

# Analyze error types
grep "Error:" /path/to/results/error_analysis.log | sort | uniq -c
```

## 🔧 Requirements

### System Requirements
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA support (for LLM processing)
- **Memory**: 8GB+ RAM, 4GB+ GPU memory
- **Storage**: Sufficient space for audio files and results

### Python Dependencies

#### Option 1: Install from requirements.txt (Recommended)
```bash
# Install all dependencies at once
pip install -r requirements.txt

# For CUDA support (recommended for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### Option 2: Manual Installation
```bash
# Core dependencies for ASR pipeline
pip install pandas matplotlib seaborn psutil pynvml librosa
pip install torch torchaudio transformers openai-whisper
pip install "nemo_toolkit[asr]" pathlib2 tqdm datasets soundfile pydub jiwer

# Additional dependencies for LLM pipeline
pip install accelerate bitsandbytes>=0.41.0 requests
pip install sentencepiece protobuf  # Optional performance enhancements
```

### Hardware Recommendations

#### For ASR Processing (Stage 1)
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ for large audio files
- **Storage**: SSD recommended for faster I/O

#### For LLM Processing (Stage 2)
- **GPU**: NVIDIA GPU with 8GB+ VRAM
  - RTX 4070/4080/4090 (recommended)
  - RTX 3080/3090
  - Tesla V100, A100
- **RAM**: 16GB+ system RAM
- **Storage**: Fast SSD for model loading

## 🧪 Testing and Validation

### Pipeline Testing
```bash
# Test ASR pipeline
cd unit_test
python3 test_pipeline_status.py

# Test LLM components
python3 test_local_models.py

# Test error handling
python3 test_error_handling.py
```

### Quality Validation
```bash
# Validate ASR results
python3 tool/analyze_model_files_enhanced.py \
    --transcript_dir /path/to/transcripts \
    --ground_truth_file /path/to/ground_truth.csv

# Check LLM enhancement quality
python3 tool/analyze_llm_results.py \
    --original_dir /path/to/original \
    --corrected_dir /path/to/corrected
```

## 📚 Documentation

### Detailed Guides
- [ASR Pipeline Guide](tool/ASR_PIPELINE_GUIDE.md) - Comprehensive ASR processing documentation
- [LLM Pipeline Guide](tool/LLM_PIPELINE_GUIDE.md) - LLM enhancement and quantization guide
- [Error Handling Guide](tool/ERROR_HANDLING_GUIDE.md) - Troubleshooting and error resolution
- [Model Configuration Guide](tool/MODEL_CONFIG_GUIDE.md) - LLM model setup and optimization

### Quick References
- [Command Reference](tool/COMMAND_REFERENCE.md) - All available parameters and options
- [File Format Guide](tool/FILE_FORMAT_GUIDE.md) - Input/output file specifications
- [Performance Tuning](tool/PERFORMANCE_TUNING.md) - Optimization tips and best practices

### Legacy Documentation
- [VAD Integration Summary](tool/VAD_INTEGRATION_SUMMARY.md) - Voice Activity Detection integration
- [VAD README](tool/VAD_README.md) - Voice Activity Detection detailed guide
- [Ground Truth Preprocessing Guide](tool/GROUND_TRUTH_PREPROCESSING_GUIDE.md) - Text preprocessing methods
- [Enhanced Preprocessor Usage Guide](tool/ENHANCED_PREPROCESSOR_USAGE_GUIDE.md) - Advanced preprocessing
- [Comprehensive Text Preprocessing Guide](tool/COMPREHENSIVE_TEXT_PREPROCESSING_GUIDE.md) - Complete preprocessing reference
- [ASR Number Processing Analysis](tool/ASR_NUMBER_PROCESSING_ANALYSIS.md) - Number handling analysis
- [Pipeline Status Guide](tool/PIPELINE_STATUS_GUIDE.md) - Pipeline monitoring and status

## 🚨 Troubleshooting

### Common Issues

#### Stage 1 (ASR Pipeline)
```bash
# Audio file format issues
# Solution: Convert to 16kHz WAV format
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# VAD processing failures
# Solution: Adjust VAD thresholds
./run_pipeline.sh --vad-threshold 0.3 --vad-min-speech 0.3

# Memory issues with long audio
# Solution: Enable long audio splitting
./run_pipeline.sh --use-long-audio-split --max-segment-duration 60
```

#### Stage 2 (LLM Pipeline)
```bash
# GPU memory issues
# Solution: Use quantization
./run_llm_pipeline.sh --load_in_8bit  # or --load_in_4bit

# CUDA not available
# Solution: Use CPU processing
./run_llm_pipeline.sh --device "cpu"

# Model loading failures
# Solution: Check model availability and paths
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BioMistral/BioMistral-7B')"
```

### Performance Optimization

#### Memory Optimization
```bash
# Reduce batch size
./run_llm_pipeline.sh --batch_size 1

# Use 4-bit quantization
./run_llm_pipeline.sh --load_in_4bit

# Process only essential features
./run_llm_pipeline.sh --disable_page_generation
```

#### Speed Optimization
```bash
# Use 8-bit quantization (best balance)
./run_llm_pipeline.sh --load_in_8bit

# Parallel processing (if memory allows)
./run_llm_pipeline.sh --batch_size 2

# GPU acceleration
./run_llm_pipeline.sh --device "cuda"
```

## 🤝 Contributing

1. **Testing**: Add tests for new features in `unit_test/`
2. **Documentation**: Update relevant guides in `tool/`
3. **Error Handling**: Ensure comprehensive error logging
4. **Performance**: Consider memory and speed optimizations

## 📄 License

This project is for research and development purposes in emergency medical service analysis.

## 📞 Support

For issues and questions:
1. Check the error logs: `error_analysis.log`
2. Review the documentation in `tool/`
3. Run diagnostic tests in `unit_test/`
4. Check system requirements and dependencies

---

**Note**: This pipeline is specifically designed for emergency medical service call analysis and includes specialized preprocessing for medical terminology and emergency response protocols.
