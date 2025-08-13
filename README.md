# EMS Call ASR and LLM-Enhanced Pipeline

A comprehensive two-stage pipeline for emergency medical service (EMS) call analysis, combining Automatic Speech Recognition (ASR) evaluation with Large Language Model (LLM) enhancement for medical term correction and emergency page generation.

## 📋 Overview

This project provides a complete two-stage processing system:

1. **Stage 1: ASR Pipeline** (`run_pipeline.sh`) - Transcribes audio files using multiple ASR models with optional preprocessing
2. **Stage 2: LLM Enhancement** (`run_llm_pipeline.sh`) - Enhances ASR transcripts with medical term correction and emergency page generation

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

The ASR models are automatically configured in `run_all_asrs.py` and executed by the pipeline:

```bash
# ASR models are automatically run - no manual configuration needed
./run_pipeline.sh --input_dir /path/to/audio --output_dir /path/to/results

# All models will generate transcripts with prefixes:
# - large-v3_filename.txt      (Whisper Large-v3)
# - wav2vec-xls-r_filename.txt (Wav2Vec2)
# - canary-1b_filename.txt     (Canary-1B)  
# - parakeet-tdt-0.6b-v2_filename.txt (Parakeet)
```

#### ASR Model Requirements
- **Whisper**: `pip install openai-whisper`
- **Transformers**: `pip install transformers torch torchaudio`
- **NeMo Models**: `pip install nemo_toolkit[asr]`
- **GPU**: CUDA-compatible GPU recommended for faster processing

### Stage 2: LLM Models (`run_llm_pipeline.sh`)

The LLM pipeline supports specialized medical language models for enhancement:

#### Medical Correction Models
| Model | Size | Specialization | Memory (FP16) | Memory (8-bit) | Memory (4-bit) |
|-------|------|----------------|---------------|----------------|----------------|
| **BioMistral-7B** ⭐ | 7B | Medical domain | ~14GB | ~4GB | ~2GB |
| **Meditron-7B** | 7B | Medical literature | ~14GB | ~4GB | ~2GB |
| **Llama-3-8B-UltraMedica** | 8B | Medical fine-tuned | ~16GB | ~4.5GB | ~2.5GB |
| **gpt-oss-20b** | 20B | General purpose | ~40GB | ~12GB | ~6GB |

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
- **BioMistral-7B**: Best overall medical accuracy and terminology
- **Meditron-7B**: Strong medical literature understanding
- **Llama-3-8B-UltraMedica**: Advanced medical reasoning (requires more memory)

**For Emergency Page Generation:**
- **BioMistral-7B**: Excellent structured medical reporting
- **Meditron-7B**: Good clinical documentation
- **gpt-oss-20b**: Best general language capabilities (high memory requirement)

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

### Model Configuration Files

#### ASR Models (`run_all_asrs.py`)
```python
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

### Complete Workflow Example

```bash
# Stage 1: ASR with VAD and preprocessing
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