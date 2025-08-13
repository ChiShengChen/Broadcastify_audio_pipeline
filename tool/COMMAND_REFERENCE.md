# Command Reference Guide

A comprehensive reference for all available parameters and options in the EMS Call ASR and LLM-Enhanced Pipeline.

## 📋 Overview

This guide provides detailed documentation for all command-line parameters, configuration options, and usage examples for both pipeline stages.

## 🎤 Stage 1: ASR Pipeline (`run_pipeline.sh`)

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input_dir DIR` | Audio input directory | `--input_dir "/path/to/audio"` |
| `--ground_truth FILE` | Ground truth CSV file | `--ground_truth "/path/to/gt.csv"` |

### Optional Parameters

#### Output Configuration
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--output_dir DIR` | Output directory | `pipeline_results_YYYYMMDD_HHMMSS` | `--output_dir "/path/to/results"` |

#### Processing Options
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--use-vad` | Enable VAD preprocessing | `false` | `--use-vad` |
| `--disable-vad` | Disable VAD preprocessing | - | `--disable-vad` |
| `--use-long-audio-split` | Enable long audio splitting | `false` | `--use-long-audio-split` |
| `--disable-long-audio-split` | Disable long audio splitting | - | `--disable-long-audio-split` |
| `--max-segment-duration SEC` | Max segment duration (seconds) | `120` | `--max-segment-duration 60` |

#### VAD Configuration
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--vad-threshold FLOAT` | VAD detection threshold | `0.5` | `--vad-threshold 0.3` |
| `--vad-min-speech FLOAT` | Min speech duration (seconds) | `0.25` | `--vad-min-speech 0.2` |
| `--vad-max-speech FLOAT` | Max speech duration (seconds) | `30` | `--vad-max-speech 25` |

#### Audio Processing
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--enable-audio-filter` | Enable audio filtering | `false` | `--enable-audio-filter` |
| `--disable-audio-filter` | Disable audio filtering | - | `--disable-audio-filter` |
| `--filter-mode MODE` | Filter mode | `moderate` | `--filter-mode aggressive` |
| `--use-enhanced-preprocessor` | Enable enhanced preprocessing | `false` | `--use-enhanced-preprocessor` |

#### Ground Truth Processing
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--preprocess-ground-truth` | Enable GT preprocessing | `false` | `--preprocess-ground-truth` |
| `--enhanced-preprocessor-mode MODE` | Preprocessing mode | `moderate` | `--enhanced-preprocessor-mode aggressive` |

#### Performance Options
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--max-workers INT` | Maximum parallel workers | `4` | `--max-workers 2` |
| `--memory-limit SIZE` | Memory limit | `8GB` | `--memory-limit 4GB` |
| `--enable-gpu` | Enable GPU processing | `false` | `--enable-gpu` |
| `--disable-gpu` | Disable GPU processing | - | `--disable-gpu` |

#### Help and Debug
| Parameter | Description | Example |
|-----------|-------------|---------|
| `-h, --help` | Show help message | `--help` |
| `--dry-run` | Validate configuration without processing | `--dry-run` |
| `--validate-config` | Validate configuration | `--validate-config` |
| `--verbose` | Enable verbose output | `--verbose` |

### Filter Mode Options

| Mode | Description | Use Case |
|------|-------------|----------|
| `conservative` | Minimal filtering | High-quality audio |
| `moderate` | Balanced filtering | General use |
| `aggressive` | Maximum noise reduction | Noisy environments |

### Enhanced Preprocessor Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `conservative` | Basic text normalization | Clean transcripts |
| `moderate` | Standard medical preprocessing | General medical text |
| `aggressive` | Extensive normalization | Noisy/inconsistent text |

## 🧠 Stage 2: LLM Pipeline (`run_llm_pipeline.sh`)

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--asr_results_dir DIR` | ASR results directory from Stage 1 | `--asr_results_dir "/path/to/asr_results"` |

### Optional Parameters

#### Output Configuration
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--output_dir DIR` | Output directory | `llm_results_YYYYMMDD_HHMMSS` | `--output_dir "/path/to/llm_results"` |
| `--ground_truth FILE` | Ground truth CSV for evaluation | - | `--ground_truth "/path/to/gt.csv"` |

#### Model Selection
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--medical_correction_model MODEL` | Medical correction model | `gpt-oss-20b` | `--medical_correction_model "BioMistral-7B"` |
| `--page_generation_model MODEL` | Emergency page model | `BioMistral-7B` | `--page_generation_model "Meditron-7B"` |

#### Available Models
- `BioMistral-7B` (recommended for medical tasks)
- `Meditron-7B` (clinical documentation)
- `Llama-3-8B-UltraMedica` (advanced medical reasoning)
- `gpt-oss-20b` (general purpose)
- `gpt-oss-120b` (large-scale reasoning, requires multiple GPUs)

#### Feature Control
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--enable_medical_correction` | Enable medical correction | `true` | `--enable_medical_correction` |
| `--disable_medical_correction` | Disable medical correction | - | `--disable_medical_correction` |
| `--enable_page_generation` | Enable page generation | `true` | `--enable_page_generation` |
| `--disable_page_generation` | Disable page generation | - | `--disable_page_generation` |
| `--enable_evaluation` | Enable evaluation | `true` | `--enable_evaluation` |
| `--disable_evaluation` | Disable evaluation | - | `--disable_evaluation` |
| `--enable_whisper_filter` | Enable Whisper filtering | `true` | `--enable_whisper_filter` |
| `--disable_whisper_filter` | Disable Whisper filtering | - | `--disable_whisper_filter` |

#### LLM Configuration
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--model_path PATH` | Custom model path | - | `--model_path "/path/to/model"` |
| `--device DEVICE` | Processing device | `auto` | `--device "cuda"` |
| `--load_in_8bit` | Enable 8-bit quantization | `false` | `--load_in_8bit` |
| `--load_in_4bit` | Enable 4-bit quantization | `false` | `--load_in_4bit` |
| `--batch_size INT` | Batch size | `5` | `--batch_size 1` |

#### Device Options
- `auto` - Automatically select best available device
- `cpu` - Use CPU processing (slower but universal)
- `cuda` - Use first available GPU
- `cuda:0`, `cuda:1`, etc. - Use specific GPU

#### Prompt Configuration
| Parameter | Description | Example |
|-----------|-------------|---------|
| `--medical_correction_prompt TEXT` | Custom medical correction prompt | `--medical_correction_prompt "Focus on cardiac terms..."` |
| `--page_generation_prompt TEXT` | Custom page generation prompt | `--page_generation_prompt "Generate structured report..."` |

#### Help and Debug
| Parameter | Description | Example |
|-----------|-------------|---------|
| `-h, --help` | Show help message | `--help` |
| `--validate-config` | Validate configuration | `--validate-config` |

## 📝 Usage Examples

### Stage 1: ASR Pipeline Examples

#### Basic Usage
```bash
# Minimal configuration
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv"
```

#### With VAD and Audio Processing
```bash
# Enable preprocessing features
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv" \
    --use-vad \
    --vad-threshold 0.5 \
    --enable-audio-filter \
    --filter-mode moderate
```

#### Long Audio Processing
```bash
# Handle long audio files
./run_pipeline.sh \
    --input_dir "/path/to/long_audio" \
    --ground_truth "/path/to/ground_truth.csv" \
    --use-long-audio-split \
    --max-segment-duration 60
```

#### Advanced Processing
```bash
# Full feature processing
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --output_dir "/path/to/results" \
    --ground_truth "/path/to/ground_truth.csv" \
    --use-vad \
    --vad-threshold 0.4 \
    --vad-min-speech 0.3 \
    --use-long-audio-split \
    --max-segment-duration 90 \
    --preprocess-ground-truth \
    --enhanced-preprocessor-mode aggressive \
    --enable-audio-filter \
    --filter-mode moderate \
    --max-workers 2
```

#### Memory-Constrained Systems
```bash
# Optimize for limited resources
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv" \
    --use-long-audio-split \
    --max-segment-duration 30 \
    --max-workers 1 \
    --memory-limit 2GB \
    --disable-gpu
```

### Stage 2: LLM Pipeline Examples

#### Basic LLM Enhancement
```bash
# Standard configuration
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B"
```

#### With Quantization
```bash
# Memory-optimized processing
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit \
    --device "cuda"
```

#### Medical Correction Only
```bash
# Focus on terminology correction
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --disable_page_generation \
    --load_in_8bit
```

#### Emergency Page Generation Only
```bash
# Generate structured reports only
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --page_generation_model "BioMistral-7B" \
    --disable_medical_correction \
    --load_in_4bit
```

#### Custom Prompts
```bash
# Specialized for cardiac emergencies
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --medical_correction_prompt "Focus on cardiac terminology: arrhythmias, medications, procedures. Correct drug dosages and timing." \
    --page_generation_prompt "CARDIAC EMERGENCY REPORT: RHYTHM STATUS, MEDICATIONS GIVEN, CPR DURATION, TRANSPORT DESTINATION." \
    --load_in_8bit
```

#### Full Pipeline with Evaluation
```bash
# Complete processing with metrics
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --ground_truth "/path/to/ground_truth.csv" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit \
    --device "cuda" \
    --batch_size 1
```

#### CPU-Only Processing
```bash
# For systems without GPU
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --device "cpu" \
    --batch_size 1
```

#### Maximum Memory Savings
```bash
# Minimal memory configuration
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_4bit \
    --batch_size 1 \
    --device "cuda"
```

#### High-Capability Processing (gpt-oss-120b)
```bash
# Maximum model capability (requires multiple high-end GPUs)
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "gpt-oss-120b" \
    --page_generation_model "gpt-oss-120b" \
    --load_in_4bit \
    --batch_size 1 \
    --device "cuda"

# Note: Requires 2+ RTX 4090s with 4-bit quantization
```

## 🔧 Configuration Files

### Environment Variables

#### ASR Pipeline Environment
```bash
# Set environment variables for ASR pipeline
export AUDIO_INPUT_DIR="/path/to/audio"
export GROUND_TRUTH_FILE="/path/to/ground_truth.csv"
export OUTPUT_DIR="/path/to/results"
export USE_VAD=true
export VAD_THRESHOLD=0.5
export MAX_WORKERS=4
```

#### LLM Pipeline Environment
```bash
# Set environment variables for LLM pipeline
export ASR_RESULTS_DIR="/path/to/asr_results"
export MEDICAL_MODEL="BioMistral-7B"
export PAGE_MODEL="BioMistral-7B"
export DEVICE="cuda"
export LOAD_IN_8BIT=true
export BATCH_SIZE=1
```

### Configuration Validation

#### Pre-flight Checks
```bash
# Validate ASR pipeline configuration
./run_pipeline.sh --validate-config \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv"

# Validate LLM pipeline configuration
./run_llm_pipeline.sh --validate-config \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B"
```

#### Dry Run Mode
```bash
# Test configuration without processing
./run_pipeline.sh --dry-run \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv"
```

## 🚀 Performance Optimization Commands

### Resource Monitoring
```bash
# Monitor system resources during processing
watch -n 1 'nvidia-smi; echo ""; free -h; echo ""; df -h'

# Monitor GPU memory usage
watch -n 1 nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Monitor processing logs
tail -f /path/to/results/error_analysis.log
```

### Batch Processing Scripts

#### Process Multiple Directories
```bash
#!/bin/bash
# Process multiple audio directories

for dir in /path/to/audio/*/; do
    echo "Processing $dir"
    ./run_pipeline.sh \
        --input_dir "$dir" \
        --ground_truth "/path/to/ground_truth.csv" \
        --output_dir "/path/to/results/$(basename $dir)"
done
```

#### Chain Pipeline Stages
```bash
#!/bin/bash
# Chain ASR and LLM processing

# Stage 1: ASR Processing
ASR_OUTPUT=$(./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv" | \
    grep "Results saved to:" | awk '{print $4}')

# Stage 2: LLM Enhancement
./run_llm_pipeline.sh \
    --asr_results_dir "$ASR_OUTPUT" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit
```

## 🔍 Debugging and Troubleshooting Commands

### Debug Mode
```bash
# Enable debug output
export DEBUG=1
./run_pipeline.sh --verbose \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv"
```

### System Diagnostics
```bash
# Check system capabilities
python3 -c "
import torch
import transformers
import whisper
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Transformers: {transformers.__version__}')
print('Whisper: Available')
"

# Check model availability
python3 -c "
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained('BioMistral/BioMistral-7B')
    print('✓ BioMistral-7B available')
except:
    print('✗ BioMistral-7B not available')
"
```

### Error Analysis
```bash
# Analyze error logs
grep -E "(ERROR|FAILED)" /path/to/results/error_analysis.log

# Count error types
grep "Error:" /path/to/results/error_analysis.log | sort | uniq -c

# Find problematic files
grep "FAILED FILE:" /path/to/results/error_analysis.log | awk '{print $3}'
```

## 📚 Quick Reference

### Most Common Commands

#### Standard ASR Processing
```bash
./run_pipeline.sh --input_dir "/path/to/audio" --ground_truth "/path/to/gt.csv"
```

#### Standard LLM Enhancement
```bash
./run_llm_pipeline.sh --asr_results_dir "/path/to/asr" --load_in_8bit
```

#### Memory-Optimized Processing
```bash
# ASR with memory optimization
./run_pipeline.sh --input_dir "/path/to/audio" --ground_truth "/path/to/gt.csv" --use-long-audio-split --max-workers 1

# LLM with memory optimization
./run_llm_pipeline.sh --asr_results_dir "/path/to/asr" --load_in_4bit --batch_size 1
```

### Parameter Shortcuts

| Full Parameter | Short Form | Description |
|----------------|------------|-------------|
| `--input_dir` | `-i` | Input directory |
| `--output_dir` | `-o` | Output directory |
| `--ground_truth` | `-g` | Ground truth file |
| `--help` | `-h` | Help message |
| `--verbose` | `-v` | Verbose output |

## 🔗 Related Documentation

- [ASR Pipeline Guide](ASR_PIPELINE_GUIDE.md) - Detailed ASR processing guide
- [LLM Pipeline Guide](LLM_PIPELINE_GUIDE.md) - Detailed LLM enhancement guide
- [Error Handling Guide](ERROR_HANDLING_GUIDE.md) - Troubleshooting reference
- [Model Configuration Guide](MODEL_CONFIG_GUIDE.md) - Model setup details
- [Performance Tuning](PERFORMANCE_TUNING.md) - Optimization strategies

---

**Note**: This reference covers all available parameters for both pipeline stages. For detailed usage instructions and examples, refer to the specific pipeline guides.