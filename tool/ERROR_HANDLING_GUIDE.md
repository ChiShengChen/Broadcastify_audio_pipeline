# Error Handling and Troubleshooting Guide

A comprehensive guide for error handling, troubleshooting, and problem resolution in the EMS Call ASR and LLM-Enhanced Pipeline.

## 📋 Overview

Both pipeline stages (`run_pipeline.sh` and `run_llm_pipeline.sh`) include comprehensive error handling and logging capabilities that automatically detect, record, and analyze various issues encountered during processing.

## 🏗️ Error Handling Architecture

```
Input → Error Detection → Processing → Error Logging → Error Analysis → Resolution
```

## 🔍 Error Detection System

### Stage 1: ASR Pipeline Errors

#### File-Related Errors
- **File not found**: Audio files or ground truth missing
- **Permission denied**: Insufficient file access rights
- **Format errors**: Unsupported audio formats or CSV structure
- **Encoding issues**: Character encoding problems

#### Processing Errors
- **ASR model failures**: Model loading or processing errors
- **VAD processing issues**: Voice activity detection failures
- **Audio preprocessing errors**: Filtering or conversion failures
- **Memory errors**: Out-of-memory conditions

#### Data Quality Errors
- **Empty files**: Zero-length or corrupted files
- **Invalid audio**: Unsupported sample rates or formats
- **Missing ground truth**: Filename mismatches
- **Evaluation errors**: Metric calculation failures

### Stage 2: LLM Pipeline Errors

#### Model-Related Errors
- **Model loading failures**: CUDA, memory, or download issues
- **Quantization errors**: Unsupported quantization settings
- **GPU memory issues**: Insufficient VRAM
- **Model timeout**: Processing timeout exceeded

#### Processing Errors
- **Empty transcripts**: Input files with no content
- **LLM processing failures**: Model inference errors
- **Prompt processing errors**: Invalid or problematic prompts
- **Output generation failures**: File writing or formatting issues

#### Configuration Errors
- **Invalid model names**: Unsupported or misspelled model names
- **Path errors**: Incorrect ASR results directory paths
- **Parameter conflicts**: Conflicting configuration options

## 📝 Error Logging System

### Error Log Structure

Both pipelines generate detailed error logs with the following information:

```
=== Pipeline Error Analysis Log ===
Analysis Date: 2025-08-13 07:47:01 CST
Pipeline Output Directory: /path/to/results
Input Directory: /path/to/input

FAILED FILE: /path/to/file.txt
  Processing Mode: medical_correction
  Model: BioMistral-7B
  Error: Empty or unreadable transcript
  Timestamp: 2025-08-13 07:51:40

ERROR SUMMARY:
  - Total files processed: 150
  - Successful: 147
  - Failed: 3
  - Error types:
    * Empty/unreadable files: 2
    * Model processing failures: 1
```

### Error Categories

| Category | Description | Severity | Stage |
|----------|-------------|----------|-------|
| **FILE_NOT_FOUND** | File or directory missing | High | Both |
| **INVALID_FORMAT** | Incorrect file format | High | Both |
| **ENCODING_ERROR** | File encoding issues | Medium | Both |
| **EMPTY_DATA** | Empty or invalid data | Medium | Both |
| **MODEL_ERROR** | Model loading/processing failure | High | Both |
| **MEMORY_ERROR** | Insufficient memory | High | Both |
| **CUDA_ERROR** | GPU/CUDA related issues | High | LLM |
| **QUANTIZATION_ERROR** | Quantization setup failure | Medium | LLM |
| **TIMEOUT_ERROR** | Processing timeout | Medium | LLM |
| **PERMISSION_ERROR** | File permission issues | High | Both |

## 🚨 Common Errors and Solutions

### Stage 1: ASR Pipeline Errors

#### Audio Format Issues
**Error**: `Unsupported audio format`
```bash
# Solution: Convert to supported format
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Batch conversion
for file in *.mp3; do
    ffmpeg -i "$file" -ar 16000 -ac 1 "${file%.mp3}.wav"
done
```

#### Memory Issues
**Error**: `CUDA out of memory` or `RAM insufficient`
```bash
# Solution: Enable long audio splitting
./run_pipeline.sh --use-long-audio-split --max-segment-duration 60

# Reduce parallel processing
./run_pipeline.sh --max-workers 2
```

#### VAD Processing Failures
**Error**: `VAD processing failed`
```bash
# Solution: Adjust VAD parameters
./run_pipeline.sh \
    --vad-threshold 0.3 \
    --vad-min-speech 0.2 \
    --vad-max-speech 25

# Disable VAD if persistent issues
./run_pipeline.sh --disable-vad
```

#### Ground Truth Issues
**Error**: `Missing ground truth` or `Filename mismatch`
```bash
# Solution: Check filename matching
python3 -c "
import pandas as pd
df = pd.read_csv('ground_truth.csv')
print('Ground truth files:', df['Filename'].tolist())
"

# Check audio files
ls -la /path/to/audio/*.wav
```

### Stage 2: LLM Pipeline Errors

#### CUDA Not Available
**Error**: `Torch not compiled with CUDA enabled`
```bash
# Solution: Install CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### GPU Memory Issues
**Error**: `CUDA out of memory`
```bash
# Solution: Use quantization
./run_llm_pipeline.sh --load_in_8bit  # or --load_in_4bit

# Reduce batch size
./run_llm_pipeline.sh --batch_size 1

# Use CPU processing (slower)
./run_llm_pipeline.sh --device cpu
```

#### Model Loading Failures
**Error**: `Model not found` or `Connection error`
```bash
# Solution: Manual model download
python3 -c "
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('BioMistral/BioMistral-7B')
model = AutoModel.from_pretrained('BioMistral/BioMistral-7B')
print('Model downloaded successfully')
"

# Check internet connection
ping huggingface.co

# Clear model cache if corrupted
rm -rf ~/.cache/huggingface/transformers/
```

#### Empty Transcript Errors
**Error**: `Empty or unreadable transcript`
```bash
# Solution: Check ASR results quality
find /path/to/asr_results -name "*.txt" -empty
find /path/to/asr_results -name "*.txt" -exec wc -l {} \; | awk '$1==0 {print $2}'

# Verify transcript content
head -5 /path/to/asr_results/asr_transcripts/*.txt
```

#### Quantization Issues
**Error**: `bitsandbytes not installed` or `Quantization failed`
```bash
# Solution: Install/upgrade bitsandbytes
pip install bitsandbytes>=0.41.0

# For specific CUDA versions
pip install bitsandbytes-cuda118  # For CUDA 11.8

# Verify installation
python3 -c "import bitsandbytes; print('BitsAndBytes available')"
```

## 🔧 Troubleshooting Tools

### Error Analysis Commands

```bash
# View error summary
cat /path/to/results/error_analysis.log

# Count errors by type
grep "Error:" /path/to/results/error_analysis.log | sort | uniq -c

# Find failed files
grep "FAILED FILE:" /path/to/results/error_analysis.log

# Check processing statistics
grep "Processing:" /path/to/results/*_summary.txt
```

### System Diagnostics

```bash
# Check GPU status
nvidia-smi

# Monitor GPU usage during processing
watch -n 1 nvidia-smi

# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check disk space
df -h /path/to/workspace

# Check memory usage
free -h

# Check Python environment
pip list | grep -E "(torch|transformers|whisper|nemo)"
```

### Log Analysis Tools

```bash
# Analyze error patterns
python3 tool/analyze_errors.py --error_log /path/to/error_analysis.log

# Generate error report
python3 tool/generate_error_report.py --results_dir /path/to/results

# Monitor processing progress
tail -f /path/to/results/error_analysis.log
```

## 📊 Error Analysis and Reporting

### Automated Error Analysis

Both pipelines include automated error analysis that generates:

1. **Error Statistics**: Count and categorization of all errors
2. **Failed File Lists**: Detailed list of problematic files
3. **Error Patterns**: Common error types and frequencies
4. **Resolution Suggestions**: Specific recommendations for each error type

### Error Report Structure

```
=== Error Analysis Report ===
Date: 2025-08-13 08:00:00
Pipeline: LLM Enhancement
Total Files: 150
Successful: 147 (98%)
Failed: 3 (2%)

Error Breakdown:
  - Empty/unreadable files: 2
  - Model processing failures: 1
  - GPU memory issues: 0

Failed Files:
  1. large-v3_file1.txt - Empty or unreadable transcript
  2. large-v3_file2.txt - Empty or unreadable transcript
  3. large-v3_file3.txt - Model correction failed

Recommendations:
  - Check ASR output quality for empty files
  - Verify model configuration for processing failures
  - Consider using quantization for memory issues
```

## 🔍 Performance Troubleshooting

### Slow Processing Issues

#### ASR Pipeline
```bash
# Enable parallel processing
./run_pipeline.sh --max-workers 4

# Use GPU acceleration
./run_pipeline.sh --enable-gpu

# Skip unnecessary preprocessing
./run_pipeline.sh --disable-vad --disable-audio-filter
```

#### LLM Pipeline
```bash
# Use quantization for speed
./run_llm_pipeline.sh --load_in_8bit

# Increase batch size (if memory allows)
./run_llm_pipeline.sh --batch_size 2

# Use faster models
./run_llm_pipeline.sh --medical_correction_model "BioMistral-7B"
```

### Memory Optimization

#### High RAM Usage
```bash
# Process smaller batches
./run_pipeline.sh --batch-size 10

# Enable long audio splitting
./run_pipeline.sh --use-long-audio-split --max-segment-duration 60

# Reduce parallel workers
./run_pipeline.sh --max-workers 2
```

#### High GPU Memory Usage
```bash
# Use 4-bit quantization
./run_llm_pipeline.sh --load_in_4bit

# Reduce batch size
./run_llm_pipeline.sh --batch_size 1

# Process sequentially
./run_llm_pipeline.sh --disable_page_generation  # Process one task at a time
```

## 🛠️ Advanced Troubleshooting

### Environment Issues

#### Python Environment Problems
```bash
# Create clean environment
conda create -n ems_pipeline python=3.8
conda activate ems_pipeline

# Install requirements
pip install -r requirements.txt

# Verify installation
python3 -c "import torch, transformers, whisper; print('All packages available')"
```

#### CUDA Environment Issues
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Reinstall PyTorch for specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test CUDA functionality
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Model-Specific Troubleshooting

#### Whisper Issues
```bash
# Reinstall Whisper
pip uninstall openai-whisper
pip install openai-whisper

# Clear Whisper cache
rm -rf ~/.cache/whisper/

# Test Whisper installation
python3 -c "import whisper; model = whisper.load_model('base'); print('Whisper working')"
```

#### Transformers Issues
```bash
# Update transformers
pip install --upgrade transformers

# Clear transformers cache
rm -rf ~/.cache/huggingface/transformers/

# Test model loading
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BioMistral/BioMistral-7B')"
```

#### NeMo Issues
```bash
# Install NeMo with ASR support
pip install nemo_toolkit[asr]

# Check NeMo installation
python3 -c "import nemo; print('NeMo available')"

# Test NeMo ASR
python3 -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecCTCModel.from_pretrained('nvidia/parakeet-ctc-0.6b')
print('NeMo ASR working')
"
```

## 📚 Error Prevention Best Practices

### Pre-Processing Checks

```bash
# Verify input data before processing
python3 tool/validate_input_data.py \
    --audio_dir /path/to/audio \
    --ground_truth /path/to/ground_truth.csv

# Check system resources
python3 tool/check_system_requirements.py

# Test configuration
./run_pipeline.sh --dry-run --input_dir /path/to/test
```

### Configuration Validation

```bash
# Validate ASR pipeline configuration
./run_pipeline.sh --validate-config

# Validate LLM pipeline configuration
./run_llm_pipeline.sh --validate-config

# Check model availability
python3 tool/check_model_availability.py
```

### Monitoring and Maintenance

```bash
# Set up error monitoring
python3 tool/setup_error_monitoring.py

# Regular health checks
python3 tool/pipeline_health_check.py

# Clean up old error logs
find /path/to/results -name "error_analysis.log" -mtime +30 -delete
```

## 🔗 Related Documentation

- [ASR Pipeline Guide](ASR_PIPELINE_GUIDE.md) - ASR processing details
- [LLM Pipeline Guide](LLM_PIPELINE_GUIDE.md) - LLM enhancement guide
- [Performance Tuning](PERFORMANCE_TUNING.md) - Optimization strategies
- [Model Configuration Guide](MODEL_CONFIG_GUIDE.md) - Model setup details

## 📞 Support and Contact

For persistent issues:

1. **Check error logs**: Review detailed error messages
2. **Verify system requirements**: Ensure all dependencies are met
3. **Test with sample data**: Use provided test datasets
4. **Check documentation**: Review relevant guides
5. **Report issues**: Include error logs and system information

### Error Reporting Template

```
System Information:
- OS: Linux/Windows/macOS
- Python Version: 3.x.x
- CUDA Version: x.x
- GPU Model: 
- Available RAM: 
- Available GPU Memory: 

Error Information:
- Pipeline Stage: ASR/LLM
- Error Message: 
- Error Log: [attach error_analysis.log]
- Configuration Used: [attach command or config]
- Input Data Size: 

Steps to Reproduce:
1. 
2. 
3. 

Expected Behavior:

Actual Behavior:
```

---

**Note**: This guide covers error handling for both pipeline stages. For specific configuration and usage details, refer to the respective pipeline guides.