# Audio Filter + LLM Enhanced Experimental Pipeline

## Overview

The `run_audio_filter_llm_enhanced_exp_pipeline.sh` script is a comprehensive experimental pipeline that combines audio preprocessing, ASR (Automatic Speech Recognition), and LLM (Large Language Model) post-processing for medical speech recognition enhancement.

## Key Features

### 🎯 **Multiple Input Directory Support**
- **Supports multiple input directories** just like `run_pipeline.sh`
- Automatically combines files from multiple directories for processing
- Usage: `--input_dir "/path/to/dir1 /path/to/dir2 /path/to/dir3"`

### 🔧 **Audio Preprocessing Options**
The script offers 5 preprocessing modes that can be switched via parameters:

- **Mode a**: Bandpass filter 300-3400 Hz only
- **Mode b**: Wiener filter with attenuation 0.15-0.3 only  
- **Mode c**: VAD (Voice Activity Detection) only
- **Mode d**: Bandpass + Wiener filter combination
- **Mode e**: No preprocessing (baseline)

### 🎤 **ASR Configuration**
- **Fixed to Whisper Large-v3 only** as requested
- Optimized for medical speech recognition

### 🧠 **LLM Post-processing**
- **Three specialized medical models**:
  - BioMistral-7B
  - Meditron-7B  
  - Llama-3-8B-UltraMedica
- **Bypass option** available with `--disable_llm`
- **Focused on medical term correction** and enhancement

### 📊 **Experimental Matrix**
The script automatically runs all combinations you requested:

1. **`baseline`**: No preprocessing, no post-processing
2. **`preprocessing_only`**: Preprocessing + ASR only
3. **`postprocessing_only`**: ASR + LLM post-processing only  
4. **`preprocessing_postprocessing`**: Full pipeline (preprocessing + ASR + LLM)

### ⏱️ **Comprehensive Evaluation & Timing**
- **WER and BLEU scores** calculation
- **Detailed timing** for each stage:
  - Preprocessing time
  - ASR time
  - LLM processing time
  - Total pipeline time
- **Performance comparison** across all combinations

## Usage Examples

### Basic Usage (Single Directory)
```bash
./run_audio_filter_llm_enhanced_exp_pipeline.sh \
  --input_dir "/media/meow/One Touch/ems_call/random_samples_1" \
  --ground_truth "/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv" \
  --preprocessing_mode a
```

### Multiple Input Directories (Like run_pipeline.sh)
```bash
./run_audio_filter_llm_enhanced_exp_pipeline.sh \
  --input_dir "/media/meow/One Touch/ems_call/random_samples_1 /media/meow/One Touch/ems_call/random_samples_2" \
  --ground_truth "/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv" \
  --preprocessing_mode d \
  --llm_model Meditron-7B
```

### Test Specific Combinations
```bash
./run_audio_filter_llm_enhanced_exp_pipeline.sh \
  --input_dir "/path/to/audio" \
  --ground_truth "/path/to/gt.csv" \
  --preprocessing_mode b \
  --wiener_attenuation 0.25 \
  --run_combinations "preprocessing_postprocessing,baseline"
```

### Disable LLM Post-processing
```bash
./run_audio_filter_llm_enhanced_exp_pipeline.sh \
  --input_dir "/path/to/audio" \
  --ground_truth "/path/to/gt.csv" \
  --preprocessing_mode c \
  --disable_llm
```

## Output Structure

```
output_dir_TIMESTAMP/
├── baseline/
│   ├── asr_results/
│   └── evaluation_results.csv
├── preprocessing_only/
│   ├── preprocessed_audio/
│   ├── asr_results/
│   └── evaluation_results.csv
├── postprocessing_only/
│   ├── asr_results/
│   ├── llm_corrected/
│   └── evaluation_results.csv
├── preprocessing_postprocessing/
│   ├── preprocessed_audio/
│   ├── asr_results/
│   ├── llm_corrected/
│   └── evaluation_results.csv
├── experimental_results_summary.csv
├── timing_results.txt
├── final_experimental_summary.txt
└── error_analysis.log
```

## Key Results Files

1. **`experimental_results_summary.csv`**: Comparison table of all combinations with WER, BLEU, and timing
2. **`final_experimental_summary.txt`**: Comprehensive analysis with best performing combinations
3. **`timing_results.txt`**: Detailed timing breakdown for each stage
4. **`error_analysis.log`**: Error tracking and troubleshooting information

## Parameters Reference

### Required Parameters
- `--input_dir "DIR1 DIR2..."`: Input directory(ies) with audio files
- `--ground_truth FILE`: Ground truth CSV file for evaluation

### Audio Preprocessing Parameters
- `--preprocessing_mode {a,b,c,d,e}`: Preprocessing mode selection
- `--bandpass_lowcut FLOAT`: Low cutoff frequency (default: 300.0 Hz)
- `--bandpass_highcut FLOAT`: High cutoff frequency (default: 3400.0 Hz)  
- `--wiener_attenuation FLOAT`: Wiener filter attenuation (default: 0.2, range: 0.15-0.3)
- `--vad_threshold FLOAT`: VAD speech threshold (default: 0.5)

### LLM Parameters
- `--llm_model {BioMistral-7B,Meditron-7B,Llama-3-8B-UltraMedica}`: Model selection
- `--enable_llm` / `--disable_llm`: Enable/disable LLM post-processing
- `--llm_device {auto,cpu,cuda}`: Processing device
- `--llm_batch_size INT`: Batch size for processing
- `--load_in_8bit` / `--load_in_4bit`: Quantization options

### Experimental Parameters
- `--run_combinations LIST`: Specify which combinations to run
- `--output_dir DIR`: Custom output directory

## Integration with Existing Pipeline

The script is designed to work seamlessly with your existing infrastructure:

- **Compatible with existing Python scripts**: Uses the same `audio_filter.py`, `vad_pipeline.py`, `run_all_asrs.py`, and `llm_local_models.py`
- **Same evaluation framework**: Uses `evaluate_asr.py` for consistent WER/BLEU calculation
- **Multiple directory support**: Handles multiple input directories exactly like `run_pipeline.sh`
- **Error handling**: Robust error logging and recovery mechanisms

## Performance Analysis

The script automatically identifies:
- **Best WER combination**: Lowest Word Error Rate
- **Best BLEU combination**: Highest BLEU score  
- **Fastest processing**: Most time-efficient combination
- **Processing bottlenecks**: Stage-wise timing analysis

This comprehensive experimental framework allows you to systematically evaluate the impact of different preprocessing and post-processing combinations on your medical ASR system performance.