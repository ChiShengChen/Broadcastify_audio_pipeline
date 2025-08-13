# LLM Pipeline Guide

A comprehensive guide to the Large Language Model (LLM) enhancement pipeline for emergency medical service call analysis.

## 📋 Overview

The LLM Pipeline (`run_llm_pipeline.sh`) is the second stage of the EMS call analysis system. It enhances ASR transcripts using specialized medical language models for terminology correction and emergency page generation.

## 🏗️ Pipeline Architecture

```
ASR Results → Whisper Filter → Medical Correction → Emergency Page Generation → Evaluation → Output
```

### Processing Stages

1. **Whisper Filtering** (Optional)
   - Extracts only Whisper model results
   - Focuses on highest quality transcripts
   - Reduces processing overhead

2. **Medical Term Correction**
   - LLM-based medical terminology enhancement
   - Drug name standardization
   - Anatomical term correction
   - Procedure name normalization

3. **Emergency Page Generation**
   - Structured emergency report creation
   - Patient condition summarization
   - Resource requirement identification
   - Priority level assessment

4. **Enhanced Evaluation**
   - Compares enhanced vs. original transcripts
   - Medical terminology accuracy metrics
   - Clinical relevance scoring

## 🤖 Supported LLM Models

### Medical Language Models

| Model | Size | Specialization | Memory (FP16) | Memory (8-bit) | Memory (4-bit) | Recommended Use |
|-------|------|----------------|---------------|----------------|----------------|-----------------|
| **BioMistral-7B** ⭐ | 7B | Medical domain | ~14GB | ~4GB | ~2GB | General medical, recommended |
| **Meditron-7B** | 7B | Medical literature | ~14GB | ~4GB | ~2GB | Clinical documentation |
| **Llama-3-8B-UltraMedica** | 8B | Medical fine-tuned | ~16GB | ~4.5GB | ~2.5GB | Advanced medical reasoning |
| **gpt-oss-20b** | 20B | General purpose | ~40GB | ~12GB | ~6GB | Complex language tasks |
| **gpt-oss-120b** | 120B | Large-scale reasoning | ~240GB | ~70GB | ~35GB | Maximum capability, research use |

⭐ **Recommended**: BioMistral-7B offers the best balance of medical accuracy and efficiency.

### Model Capabilities

#### BioMistral-7B
- **Strengths**: Medical terminology, drug interactions, clinical protocols
- **Best for**: Emergency medical terminology, standard medical abbreviations
- **Training**: Medical literature, clinical notes, medical textbooks

#### Meditron-7B
- **Strengths**: Medical literature understanding, clinical documentation
- **Best for**: Detailed medical reports, complex medical reasoning
- **Training**: PubMed abstracts, medical journals, clinical guidelines

#### Llama-3-8B-UltraMedica
- **Strengths**: Advanced medical reasoning, contextual understanding
- **Best for**: Complex medical scenarios, multi-step medical reasoning
- **Training**: Medical conversations, clinical case studies, medical Q&A

## ⚙️ Configuration

### Basic Configuration

Edit `run_llm_pipeline.sh` (lines 31-78):

```bash
# Required: ASR results directory (from Stage 1)
ASR_RESULTS_DIR="/path/to/pipeline_results_YYYYMMDD_HHMMSS"

# Optional: Ground truth file for evaluation
GROUND_TRUTH_FILE="/path/to/ground_truth.csv"

# Model Selection
MEDICAL_CORRECTION_MODEL="BioMistral-7B"
PAGE_GENERATION_MODEL="BioMistral-7B"

# Quantization Options
LOAD_IN_8BIT=false  # Recommended for most systems
LOAD_IN_4BIT=false  # For memory-constrained systems

# Processing Configuration
DEVICE="auto"       # auto, cpu, cuda
BATCH_SIZE=1        # Files processed simultaneously
```

### Advanced Configuration

```bash
# Feature Switches
ENABLE_MEDICAL_CORRECTION=true
ENABLE_PAGE_GENERATION=true
ENABLE_EVALUATION=true
ENABLE_WHISPER_FILTER=true

# Performance Options
MAX_RETRIES=3
REQUEST_TIMEOUT=60
USE_LOCAL_MODELS=true

# Custom Model Paths
MODEL_PATHS=(
    "BioMistral-7B:BioMistral/BioMistral-7B"
    "Meditron-7B:epfl-llm/meditron-7b"
    "Llama-3-8B-UltraMedica:/path/to/local/model"
    "gpt-oss-20b:openai/gpt-oss-20b"
    "gpt-oss-120b:openai/gpt-oss-120b"
)
```

## 🔧 Quantization Guide

### Understanding Quantization

Quantization reduces model precision to save memory and increase speed:

| Quantization | Precision | Memory Usage | Speed | Quality | Use Case |
|-------------|-----------|--------------|--------|---------|----------|
| **None** | FP16/FP32 | 100% | Baseline | Highest | High-end GPUs (24GB+) |
| **8-bit** | INT8 | ~25% | 1.5-2x faster | Very High | Most GPUs (8GB+) |
| **4-bit** | INT4 | ~12% | 2-4x faster | High | Low-memory GPUs (4GB+) |

### Quantization Selection Guide

#### No Quantization
```bash
# Highest quality, most memory
./run_llm_pipeline.sh --asr_results_dir /path/to/asr
```
**When to use**: Research environments, high-end hardware, maximum accuracy required

#### 8-bit Quantization (Recommended)
```bash
# Best balance of quality and efficiency
./run_llm_pipeline.sh --asr_results_dir /path/to/asr --load_in_8bit
```
**When to use**: Production environments, RTX 3080/4070+, balanced performance

#### 4-bit Quantization
```bash
# Maximum memory savings
./run_llm_pipeline.sh --asr_results_dir /path/to/asr --load_in_4bit
```
**When to use**: Limited GPU memory, GTX 1660/RTX 3060, mobile deployment

### Hardware Recommendations

| GPU Model | VRAM | Recommended Quantization | Max Model Size |
|-----------|------|-------------------------|----------------|
| RTX 4090 | 24GB | None/8-bit | 20B models |
| RTX 4080 | 16GB | 8-bit | 8B models |
| RTX 4070 | 12GB | 8-bit | 7B models |
| RTX 3080 | 10GB | 8-bit | 7B models |
| RTX 3060 | 8GB | 4-bit | 7B models |
| GTX 1660 | 6GB | 4-bit | 7B models |

**Note**: For gpt-oss-120b (120B parameters), you need:
- **Multiple RTX 4090s**: 4-bit quantization (~35GB) requires 2+ GPUs
- **High-end workstations**: 8-bit quantization (~70GB) requires 3+ RTX 4090s  
- **Research clusters**: Full precision (~240GB) requires 10+ RTX 4090s

## 🚀 Usage Examples

### Basic LLM Enhancement

```bash
# Minimal configuration with recommended settings
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit \
    --device "cuda"
```

### Medical Correction Only

```bash
# Focus on terminology correction
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --disable_page_generation \
    --load_in_8bit
```

### Emergency Page Generation Only

```bash
# Generate structured emergency reports
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --page_generation_model "BioMistral-7B" \
    --disable_medical_correction \
    --load_in_4bit
```

### Complete Pipeline with Evaluation

```bash
# Full processing with performance metrics
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --ground_truth "/path/to/ground_truth.csv" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit \
    --device "cuda" \
    --batch_size 1
```

### Memory-Optimized Processing

```bash
# For systems with limited GPU memory
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_4bit \
    --device "cuda" \
    --batch_size 1
```

### High-Capability Processing (gpt-oss-120b)

```bash
# Maximum model capability (requires multiple high-end GPUs)
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "gpt-oss-120b" \
    --page_generation_model "gpt-oss-120b" \
    --load_in_4bit \
    --device "cuda" \
    --batch_size 1

# Note: Requires 2+ RTX 4090s with 4-bit quantization
# Monitor GPU memory usage: watch -n 1 nvidia-smi
```

## 💬 Prompt Engineering

### Default Prompts

#### Medical Correction Prompt
```
You are a medical transcription specialist. Please correct any medical terms, drug names, anatomical terms, and medical procedures in the following ASR transcript. Maintain the original meaning and context. Only correct obvious medical errors and standardize medical terminology. Return only the corrected transcript without explanations.
```

#### Emergency Page Generation Prompt
```
You are an emergency medical dispatcher. Based on the following corrected medical transcript, generate a structured emergency page that includes: 1) Patient condition summary, 2) Location details, 3) Required medical resources, 4) Priority level, 5) Key medical information. Format the response as a structured emergency page.
```

### Custom Prompt Configuration

#### Method 1: Edit Script Configuration
```bash
# Edit run_llm_pipeline.sh (lines 75-78)
MEDICAL_CORRECTION_PROMPT="Your custom medical correction prompt..."
PAGE_GENERATION_PROMPT="Your custom emergency page prompt..."
```

#### Method 2: Command-Line Parameters
```bash
./run_llm_pipeline.sh \
    --medical_correction_prompt "Custom medical prompt..." \
    --page_generation_prompt "Custom page generation prompt..."
```

### Specialized Prompts

#### Cardiac Emergency Prompts
```bash
--medical_correction_prompt "Focus on cardiac terminology: arrhythmias, medications (beta-blockers, ACE inhibitors), procedures (CPR, defibrillation), and cardiac anatomy. Correct drug dosages and timing."

--page_generation_prompt "CARDIAC EMERGENCY REPORT: RHYTHM STATUS, CARDIAC MEDICATIONS GIVEN, CPR DURATION, DEFIBRILLATION ATTEMPTS, CARDIAC HISTORY, TRANSPORT DESTINATION (PCI-capable facility)."
```

#### Trauma Case Prompts
```bash
--medical_correction_prompt "Specialize in trauma terminology: injury mechanisms, anatomical locations, Glasgow Coma Scale, vital signs, trauma interventions. Preserve mechanism of injury details."

--page_generation_prompt "TRAUMA ALERT: MECHANISM OF INJURY, INJURIES IDENTIFIED, VITAL SIGNS, GCS, INTERVENTIONS PERFORMED, TRAUMA CENTER CRITERIA MET, TRANSPORT PRIORITY."
```

#### Pediatric Emergency Prompts
```bash
--medical_correction_prompt "Focus on pediatric medical terms: age-appropriate vital signs, pediatric drug dosages, developmental considerations, family dynamics."

--page_generation_prompt "PEDIATRIC EMERGENCY: AGE/WEIGHT, PEDIATRIC VITAL SIGNS, PARENT/GUARDIAN PRESENT, PEDIATRIC-SPECIFIC INTERVENTIONS, PEDIATRIC FACILITY REQUIREMENTS."
```

### Prompt Best Practices

1. **Be Specific**: Include domain-specific requirements
2. **Maintain Context**: Preserve important medical context
3. **Format Requirements**: Specify desired output structure
4. **Medical Accuracy**: Emphasize accuracy for medical terminology
5. **EMS Protocols**: Include standard EMS reporting requirements
6. **Length Control**: Keep prompts concise but comprehensive
7. **Example Integration**: Include examples when helpful

## 📁 Input Requirements

### ASR Results Directory

Expected structure from Stage 1:
```
pipeline_results_YYYYMMDD_HHMMSS/
├── asr_transcripts/
│   ├── large-v3_file1.txt
│   ├── large-v3_file2.txt
│   └── [other_model]_file.txt
├── merged_transcripts/
└── pipeline_summary.txt
```

### Ground Truth File (Optional)

Same CSV format as Stage 1:
```csv
Filename,transcript
call_001.wav,"Patient reports chest pain and shortness of breath"
call_002.wav,"Motor vehicle accident at Main Street intersection"
```

## 📊 Output Structure

```
llm_results_YYYYMMDD_HHMMSS/
├── whisper_filtered/                    # Filtered Whisper transcripts
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

### Output File Descriptions

#### Medical Correction Results
- **corrected_transcripts/**: Enhanced transcripts with corrected medical terminology
- **local_medical_correction_summary.json**: Processing statistics and success rates

#### Emergency Page Results
- **emergency_pages/**: Structured emergency reports
- **local_emergency_page_summary.json**: Generation statistics

#### Evaluation Results
- **llm_enhanced_evaluation_results.csv**: WER, MER, WIL metrics comparison
- **error_analysis.log**: Detailed failure analysis with error categorization

## 📈 Performance Monitoring

### Processing Metrics

The pipeline tracks comprehensive performance metrics:

```json
{
  "processing_stats": {
    "total_files": 150,
    "successful_corrections": 148,
    "successful_pages": 147,
    "failed_files": 2,
    "processing_time": "00:45:23",
    "average_time_per_file": "00:00:18"
  },
  "model_performance": {
    "medical_correction": {
      "model": "BioMistral-7B",
      "quantization": "8-bit",
      "gpu_memory_used": "4.2GB",
      "success_rate": 98.7
    },
    "page_generation": {
      "model": "BioMistral-7B",
      "quantization": "8-bit",
      "gpu_memory_used": "4.2GB",
      "success_rate": 98.0
    }
  }
}
```

### Real-time Monitoring

```bash
# Monitor GPU usage during processing
watch -n 1 nvidia-smi

# Monitor processing logs
tail -f llm_results_*/error_analysis.log

# Check processing progress
grep "Processing:" llm_results_*/llm_enhanced_pipeline_summary.txt
```

## 🚨 Error Handling

### Common Error Types

#### Model Loading Errors
```
Error: CUDA out of memory
Solution: Use quantization (--load_in_8bit or --load_in_4bit)

Error: Model not found
Solution: Check internet connection, models download automatically

Error: Torch not compiled with CUDA
Solution: Install CUDA-enabled PyTorch
```

#### Processing Errors
```
Error: Empty or unreadable transcript
Cause: Input file is empty or corrupted
Solution: Check ASR results quality, file permissions

Error: Model correction failed
Cause: GPU memory insufficient, model timeout
Solution: Reduce batch size, use quantization, increase timeout
```

#### File I/O Errors
```
Error: Permission denied
Solution: Check file permissions, disk space

Error: Path not found
Solution: Verify ASR results directory path
```

### Error Analysis

```bash
# View error summary
cat llm_results_*/error_analysis.log

# Count errors by type
grep "Error:" llm_results_*/error_analysis.log | sort | uniq -c

# Find failed files
grep "FAILED FILE:" llm_results_*/error_analysis.log

# Analyze error patterns
python3 tool/analyze_llm_errors.py --error_log llm_results_*/error_analysis.log
```

## 🔍 Troubleshooting

### Performance Issues

#### Slow Processing
1. **Use quantization**: Enable 8-bit or 4-bit quantization
2. **Reduce batch size**: Set `--batch_size 1`
3. **GPU acceleration**: Ensure CUDA is available
4. **Model caching**: Models download once, then cached

#### High Memory Usage
1. **Enable 4-bit quantization**: `--load_in_4bit`
2. **Reduce batch size**: `--batch_size 1`
3. **Use CPU processing**: `--device cpu` (slower but less memory)
4. **Close other GPU applications**: Free GPU memory

### Quality Issues

#### Poor Medical Correction
1. **Try different models**: Meditron-7B for clinical text
2. **Customize prompts**: Add domain-specific instructions
3. **Check input quality**: Verify ASR transcript accuracy
4. **Use larger models**: Consider Llama-3-8B-UltraMedica

#### Inconsistent Emergency Pages
1. **Refine prompts**: Add specific format requirements
2. **Use consistent models**: Same model for both tasks
3. **Check medical context**: Ensure sufficient medical information

### Model-Specific Issues

#### BioMistral-7B
```bash
# Clear model cache if corrupted
rm -rf ~/.cache/huggingface/transformers/BioMistral*

# Verify model download
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BioMistral/BioMistral-7B')"
```

#### Quantization Issues
```bash
# Install bitsandbytes
pip install bitsandbytes>=0.41.0

# Check CUDA compatibility
python3 -c "import bitsandbytes; print(bitsandbytes.cuda_setup.common.get_cuda_version())"
```

## 🔧 Advanced Usage

### Custom Model Integration

Add custom models to the pipeline:

```bash
# Edit MODEL_PATHS in run_llm_pipeline.sh
MODEL_PATHS=(
    "custom-medical-model:/path/to/custom/model"
    "BioMistral-7B:BioMistral/BioMistral-7B"
)

# Use custom model
./run_llm_pipeline.sh \
    --medical_correction_model "custom-medical-model" \
    --model_path "/path/to/custom/model"
```

### Batch Processing Multiple Results

```bash
# Process multiple ASR result directories
for dir in pipeline_results_*/; do
    ./run_llm_pipeline.sh --asr_results_dir "$dir" --load_in_8bit
done
```

### API Integration

```python
# Python API example
import requests

def process_with_llm_api(transcript_text, model="BioMistral-7B"):
    response = requests.post("http://localhost:8000/correct", {
        "text": transcript_text,
        "model": model,
        "quantization": "8bit"
    })
    return response.json()
```

### Performance Benchmarking

```bash
# Benchmark different configurations
python3 tool/benchmark_llm_models.py \
    --models "BioMistral-7B,Meditron-7B" \
    --quantizations "none,8bit,4bit" \
    --test_data test_transcripts/
```

## 📚 Model Comparison

### Medical Correction Performance

| Model | Medical Accuracy | Speed | Memory (8-bit) | Best Use Case |
|-------|------------------|-------|----------------|---------------|
| BioMistral-7B | 95% | Fast | 4GB | General medical terms |
| Meditron-7B | 93% | Medium | 4GB | Clinical documentation |
| Llama-3-8B-UltraMedica | 97% | Slow | 4.5GB | Complex medical reasoning |
| gpt-oss-20b | 89% | Very Slow | 12GB | General language tasks |

### Emergency Page Quality

| Model | Structure Quality | Medical Relevance | Completeness | Consistency |
|-------|------------------|-------------------|--------------|-------------|
| BioMistral-7B | Excellent | Excellent | High | High |
| Meditron-7B | Good | Excellent | Medium | Medium |
| Llama-3-8B-UltraMedica | Excellent | Good | High | High |
| gpt-oss-20b | Good | Medium | Medium | Low |

## 🔗 Related Documentation

- [ASR Pipeline Guide](ASR_PIPELINE_GUIDE.md) - Stage 1 processing
- [Model Configuration Guide](MODEL_CONFIG_GUIDE.md) - Model setup details
- [Error Handling Guide](ERROR_HANDLING_GUIDE.md) - Troubleshooting
- [Performance Tuning](PERFORMANCE_TUNING.md) - Optimization tips
- [Command Reference](COMMAND_REFERENCE.md) - All parameters

## 📞 Support

For LLM pipeline issues:

1. **Check error logs**: `error_analysis.log`
2. **Verify GPU setup**: `nvidia-smi` and CUDA availability
3. **Test model loading**: Manual model download verification
4. **Monitor resources**: GPU memory and system RAM usage
5. **Review configuration**: Validate all model and quantization settings

---

**Note**: This guide covers the LLM enhancement stage. For complete EMS call analysis, start with the ASR Pipeline Guide and then proceed to this LLM processing stage.