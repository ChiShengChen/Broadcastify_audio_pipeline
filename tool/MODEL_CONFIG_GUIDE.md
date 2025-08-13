# Model Configuration Guide

A comprehensive guide for configuring, optimizing, and managing ASR and LLM models in the EMS Call Analysis Pipeline.

## 📋 Overview

This guide covers the setup, configuration, and optimization of both ASR models (Stage 1) and LLM models (Stage 2) used in the emergency medical service call analysis pipeline.

## 🤖 ASR Models Configuration (Stage 1)

### Supported ASR Models

| Model | Framework | Size | Memory | Strengths | Best Use Case |
|-------|-----------|------|--------|-----------|---------------|
| **Whisper Large-v3** ⭐ | OpenAI | 1.55GB | ~4GB | Multilingual, robust | General purpose, recommended |
| **Wav2Vec2** | HuggingFace | 360MB | ~2GB | English optimized | Clean audio, fast processing |
| **Canary-1B** | NVIDIA NeMo | 1.2GB | ~8GB | Enterprise features | Professional deployment |
| **Parakeet CTC-0.6B** | NVIDIA NeMo | 600MB | ~3GB | Low latency | Real-time applications |

### ASR Model Installation

#### Whisper Models
```bash
# Install Whisper
pip install openai-whisper

# Download specific model (optional, auto-downloaded on first use)
python3 -c "import whisper; whisper.load_model('large-v3')"

# Available sizes: tiny, base, small, medium, large, large-v2, large-v3
```

#### HuggingFace Transformers Models
```bash
# Install transformers
pip install transformers torch torchaudio

# Pre-download Wav2Vec2 model (optional)
python3 -c "
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')
print('Wav2Vec2 model ready')
"
```

#### NVIDIA NeMo Models
```bash
# Install NeMo toolkit
pip install nemo_toolkit[asr]

# Pre-download NeMo models (optional)
python3 -c "
import nemo.collections.asr as nemo_asr
canary = nemo_asr.models.EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
parakeet = nemo_asr.models.EncDecCTCModel.from_pretrained('nvidia/parakeet-ctc-0.6b')
print('NeMo models ready')
"
```

### ASR Model Configuration

#### Model Selection in `run_all_asrs.py`

```python
MODELS = {
    'large-v3': {
        'path': 'large-v3',
        'framework': 'whisper',
        'enabled': True,
        'memory_requirement': '4GB',
        'recommended': True
    },
    'wav2vec-xls-r': {
        'path': 'facebook/wav2vec2-base-960h',
        'framework': 'transformers',
        'enabled': True,
        'memory_requirement': '2GB',
        'best_for': 'clean_audio'
    },
    'canary-1b': {
        'path': 'nvidia/canary-1b',
        'framework': 'nemo',
        'enabled': True,
        'memory_requirement': '8GB',
        'features': ['multilingual', 'punctuation']
    },
    'parakeet-tdt-0.6b-v2': {
        'path': 'nvidia/parakeet-ctc-0.6b',
        'framework': 'nemo',
        'enabled': True,
        'memory_requirement': '3GB',
        'best_for': 'streaming'
    }
}
```

#### Custom ASR Model Integration

```python
# Add custom model to run_all_asrs.py
MODELS['custom_model'] = {
    'path': 'path/to/custom/model',
    'framework': 'transformers',  # or 'whisper', 'nemo'
    'enabled': True,
    'memory_requirement': '4GB',
    'preprocessing': custom_preprocess_function,
    'postprocessing': custom_postprocess_function,
    'config': {
        'sample_rate': 16000,
        'language': 'en',
        'task': 'transcribe'
    }
}
```

## 🧠 LLM Models Configuration (Stage 2)

### Supported LLM Models

| Model | Size | Specialization | HuggingFace ID | Memory (FP16) | Memory (8-bit) | Memory (4-bit) |
|-------|------|----------------|----------------|---------------|----------------|----------------|
| **BioMistral-7B** ⭐ | 7B | Medical domain | `BioMistral/BioMistral-7B` | ~14GB | ~4GB | ~2GB |
| **Meditron-7B** | 7B | Medical literature | `epfl-llm/meditron-7b` | ~14GB | ~4GB | ~2GB |
| **Llama-3-8B-UltraMedica** | 8B | Medical fine-tuned | Custom path | ~16GB | ~4.5GB | ~2.5GB |
| **gpt-oss-20b** | 20B | General purpose | Custom path | ~40GB | ~12GB | ~6GB |

### LLM Model Installation

#### BioMistral-7B (Recommended)
```bash
# Pre-download BioMistral model
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('BioMistral/BioMistral-7B')
model = AutoModelForCausalLM.from_pretrained('BioMistral/BioMistral-7B')
print('BioMistral-7B ready')
"
```

#### Meditron-7B
```bash
# Pre-download Meditron model
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('epfl-llm/meditron-7b')
model = AutoModelForCausalLM.from_pretrained('epfl-llm/meditron-7b')
print('Meditron-7B ready')
"
```

#### Custom Local Models
```bash
# For custom models, ensure proper directory structure
/path/to/custom/model/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

### LLM Model Configuration

#### Model Path Configuration in `run_llm_pipeline.sh`

```bash
# Default model paths
MODEL_PATHS=(
    "BioMistral-7B:BioMistral/BioMistral-7B"
    "Meditron-7B:epfl-llm/meditron-7b"
    "Llama-3-8B-UltraMedica:/path/to/llama-3-8b-ultramedica"
    "gpt-oss-20b:/path/to/gpt-oss-20b"
)

# Custom model addition
MODEL_PATHS+=(
    "custom-medical-model:/path/to/custom/medical/model"
)
```

#### Model Selection and Quantization

```bash
# Basic configuration
MEDICAL_CORRECTION_MODEL="BioMistral-7B"
PAGE_GENERATION_MODEL="BioMistral-7B"

# Quantization settings
LOAD_IN_8BIT=false  # Set to true for 8-bit quantization
LOAD_IN_4BIT=false  # Set to true for 4-bit quantization

# Device configuration
DEVICE="auto"  # auto, cpu, cuda, cuda:0, cuda:1, etc.
```

## ⚙️ Quantization Configuration

### Understanding Quantization

Quantization reduces model precision to save memory and increase speed:

| Type | Precision | Memory Reduction | Speed Increase | Quality Loss | Use Case |
|------|-----------|------------------|----------------|--------------|----------|
| **FP16** | 16-bit float | 50% vs FP32 | 1.5x | Minimal | High-end GPUs |
| **8-bit** | INT8 | 75% vs FP16 | 1.5-2x | Low | Most GPUs |
| **4-bit** | INT4 | 87.5% vs FP16 | 2-4x | Moderate | Limited GPUs |

### Quantization Setup

#### Prerequisites
```bash
# Install quantization dependencies
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.20.0

# Verify installation
python3 -c "import bitsandbytes; print('BitsAndBytes available')"
```

#### 8-bit Quantization Configuration
```bash
# Enable 8-bit quantization (recommended)
./run_llm_pipeline.sh \
    --load_in_8bit \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B"
```

#### 4-bit Quantization Configuration
```bash
# Enable 4-bit quantization (maximum memory savings)
./run_llm_pipeline.sh \
    --load_in_4bit \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B"
```

### Advanced Quantization Settings

#### Custom Quantization Configuration in Python
```python
# In llm_local_models.py, modify model loading:
from transformers import BitsAndBytesConfig

# 8-bit configuration
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# 4-bit configuration
quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config_8bit,
    device_map="auto"
)
```

## 🚀 Performance Optimization

### Hardware-Based Optimization

#### GPU Configuration
```bash
# Check GPU capabilities
nvidia-smi
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

#### Memory Optimization Strategies

| GPU Memory | Recommended Configuration | Models |
|------------|--------------------------|--------|
| **24GB+** | No quantization | All models, highest quality |
| **12-16GB** | 8-bit quantization | 7B-8B models |
| **8-12GB** | 8-bit quantization | 7B models only |
| **4-8GB** | 4-bit quantization | 7B models only |
| **<4GB** | CPU processing | All models (slower) |

### Model-Specific Optimizations

#### BioMistral-7B Optimization
```bash
# Optimal configuration for BioMistral-7B
./run_llm_pipeline.sh \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit \
    --device "cuda" \
    --batch_size 1
```

#### Multi-GPU Configuration
```bash
# For multi-GPU systems
export CUDA_VISIBLE_DEVICES=0,1
./run_llm_pipeline.sh \
    --device "auto" \
    --load_in_8bit
```

#### CPU-Only Configuration
```bash
# For systems without GPU
./run_llm_pipeline.sh \
    --device "cpu" \
    --batch_size 1 \
    --medical_correction_model "BioMistral-7B"
```

## 🔧 Advanced Model Configuration

### Custom Model Integration

#### Adding New LLM Models

1. **Update model configuration in `run_llm_pipeline.sh`:**
```bash
# Add to AVAILABLE_MODELS
AVAILABLE_MODELS=("gpt-oss-20b" "BioMistral-7B" "Meditron-7B" "Llama-3-8B-UltraMedica" "CustomMedicalModel")

# Add to MODEL_PATHS
MODEL_PATHS+=(
    "CustomMedicalModel:/path/to/custom/model"
)
```

2. **Implement model loading in `llm_local_models.py`:**
```python
def load_custom_model(model_path, quantization_config=None):
    """Load custom medical model"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model, tokenizer
```

### Model Fine-tuning for Medical Domain

#### Preparing Medical Training Data
```python
# Example medical training data format
medical_training_data = [
    {
        "input": "Patient presents with chest pain and shortness of breath",
        "output": "Patient presents with chest pain and dyspnea"
    },
    {
        "input": "Give aspirin 325mg by mouth",
        "output": "Administer aspirin 325 mg PO"
    }
]
```

#### Fine-tuning Script Example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

def fine_tune_medical_model(base_model, training_data, output_dir):
    """Fine-tune model for medical terminology"""
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
        dataloader_pin_memory=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        tokenizer=tokenizer
    )
    
    # Fine-tune
    trainer.train()
    trainer.save_model()
```

## 📊 Model Performance Monitoring

### Performance Metrics

#### ASR Model Metrics
```bash
# Analyze ASR model performance
python3 tool/analyze_model_files_enhanced.py \
    --transcript_dir /path/to/transcripts \
    --ground_truth_file /path/to/ground_truth.csv

# Compare model performance
python3 tool/compare_asr_models.py \
    --results_dir /path/to/pipeline_results
```

#### LLM Model Metrics
```python
# Monitor LLM processing metrics
import psutil
import time
import torch

def monitor_model_performance(model_name, processing_function):
    """Monitor model performance during processing"""
    
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    if torch.cuda.is_available():
        start_gpu_memory = torch.cuda.memory_allocated()
    
    # Run processing
    result = processing_function()
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    if torch.cuda.is_available():
        end_gpu_memory = torch.cuda.memory_allocated()
        gpu_memory_used = (end_gpu_memory - start_gpu_memory) / 1e9
    else:
        gpu_memory_used = 0
    
    metrics = {
        'model': model_name,
        'processing_time': end_time - start_time,
        'ram_used': (end_memory - start_memory) / 1e9,
        'gpu_memory_used': gpu_memory_used,
        'success': result['success'] if isinstance(result, dict) else True
    }
    
    return metrics
```

### Benchmarking Tools

```bash
# Benchmark different model configurations
python3 tool/benchmark_models.py \
    --models "BioMistral-7B,Meditron-7B" \
    --quantizations "none,8bit,4bit" \
    --test_data /path/to/test_data

# Performance comparison report
python3 tool/generate_performance_report.py \
    --benchmark_results /path/to/benchmark_results.json
```

## 🛠️ Model Management

### Model Caching and Storage

#### HuggingFace Model Cache
```bash
# View cached models
ls -la ~/.cache/huggingface/transformers/

# Clear specific model cache
rm -rf ~/.cache/huggingface/transformers/models--BioMistral--BioMistral-7B

# Set custom cache directory
export HF_HOME=/path/to/custom/cache
export TRANSFORMERS_CACHE=/path/to/custom/cache/transformers
```

#### Local Model Storage
```bash
# Organize local models
mkdir -p /path/to/models/{asr,llm}
mkdir -p /path/to/models/llm/{BioMistral-7B,Meditron-7B}

# Download and store models locally
python3 tool/download_models.py \
    --models "BioMistral/BioMistral-7B,epfl-llm/meditron-7b" \
    --output_dir /path/to/models/llm
```

### Model Version Management

```bash
# Track model versions
python3 tool/track_model_versions.py \
    --models_dir /path/to/models \
    --output_file model_versions.json

# Update models
python3 tool/update_models.py \
    --models "BioMistral-7B" \
    --check_updates
```

## 🔍 Troubleshooting Model Issues

### Common Model Problems

#### Model Loading Issues
```bash
# Debug model loading
python3 -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tokenizer = AutoTokenizer.from_pretrained('BioMistral/BioMistral-7B')
    print('✓ Tokenizer loaded successfully')
    
    model = AutoModelForCausalLM.from_pretrained('BioMistral/BioMistral-7B')
    print('✓ Model loaded successfully')
    print(f'Model device: {model.device}')
    print(f'Model dtype: {model.dtype}')
    
except Exception as e:
    print(f'✗ Error loading model: {e}')
"
```

#### Memory Issues
```bash
# Check memory usage
python3 -c "
import psutil
import torch

print(f'Total RAM: {psutil.virtual_memory().total / 1e9:.1f}GB')
print(f'Available RAM: {psutil.virtual_memory().available / 1e9:.1f}GB')

if torch.cuda.is_available():
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    print(f'GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.1f}GB')
"
```

#### Quantization Issues
```bash
# Test quantization setup
python3 -c "
import bitsandbytes as bnb
import torch

try:
    # Test 8-bit quantization
    x = torch.randn(10, 10).cuda()
    x_quantized = bnb.nn.Int8Params(x)
    print('✓ 8-bit quantization working')
    
    # Test 4-bit quantization
    from bitsandbytes.nn import Linear4bit
    layer = Linear4bit(10, 10)
    print('✓ 4-bit quantization working')
    
except Exception as e:
    print(f'✗ Quantization error: {e}')
"
```

## 🔗 Related Documentation

- [ASR Pipeline Guide](ASR_PIPELINE_GUIDE.md) - ASR processing details
- [LLM Pipeline Guide](LLM_PIPELINE_GUIDE.md) - LLM enhancement guide
- [Error Handling Guide](ERROR_HANDLING_GUIDE.md) - Troubleshooting
- [Performance Tuning](PERFORMANCE_TUNING.md) - Optimization strategies

## 📞 Support

For model configuration issues:

1. **Verify hardware requirements**: Check GPU memory and CUDA version
2. **Test model loading**: Use provided diagnostic scripts
3. **Check dependencies**: Ensure all required packages are installed
4. **Monitor resources**: Watch memory usage during model loading
5. **Review error logs**: Check detailed error messages in logs

---

**Note**: This guide covers model configuration for both pipeline stages. For specific usage instructions, refer to the respective pipeline guides.