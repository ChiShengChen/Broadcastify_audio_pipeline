# Performance Tuning Guide

A comprehensive guide for optimizing performance, memory usage, and processing speed in the EMS Call ASR and LLM-Enhanced Pipeline.

## 📋 Overview

This guide provides detailed strategies for optimizing both pipeline stages to achieve the best performance based on your hardware configuration, processing requirements, and quality targets.

## 🏗️ Performance Architecture

```
Hardware → Configuration → Processing → Optimization → Monitoring
```

### Performance Factors

1. **Hardware Resources**: CPU, GPU, RAM, Storage
2. **Model Selection**: Size, quantization, specialization
3. **Processing Configuration**: Batch size, parallel workers
4. **Data Optimization**: Audio quality, file size, preprocessing
5. **System Optimization**: Environment, caching, I/O

## 🖥️ Hardware Optimization

### System Requirements by Use Case

#### Research/Development Environment
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 16GB | 32GB | 64GB+ |
| **GPU** | GTX 1660 | RTX 3080 | RTX 4090 |
| **GPU Memory** | 6GB | 12GB | 24GB+ |
| **Storage** | HDD | SSD | NVMe SSD |

#### Production Environment
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU** | 8 cores | 16 cores | 32+ cores |
| **RAM** | 32GB | 64GB | 128GB+ |
| **GPU** | RTX 3070 | RTX 4080 | Multiple RTX 4090 |
| **GPU Memory** | 8GB | 16GB | 24GB+ per GPU |
| **Storage** | SSD | NVMe SSD | NVMe RAID |

### GPU Optimization

#### Single GPU Configuration
```bash
# Check GPU utilization
nvidia-smi -l 1

# Optimize GPU memory
export CUDA_MEMORY_FRACTION=0.9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### Multi-GPU Setup
```bash
# Configure multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Load balancing across GPUs
./run_llm_pipeline.sh \
    --device "auto" \
    --load_in_8bit \
    --batch_size 2
```

### Memory Optimization Strategies

#### RAM Optimization
```bash
# Monitor memory usage
free -h
watch -n 1 free -h

# Optimize system memory
echo 3 > /proc/sys/vm/drop_caches  # Clear cache
swapoff -a && swapon -a           # Reset swap
```

#### GPU Memory Optimization
```bash
# Monitor GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Clear GPU cache in Python
python3 -c "
import torch
torch.cuda.empty_cache()
print('GPU cache cleared')
"
```

## ⚡ Stage 1: ASR Pipeline Optimization

### Processing Speed Optimization

#### Parallel Processing Configuration
```bash
# Optimal worker configuration based on CPU cores
CPU_CORES=$(nproc)
OPTIMAL_WORKERS=$((CPU_CORES / 2))

./run_pipeline.sh \
    --max-workers $OPTIMAL_WORKERS \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv"
```

#### Audio Preprocessing Optimization
```bash
# Skip unnecessary preprocessing for speed
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv" \
    --disable-vad \
    --disable-audio-filter \
    --disable-long-audio-split
```

#### Model-Specific Optimizations

**Whisper Optimization:**
```bash
# Use smaller Whisper model for speed
export WHISPER_MODEL="medium"  # Instead of large-v3

# Enable GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Wav2Vec2 Optimization:**
```bash
# Use optimized model variant
export WAV2VEC2_MODEL="facebook/wav2vec2-base-960h"  # Faster than large variants
```

### Memory Usage Optimization

#### Long Audio Handling
```bash
# Optimize for memory-constrained systems
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv" \
    --use-long-audio-split \
    --max-segment-duration 30 \
    --max-workers 1
```

#### Batch Size Optimization
```python
# In run_all_asrs.py, optimize batch processing
def optimize_batch_size():
    """Dynamically adjust batch size based on available memory"""
    import psutil
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    
    if available_ram_gb > 16:
        return 8
    elif available_ram_gb > 8:
        return 4
    else:
        return 2
```

### I/O Optimization

#### Storage Configuration
```bash
# Use fast storage for temporary files
export TEMP_DIR="/path/to/fast/ssd"
export TMPDIR="$TEMP_DIR"

# Optimize for SSD
echo mq-deadline > /sys/block/sda/queue/scheduler
```

#### File System Optimization
```bash
# Mount with performance options
mount -o noatime,nodiratime /dev/sda1 /path/to/workspace

# Increase file descriptor limits
ulimit -n 65536
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf
```

## 🧠 Stage 2: LLM Pipeline Optimization

### Model Selection Optimization

#### Performance vs Quality Trade-offs

| Model | Size | Speed | Quality | Memory (8-bit) | Best Use Case |
|-------|------|-------|---------|----------------|---------------|
| **BioMistral-7B** | 7B | Fast | High | 4GB | Balanced performance |
| **Meditron-7B** | 7B | Medium | High | 4GB | Quality-focused |
| **Llama-3-8B-UltraMedica** | 8B | Slow | Highest | 4.5GB | Maximum accuracy |
| **gpt-oss-20b** | 20B | Very Slow | Variable | 12GB | Complex reasoning |

#### Model Loading Optimization
```python
# Optimize model loading in llm_local_models.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_optimized_model(model_name, quantization_config):
    """Load model with optimizations"""
    
    # Enable memory mapping for faster loading
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=True
    )
    
    # Enable compilation for PyTorch 2.0+
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    return model
```

### Quantization Optimization

#### Optimal Quantization Selection

| Hardware | Quantization | Memory Usage | Speed | Quality | Recommendation |
|----------|-------------|--------------|-------|---------|----------------|
| **RTX 4090 (24GB)** | None/FP16 | 100% | 1x | 100% | Maximum quality |
| **RTX 4080 (16GB)** | 8-bit | 25% | 1.5-2x | 95% | **Recommended** |
| **RTX 4070 (12GB)** | 8-bit | 25% | 1.5-2x | 95% | **Recommended** |
| **RTX 3080 (10GB)** | 8-bit | 25% | 1.5-2x | 95% | **Recommended** |
| **RTX 3060 (8GB)** | 4-bit | 12% | 2-4x | 85% | Memory-constrained |

#### Advanced Quantization Configuration
```python
# Custom quantization setup
from transformers import BitsAndBytesConfig

# Optimized 8-bit configuration
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    llm_int8_enable_fp32_cpu_offload=True
)

# Optimized 4-bit configuration
quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### Batch Processing Optimization

#### Dynamic Batch Sizing
```python
def calculate_optimal_batch_size(model_size_gb, available_gpu_memory_gb, quantization='8bit'):
    """Calculate optimal batch size based on available resources"""
    
    # Memory overhead factors
    overhead_factors = {
        'none': 2.5,
        '8bit': 1.8,
        '4bit': 1.5
    }
    
    overhead = overhead_factors.get(quantization, 2.0)
    available_memory = available_gpu_memory_gb - (model_size_gb * overhead)
    
    # Estimate memory per batch item (approximate)
    memory_per_item = 0.5  # GB
    
    optimal_batch_size = max(1, int(available_memory / memory_per_item))
    return min(optimal_batch_size, 4)  # Cap at 4 for stability
```

#### Parallel Processing Strategies
```bash
# Process different tasks in parallel
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --batch_size 2 \
    --load_in_8bit &

# Monitor parallel processing
jobs
wait  # Wait for all background jobs to complete
```

## 📊 Performance Monitoring and Benchmarking

### Real-time Performance Monitoring

#### System Resource Monitoring
```bash
#!/bin/bash
# monitor_performance.sh

# Create monitoring script
cat > monitor_pipeline.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== EMS Pipeline Performance Monitor ==="
    echo "Time: $(date)"
    echo ""
    
    echo "=== CPU Usage ==="
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
    
    echo "=== Memory Usage ==="
    free -h | grep -E "(Mem|Swap)"
    
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    
    echo "=== Disk I/O ==="
    iostat -x 1 1 | tail -n +4
    
    sleep 5
done
EOF

chmod +x monitor_pipeline.sh
./monitor_pipeline.sh
```

#### Performance Logging
```python
# performance_logger.py
import time
import psutil
import torch
import json
from datetime import datetime

class PerformanceLogger:
    def __init__(self, log_file="performance.log"):
        self.log_file = log_file
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.log_system_info()
        
    def log_system_info(self):
        """Log initial system information"""
        info = {
            'timestamp': datetime.now().isoformat(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['gpu_name'] = torch.cuda.get_device_name(0)
        
        self.write_log(info)
    
    def log_processing_stats(self, stage, files_processed, success_rate):
        """Log processing statistics"""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'files_processed': files_processed,
            'success_rate': success_rate,
            'elapsed_time': elapsed,
            'files_per_second': files_processed / elapsed if elapsed > 0 else 0,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            stats['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_memory_percent'] = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
        
        self.write_log(stats)
    
    def write_log(self, data):
        """Write log entry"""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
```

### Benchmarking Tools

#### Performance Benchmarking Script
```bash
#!/bin/bash
# benchmark_pipeline.sh

echo "=== EMS Pipeline Performance Benchmark ==="

# Test configurations
CONFIGURATIONS=(
    "basic:--load_in_8bit"
    "optimized:--load_in_8bit --batch_size 2"
    "memory_optimized:--load_in_4bit --batch_size 1"
    "quality:--batch_size 1"
)

RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).csv"
echo "Configuration,Files,Success_Rate,Total_Time,Files_Per_Second,Peak_Memory_GB,GPU_Memory_GB" > $RESULTS_FILE

for config in "${CONFIGURATIONS[@]}"; do
    IFS=':' read -r name params <<< "$config"
    echo "Testing configuration: $name"
    
    start_time=$(date +%s)
    
    # Run pipeline with monitoring
    ./run_llm_pipeline.sh \
        --asr_results_dir "/path/to/test/data" \
        $params \
        > "benchmark_${name}.log" 2>&1
    
    end_time=$(date +%s)
    total_time=$((end_time - start_time))
    
    # Extract results from log
    files=$(grep "Total files:" "benchmark_${name}.log" | awk '{print $3}')
    success_rate=$(grep "Success rate:" "benchmark_${name}.log" | awk '{print $3}' | tr -d '%')
    
    # Calculate metrics
    files_per_second=$(echo "scale=2; $files / $total_time" | bc)
    
    echo "$name,$files,$success_rate,$total_time,$files_per_second,0,0" >> $RESULTS_FILE
done

echo "Benchmark completed. Results saved to: $RESULTS_FILE"
```

## 🔧 Advanced Optimization Techniques

### Model Optimization

#### Model Pruning and Distillation
```python
# model_optimization.py
import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM

def prune_model(model, pruning_amount=0.2):
    """Apply structured pruning to reduce model size"""
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
    
    return model

def optimize_model_for_inference(model):
    """Optimize model for faster inference"""
    
    # Enable eval mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Use half precision
    model = model.half()
    
    # Enable optimized attention (if available)
    if hasattr(model.config, 'use_flash_attention'):
        model.config.use_flash_attention = True
    
    return model
```

#### Custom Model Loading
```python
def load_optimized_medical_model(model_name, optimization_level="balanced"):
    """Load model with different optimization levels"""
    
    optimizations = {
        "speed": {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "quantization": "8bit"
        },
        "balanced": {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "quantization": "8bit"
        },
        "quality": {
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": False,
            "device_map": "auto",
            "quantization": None
        }
    }
    
    config = optimizations[optimization_level]
    
    # Load with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=config["torch_dtype"],
        low_cpu_mem_usage=config["low_cpu_mem_usage"],
        device_map=config["device_map"]
    )
    
    return model
```

### Caching and Preprocessing Optimization

#### Model Caching Strategy
```bash
# Pre-download and cache models
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

models = ['BioMistral/BioMistral-7B', 'epfl-llm/meditron-7b']
for model_name in models:
    print(f'Caching {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f'✓ {model_name} cached')
"
```

#### Preprocessing Pipeline Optimization
```python
def optimize_preprocessing_pipeline():
    """Optimize preprocessing for better performance"""
    
    # Use faster audio loading
    import librosa
    
    def fast_audio_load(file_path, target_sr=16000):
        # Use librosa's faster loading with offset and duration
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return y, sr
    
    # Batch preprocessing
    def batch_preprocess_audio(file_paths, target_sr=16000):
        results = []
        for file_path in file_paths:
            y, sr = fast_audio_load(file_path, target_sr)
            results.append((y, sr))
        return results
    
    return batch_preprocess_audio
```

## 🎯 Configuration Recommendations

### By Use Case

#### High-Throughput Processing
```bash
# Optimized for processing many files quickly
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv" \
    --max-workers 8 \
    --disable-vad \
    --disable-audio-filter

./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit \
    --batch_size 4
```

#### High-Quality Processing
```bash
# Optimized for maximum accuracy
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv" \
    --use-vad \
    --enable-audio-filter \
    --filter-mode moderate \
    --preprocess-ground-truth

./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "Llama-3-8B-UltraMedica" \
    --page_generation_model "BioMistral-7B" \
    --batch_size 1
```

#### Resource-Constrained Processing
```bash
# Optimized for limited hardware
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv" \
    --use-long-audio-split \
    --max-segment-duration 30 \
    --max-workers 1

./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_4bit \
    --batch_size 1 \
    --device "cpu"
```

### By Hardware Configuration

#### High-End Workstation (RTX 4090, 64GB RAM)
```bash
# Maximum performance configuration
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "Llama-3-8B-UltraMedica" \
    --load_in_8bit \
    --batch_size 4 \
    --device "cuda"
```

#### Mid-Range System (RTX 3080, 32GB RAM)
```bash
# Balanced configuration
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_8bit \
    --batch_size 2 \
    --device "cuda"
```

#### Budget System (GTX 1660, 16GB RAM)
```bash
# Memory-optimized configuration
./run_llm_pipeline.sh \
    --asr_results_dir "/path/to/asr_results" \
    --medical_correction_model "BioMistral-7B" \
    --page_generation_model "BioMistral-7B" \
    --load_in_4bit \
    --batch_size 1 \
    --device "cuda"
```

## 🔍 Performance Troubleshooting

### Common Performance Issues

#### Slow Processing
**Symptoms**: Very slow processing, low GPU utilization
```bash
# Diagnose and fix
# 1. Check GPU utilization
nvidia-smi

# 2. Increase batch size if memory allows
./run_llm_pipeline.sh --batch_size 2

# 3. Enable quantization
./run_llm_pipeline.sh --load_in_8bit

# 4. Check for CPU bottlenecks
htop
```

#### High Memory Usage
**Symptoms**: Out of memory errors, system slowdown
```bash
# Diagnose and fix
# 1. Monitor memory usage
free -h
nvidia-smi

# 2. Enable more aggressive quantization
./run_llm_pipeline.sh --load_in_4bit

# 3. Reduce batch size
./run_llm_pipeline.sh --batch_size 1

# 4. Enable long audio splitting
./run_pipeline.sh --use-long-audio-split --max-segment-duration 30
```

#### Poor Quality Results
**Symptoms**: High error rates, poor medical terminology
```bash
# Diagnose and fix
# 1. Disable aggressive optimizations
./run_llm_pipeline.sh  # Remove quantization flags

# 2. Use higher quality models
./run_llm_pipeline.sh --medical_correction_model "Llama-3-8B-UltraMedica"

# 3. Enable preprocessing
./run_pipeline.sh --use-vad --enable-audio-filter
```

## 🔗 Related Documentation

- [ASR Pipeline Guide](ASR_PIPELINE_GUIDE.md) - ASR processing details
- [LLM Pipeline Guide](LLM_PIPELINE_GUIDE.md) - LLM enhancement guide
- [Model Configuration Guide](MODEL_CONFIG_GUIDE.md) - Model setup optimization
- [Error Handling Guide](ERROR_HANDLING_GUIDE.md) - Performance troubleshooting

## 📞 Support

For performance optimization assistance:

1. **Profile your workload**: Use provided monitoring tools
2. **Test different configurations**: Start with recommended settings
3. **Monitor resource usage**: Watch CPU, GPU, and memory utilization
4. **Benchmark regularly**: Track performance changes over time
5. **Document optimal settings**: Save working configurations

---

**Note**: Performance optimization is highly dependent on your specific hardware, data, and quality requirements. Start with the recommended configurations and adjust based on your monitoring results.