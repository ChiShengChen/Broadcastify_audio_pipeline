# ASR Pipeline Guide

A comprehensive guide to the Automatic Speech Recognition (ASR) pipeline for emergency medical service call analysis.

## 📋 Overview

The ASR Pipeline (`run_pipeline.sh`) is the first stage of the EMS call analysis system. It processes audio files through multiple ASR models to generate transcripts for further LLM enhancement.

## 🏗️ Pipeline Architecture

```
Audio Input → Preprocessing → VAD → Long Audio Splitting → ASR Models → Evaluation → Output
```

### Processing Stages

1. **Audio Preprocessing** (Optional)
   - Upsampling to 16kHz
   - Audio filtering and enhancement
   - Format standardization

2. **Voice Activity Detection (VAD)** (Optional)
   - Speech segment extraction
   - Noise reduction
   - Optimized for medical conversations

3. **Long Audio Splitting** (Optional)
   - Prevents OOM errors
   - Maintains context boundaries
   - Configurable segment duration

4. **ASR Transcription**
   - Multiple model processing
   - Parallel execution
   - Error handling and recovery

5. **Transcript Merging**
   - Combines segmented results
   - Maintains temporal order
   - Quality validation

6. **Evaluation**
   - WER, MER, WIL metrics
   - Model comparison
   - Performance analysis

## 🤖 Supported ASR Models

### Primary Models

| Model | Framework | Strengths | Use Case |
|-------|-----------|-----------|----------|
| **Whisper Large-v3** | OpenAI Whisper | Multilingual, robust | General purpose, recommended |
| **Wav2Vec2** | HuggingFace | English optimization | Clean audio, fast processing |
| **Canary-1B** | NVIDIA NeMo | Enterprise features | Professional deployment |
| **Parakeet CTC-0.6B** | NVIDIA NeMo | Low latency | Real-time applications |

### Model Configuration

Models are automatically configured in `run_all_asrs.py`:

```python
MODELS = {
    'large-v3': {
        'path': 'large-v3',
        'framework': 'whisper',
        'memory_requirement': '4GB',
        'recommended': True
    },
    'wav2vec-xls-r': {
        'path': 'facebook/wav2vec2-base-960h',
        'framework': 'transformers',
        'memory_requirement': '2GB',
        'best_for': 'clean_audio'
    },
    'canary-1b': {
        'path': 'nvidia/canary-1b',
        'framework': 'nemo',
        'memory_requirement': '8GB',
        'features': ['punctuation', 'capitalization']
    },
    'parakeet-tdt-0.6b-v2': {
        'path': 'nvidia/parakeet-ctc-0.6b',
        'framework': 'nemo',
        'memory_requirement': '3GB',
        'best_for': 'streaming'
    }
}
```

## ⚙️ Configuration

### Basic Configuration

Edit `run_pipeline.sh` (lines 17-38):

```bash
# Required: Audio input directory
AUDIO_DIR=("/path/to/your/audio/files")

# Required: Ground truth file
GROUND_TRUTH_FILE="/path/to/ground_truth.csv"

# Optional: Enable VAD preprocessing
USE_VAD=false

# Optional: Output directory (default: timestamped)
OUTPUT_DIR=""

# Optional: Processing options
USE_LONG_AUDIO_SPLIT=true
MAX_SEGMENT_DURATION=120  # seconds
USE_ENHANCED_PREPROCESSOR=false
```

### Advanced Configuration

```bash
# VAD Configuration
VAD_THRESHOLD=0.5
VAD_MIN_SPEECH_DURATION_MS=250
VAD_MAX_SPEECH_DURATION_S=30

# Audio Preprocessing
ENABLE_AUDIO_FILTER=true
FILTER_MODE="moderate"  # conservative, moderate, aggressive

# Ground Truth Preprocessing
PREPROCESS_GROUND_TRUTH=true
ENHANCED_PREPROCESSOR_MODE="moderate"

# Performance Options
PARALLEL_PROCESSING=true
MAX_WORKERS=4
MEMORY_LIMIT="8GB"
```

## 🚀 Usage Examples

### Basic Usage

```bash
# Minimal configuration
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv"
```

### With VAD Processing

```bash
# Enable VAD for better speech extraction
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv" \
    --use-vad \
    --vad-threshold 0.5
```

### Complete Processing Pipeline

```bash
# Full feature processing
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --output_dir "/path/to/results" \
    --ground_truth "/path/to/ground_truth.csv" \
    --use-vad \
    --use-long-audio-split \
    --max-segment-duration 120 \
    --preprocess-ground-truth \
    --use-enhanced-preprocessor \
    --enhanced-preprocessor-mode aggressive \
    --enable-audio-filter \
    --filter-mode moderate
```

### Memory-Optimized Processing

```bash
# For systems with limited resources
./run_pipeline.sh \
    --input_dir "/path/to/audio" \
    --ground_truth "/path/to/ground_truth.csv" \
    --use-long-audio-split \
    --max-segment-duration 60 \
    --max-workers 2 \
    --memory-limit 4GB
```

## 📁 Input Requirements

### Audio Files

- **Format**: WAV files (preferred)
- **Sample Rate**: 16kHz (auto-converted if different)
- **Channels**: Mono (auto-converted if stereo)
- **Duration**: No strict limit (long audio auto-split)
- **Quality**: Clear speech, minimal background noise

### Ground Truth File

CSV format with required columns:

```csv
Filename,transcript
call_001.wav,"Patient reports chest pain and shortness of breath"
call_002.wav,"Motor vehicle accident at Main Street intersection"
call_003.wav,"Elderly patient fell at home, possible hip fracture"
```

**Requirements**:
- **Filename**: Exact match with audio files (case-sensitive)
- **transcript**: Human-annotated transcription
- **Encoding**: UTF-8
- **Format**: Standard CSV with comma delimiter

## 📊 Output Structure

```
pipeline_results_YYYYMMDD_HHMMSS/
├── preprocessed_audio/              # Audio preprocessing results
│   ├── upsampled/                   # 16kHz converted audio
│   └── metadata/                    # Processing metadata
├── filtered_audio/                  # Audio filtering results
│   ├── filtered/                    # Noise-reduced audio
│   └── filter_analysis.json         # Filter performance metrics
├── vad_segments/                    # VAD extracted segments
│   ├── segments/                    # Speech-only segments
│   ├── vad_analysis.json           # VAD performance metrics
│   └── segment_metadata.csv        # Segment timing information
├── long_audio_segments/             # Long audio split results
│   ├── segments/                    # Split audio segments
│   └── segment_mapping.json        # Original-to-segment mapping
├── asr_transcripts/                 # Raw ASR results
│   ├── large-v3_call_001.txt       # Whisper results
│   ├── large-v3_call_002.txt
│   ├── wav2vec2_call_001.txt        # Wav2Vec2 results
│   ├── parakeet_call_001.txt        # Parakeet results
│   └── canary-1b_call_001.txt       # Canary results
├── merged_transcripts/              # Merged segmented transcripts
│   ├── large-v3_merged/
│   ├── wav2vec2_merged/
│   └── merge_logs.json
├── asr_evaluation_results.csv       # Performance metrics
├── model_file_analysis.txt          # Model processing analysis
├── error_analysis.log               # Error tracking
└── pipeline_summary.txt             # Complete summary
```

## 📈 Performance Metrics

### Evaluation Metrics

The pipeline calculates multiple metrics for each ASR model:

- **WER (Word Error Rate)**: Percentage of word-level errors
- **MER (Match Error Rate)**: Alignment-based error measurement
- **WIL (Word Information Lost)**: Information preservation metric
- **CER (Character Error Rate)**: Character-level accuracy
- **Processing Time**: Per-file and total processing duration
- **Memory Usage**: Peak and average memory consumption

### Results Analysis

```bash
# View evaluation results
cat pipeline_results_*/asr_evaluation_results.csv

# Analyze model performance
python3 tool/analyze_model_files_enhanced.py \
    --transcript_dir pipeline_results_*/asr_transcripts \
    --ground_truth_file ground_truth.csv

# Check processing statistics
cat pipeline_results_*/model_file_analysis.txt
```

## 🔧 Preprocessing Options

### Voice Activity Detection (VAD)

VAD improves ASR accuracy by extracting speech-only segments:

```bash
# Enable VAD with custom settings
./run_pipeline.sh \
    --use-vad \
    --vad-threshold 0.5 \
    --vad-min-speech 0.25 \
    --vad-max-speech 30
```

**Benefits**:
- Removes silence and noise
- Focuses on speech content
- Reduces processing time
- Improves accuracy

**When to Use**:
- Noisy audio environments
- Long recordings with silence
- Multiple speakers
- Background noise present

### Long Audio Splitting

Prevents memory issues with long recordings:

```bash
# Enable automatic splitting
./run_pipeline.sh \
    --use-long-audio-split \
    --max-segment-duration 120
```

**Features**:
- Intelligent boundary detection
- Context preservation
- Automatic transcript merging
- Memory optimization

### Audio Filtering

Enhances audio quality before ASR processing:

```bash
# Enable audio filtering
./run_pipeline.sh \
    --enable-audio-filter \
    --filter-mode moderate
```

**Filter Modes**:
- **Conservative**: Minimal filtering, preserves original audio
- **Moderate**: Balanced noise reduction and quality preservation
- **Aggressive**: Maximum noise reduction, may affect speech quality

### Ground Truth Preprocessing

Normalizes human transcripts for better evaluation:

```bash
# Enable ground truth preprocessing
./run_pipeline.sh \
    --preprocess-ground-truth \
    --enhanced-preprocessor-mode moderate
```

**Preprocessing Steps**:
1. Text normalization (case, punctuation)
2. Number standardization
3. Medical abbreviation expansion
4. Whitespace normalization
5. Special character handling

## 🚨 Error Handling

### Common Issues and Solutions

#### Audio Format Issues
```bash
# Error: Unsupported audio format
# Solution: Convert to WAV format
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

#### Memory Issues
```bash
# Error: CUDA out of memory
# Solution: Enable long audio splitting
./run_pipeline.sh --use-long-audio-split --max-segment-duration 60
```

#### VAD Processing Failures
```bash
# Error: VAD processing failed
# Solution: Adjust VAD parameters
./run_pipeline.sh --vad-threshold 0.3 --vad-min-speech 0.2
```

#### Missing Dependencies
```bash
# Error: Module not found
# Solution: Install requirements
pip install -r requirements.txt

# For NeMo models
pip install nemo_toolkit[asr]
```

### Error Log Analysis

```bash
# Check error log
cat pipeline_results_*/error_analysis.log

# Count errors by type
grep "ERROR:" pipeline_results_*/error_analysis.log | sort | uniq -c

# Find failed files
grep "FAILED:" pipeline_results_*/error_analysis.log
```

## 🔍 Troubleshooting

### Performance Issues

#### Slow Processing
1. **Enable parallel processing**: Increase `MAX_WORKERS`
2. **Use GPU acceleration**: Install CUDA-enabled PyTorch
3. **Reduce audio quality**: Lower sample rate if acceptable
4. **Skip preprocessing**: Disable VAD/filtering for speed

#### High Memory Usage
1. **Enable long audio splitting**: Reduce `MAX_SEGMENT_DURATION`
2. **Limit parallel workers**: Reduce `MAX_WORKERS`
3. **Use CPU-only models**: Disable GPU processing
4. **Process smaller batches**: Split input into smaller sets

### Quality Issues

#### Poor ASR Accuracy
1. **Enable VAD**: Remove silence and noise
2. **Use audio filtering**: Improve signal quality
3. **Check audio quality**: Ensure clear speech
4. **Try different models**: Whisper often works best

#### Evaluation Mismatches
1. **Enable ground truth preprocessing**: Normalize text format
2. **Check filename matching**: Ensure exact filename matches
3. **Verify CSV format**: Check encoding and delimiters
4. **Review transcript alignment**: Manual verification

### Model-Specific Issues

#### Whisper Issues
```bash
# Update Whisper
pip install --upgrade openai-whisper

# Use specific model size
./run_pipeline.sh --whisper-model large-v3
```

#### NeMo Model Issues
```bash
# Install NeMo toolkit
pip install nemo_toolkit[asr]

# Check CUDA compatibility
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### Transformers Issues
```bash
# Update transformers
pip install --upgrade transformers

# Clear cache
rm -rf ~/.cache/huggingface/transformers/
```

## 📚 Advanced Usage

### Custom Model Integration

Add custom ASR models to `run_all_asrs.py`:

```python
MODELS['custom_model'] = {
    'path': 'path/to/custom/model',
    'framework': 'transformers',  # or 'whisper', 'nemo'
    'memory_requirement': '4GB',
    'preprocessing': custom_preprocess_function,
    'postprocessing': custom_postprocess_function
}
```

### Batch Processing

Process multiple directories:

```bash
# Process multiple audio directories
for dir in /path/to/audio/*/; do
    ./run_pipeline.sh --input_dir "$dir" --ground_truth ground_truth.csv
done
```

### Integration with External Systems

```bash
# API integration example
curl -X POST http://api.example.com/asr \
    -F "audio=@audio_file.wav" \
    -F "config=@pipeline_config.json"

# Database integration
python3 scripts/upload_results_to_db.py \
    --results_dir pipeline_results_* \
    --database_url postgresql://user:pass@host:port/db
```

## 🔗 Related Documentation

- [LLM Pipeline Guide](LLM_PIPELINE_GUIDE.md) - Stage 2 processing
- [Error Handling Guide](ERROR_HANDLING_GUIDE.md) - Troubleshooting
- [Performance Tuning](PERFORMANCE_TUNING.md) - Optimization tips
- [Command Reference](COMMAND_REFERENCE.md) - All parameters

## 📞 Support

For ASR pipeline issues:

1. **Check error logs**: `error_analysis.log`
2. **Verify dependencies**: `pip list | grep -E "(torch|whisper|transformers|nemo)"`
3. **Test with sample data**: Use provided test dataset
4. **Review configuration**: Validate all required parameters
5. **Check system resources**: Monitor CPU, GPU, and memory usage

---

**Note**: This guide covers the ASR processing stage. For complete EMS call analysis, proceed to the LLM Pipeline Guide after successful ASR processing.