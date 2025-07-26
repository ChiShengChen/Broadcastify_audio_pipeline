# EMS Call ASR Pipeline

A comprehensive Automatic Speech Recognition (ASR) evaluation pipeline for emergency medical service call analysis.

## 📋 Overview

This project provides a complete ASR evaluation system with advanced features including Voice Activity Detection (VAD), long audio processing, ground truth preprocessing, and comprehensive error handling. The pipeline supports multiple ASR models and provides detailed evaluation metrics.

## 🚀 Quick Start

### Basic Usage

```bash
# Run the main pipeline
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv

# Fix missing ASR files
./fix_missing_asr_integrated.sh \
    --pipeline_output_dir /path/to/pipeline_results \
    --fix_output_dir /path/to/fix_results \
    --ground_truth_file /path/to/ground_truth.csv
```

### Advanced Features

```bash
# Use VAD and long audio splitting
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-vad \
    --use-long-audio-split \
    --preprocess-ground-truth \
    --use-enhanced-preprocessor
```

## 🏗️ Project Structure

```
ems_call/
├── 📁 unit_test/           # Test files and test data
├── 📁 tool/               # Analysis tools and utilities
├── 📁 asr_models/         # ASR model configurations
├── 📁 data/               # Dataset directories
├── 📁 vb_ems_anotation/   # Annotation data
├── 📁 long_audio_test_dataset/  # Long audio test dataset
├── 📁 pipeline_results_*/ # Pipeline execution results
├── 📄 run_pipeline.sh     # Main pipeline script
├── 📄 evaluate_asr.py     # ASR evaluation core
├── 📄 run_all_asrs.py     # ASR model execution
├── 📄 long_audio_splitter.py  # Long audio segmentation
├── 📄 merge_split_transcripts.py  # Transcript merging
├── 📄 vad_pipeline.py     # VAD processing
├── 📄 enhanced_vad_pipeline.py  # Enhanced VAD
└── 📄 README.md           # This file
```

📖 **For detailed Chinese documentation, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**

## 🔧 Core Scripts

### `run_pipeline.sh`
The main ASR evaluation pipeline with features:
- **Optional VAD preprocessing**: Extract speech segments
- **Long audio splitting**: Prevent OOM errors
- **Multi-model ASR transcription**: Support for multiple ASR models
- **Ground truth preprocessing**: Improve matching accuracy
- **Evaluation metrics**: WER, MER, WIL calculations
- **Error handling**: Complete error detection and reporting
- **Status reporting**: Clear success/failure status

### `fix_missing_asr_integrated.sh`
Integrated tool for fixing missing ASR files:
- **Missing file detection**: Automatically identify missing transcript files
- **Root cause analysis**: Analyze possible reasons for missing files
- **Automatic repair**: Generate targeted repair scripts
- **Detailed reporting**: Provide comprehensive analysis reports

## 🧪 Testing

### Run Tests
```bash
# Error handling tests
cd unit_test
python3 test_error_handling.py

# Preprocessor integration tests
python3 test_enhanced_preprocessor_integration.py

# Pipeline status tests
python3 test_pipeline_status.py
```

### Test Coverage
- Error handling functionality
- Preprocessor integration
- Pipeline status reporting
- Component functionality
- Integration testing

## 🛠️ Tools and Utilities

### Analysis Tools (`tool/`)
- `analyze_asr_number_processing.py` - ASR number processing analysis
- `analyze_evaluation_issue.py` - Evaluation problem analysis
- `analyze_model_files_enhanced.py` - Enhanced model file analysis
- `analyze_model_files.py` - Model file analysis

### Preprocessing Tools (`tool/`)
- `smart_preprocess_ground_truth.py` - Smart preprocessor
- `enhanced_ground_truth_preprocessor.py` - Enhanced preprocessor
- `preprocess_ground_truth.py` - Basic preprocessor

### Repair Tools (`tool/`)
- `fix_missing_asr_integrated.sh` - Integrated repair script
- `fix_missing_asr_correct.sh` - Corrected repair script
- `fix_missing_asr_files.sh` - Missing file repair script

## 🔍 Key Features

### ASR Evaluation Pipeline
- **Multi-model support**: Whisper Large-v3, Wav2Vec2, Parakeet, Canary-1B
- **VAD preprocessing**: Optional voice activity detection
- **Long audio processing**: Automatic long audio file segmentation
- **Ground truth preprocessing**: Intelligent text normalization
- **Complete evaluation**: WER, MER, WIL metric calculations
- **Error handling**: Automatic error detection and reporting
- **Status reporting**: Clear success/failure status

### Repair Tools
- **Automatic detection**: Identify missing transcript files
- **Root cause analysis**: Analyze possible reasons for missing files
- **Intelligent repair**: Generate targeted repair scripts
- **Result integration**: Integrate repair results into original results

### Preprocessing Features
- **Basic preprocessor**: Simple text normalization
- **Smart preprocessor**: Adaptive text preprocessing
- **Enhanced preprocessor**: Comprehensive text normalization
- **Multiple modes**: Conservative and aggressive modes

## 📊 Output Results

### Pipeline Output
- `asr_evaluation_results.csv` - Evaluation results
- `model_file_analysis.txt` - Model file analysis
- `pipeline_summary.txt` - Pipeline summary
- `error_analysis.log` - Error analysis log

### Repair Output
- `missing_analysis.json` - Missing file analysis
- `rerun_missing_asr.sh` - Repair script
- `missing_files_report.txt` - Detailed report

## 🔧 Requirements

### Python Dependencies
```bash
pip install pandas jiwer torch transformers torchaudio nemo_toolkit[asr] openai-whisper tqdm scipy numpy pathlib2 soundfile pydub librosa
```

### System Requirements
- Python 3.7+
- FFmpeg (for audio processing)
- Sufficient disk space and memory

## 📝 Usage Examples

### 1. Basic Pipeline Execution
```bash
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv
```

### 2. With VAD and Long Audio Splitting
```bash
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-vad \
    --use-long-audio-split \
    --max-segment-duration 120.0
```

### 3. With Ground Truth Preprocessing
```bash
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --preprocess-ground-truth \
    --preprocess-mode aggressive
```

### 4. With Enhanced Preprocessor
```bash
./run_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-enhanced-preprocessor \
    --enhanced-preprocessor-mode aggressive
```

### 5. Fix Missing Files
```bash
./fix_missing_asr_integrated.sh \
    --pipeline_output_dir /path/to/pipeline_results \
    --fix_output_dir /path/to/fix_results \
    --ground_truth_file /path/to/ground_truth.csv \
    --models large-v3,canary-1b
```

## 🚨 Error Handling

The pipeline includes comprehensive error handling:
- **Automatic error detection**: Detects various types of errors
- **Detailed logging**: Records errors with timestamps and context
- **Error categorization**: Classifies errors by type
- **Status reporting**: Clear success/failure indication
- **Troubleshooting guidance**: Provides specific solutions

## 📚 Documentation

### Guides and Documentation
- [Error Handling Guide](tool/ERROR_HANDLING_GUIDE.md) - Complete error handling documentation
- [Pipeline Status Guide](tool/PIPELINE_STATUS_GUIDE.md) - Pipeline status reporting guide
- [Ground Truth Preprocessing Guide](tool/GROUND_TRUTH_PREPROCESSING_GUIDE.md) - Preprocessing documentation
- [Enhanced Preprocessor Guide](tool/ENHANCED_PREPROCESSOR_USAGE_GUIDE.md) - Enhanced preprocessor usage

### Test Documentation
- [Unit Test Guide](unit_test/README.md) - Testing documentation and examples
- [Tool Documentation](tool/README.md) - Tools and utilities documentation

## 🤝 Contributing

1. Add corresponding tests for new features
2. Place tool files in the `tool/` directory
3. Place test files in the `unit_test/` directory
4. Update relevant documentation and guides

## 📄 License

This project is for research and development purposes.

## 📞 Support

For issues and questions, please check the documentation in the `tool/` directory or run the appropriate test scripts in the `unit_test/` directory.

---

**Note**: This project is designed for emergency medical service call analysis and includes specialized preprocessing for medical terminology and emergency codes. 