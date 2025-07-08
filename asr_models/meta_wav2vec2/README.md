# Meta Wav2Vec2 Audio Transcription Tool

This tool uses Meta's Wav2Vec2 model to transcribe English audio files into text.

## Model Features

- **Model**: `facebook/wav2vec2-large-960h-lv60-self` (or other compatible models)
- **Language**: English
- **Parameters**: ~300M
- **Highlights**: A high-performance model trained on the LibriSpeech 960-hour English corpus.

## Installation

Before using the tool, please install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

There are two ways to use this tool: for a single file or for a batch of files in a directory.

### Method 1: Transcribing a Directory of .wav files (Recommended)

Use the `run_transcription.py` script to process all `.wav` files in a given directory.

**Syntax:**
```bash
python run_transcription.py <source_directory> [--model MODEL_NAME] [--output_dir OUTPUT_DIRECTORY] [--hf_token HF_TOKEN]
```

**Example:**
```bash
# Transcribe all .wav files in 'my_audio_files' using the default model
python run_transcription.py ./my_audio_files

# Use a different model and specify an output directory
python run_transcription.py ./my_audio_files --model 'jonatasgrosman/wav2vec2-large-xls-r-51-espeak-cv-ft' --output_dir ./transcripts
```

### Method 2: Transcribing a Single Audio File

Use the `wav2vec_transcribe.py` script for more granular control over a single file.

**Syntax:**
```bash
python wav2vec_transcribe.py <model_name> <audio_file_path> [--output_file OUTPUT_PATH] [--hf_token HF_TOKEN]
```

**Example:**
```bash
# Transcribe a single file with the default model
python wav2vec_transcribe.py facebook/wav2vec2-large-960h-lv60-self ./my_audio.wav

# Specify an output file
python wav2vec_transcribe.py facebook/wav2vec2-large-960h-lv60-self ./my_audio.wav --output_file ./my_transcription.txt
```


## Features

- **Flexible**: Transcribe a single audio file or batch process an entire directory.
- **Custom Models**: Supports any Wav2Vec2-compatible model from the Hugging Face Hub or a local path.
- **Automatic Naming**: Generates clear, descriptive output filenames based on the model and audio file name.
- **GPU Acceleration**: Automatically uses GPU if available for faster processing.
- **Resampling**: Audio files are automatically resampled to the required 16kHz.

## Output Format

When using the batch script (`run_transcription.py`), transcriptions are saved in the execution directory by default. The output filename is generated automatically:
```
<model_basename>_<original_wav_basename>.txt
```

**Example:**
- **Original file**: `202412061019-707526-14744_call_6.wav`
- **Model**: `facebook/wav2vec2-large-960h-lv60-self`
- **Transcription file**: `wav2vec2-large-960h-lv60-self_202412061019-707526-14744_call_6.txt`

## System Requirements

- Python 3.8+
- PyTorch (CUDA support recommended for performance)
- Transformers library
- torchaudio library
- Sufficient disk space for model downloads (the default model is ~1.2GB). 