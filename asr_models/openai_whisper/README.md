# OpenAI Whisper English Audio Transcription Tool

This tool uses the OpenAI Whisper `large-v3` model to transcribe English audio files into text.

## Dependencies Installation

Before use, please install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Using the Simple Run Script (Recommended)

This script is designed for batch processing all subdirectories within a main source folder.

```bash
python run_transcription.py
```

### Method 2: Running the Main Transcription Script Directly

This provides the same functionality as the run script but can be called directly.

```bash
python whisper_transcribe.py
```

### Method 3: Transcribing a Single Folder

If you need to process only one specific folder, use the `single_folder_transcribe.py` script. You will need to edit the `target_folder` variable inside the script.

```bash
python single_folder_transcribe.py
```

## Features

- Automatically traverses all subdirectories within `/media/meow/One Touch/ems_call/long_calls_filtered`.
- Finds all `.wav` files in each subdirectory.
- Uses the Whisper `large-v3` model for English transcription.
- Saves transcription results in the format `large-v3_{original_filename}.txt`.
- Automatically skips already transcribed files.
- Supports GPU acceleration if available.
- Optimized for English speech.
- Provides detailed progress logging.

## Output Format

Transcription files are saved in the same directory as the original audio files, with the following naming convention:
```
large-v3_{original_wav_filename}.txt
```

**Example:**
- **Original file**: `202412061019-707526-14744_call_6.wav`
- **Transcription file**: `large-v3_202412061019-707526-14744_call_6.txt`

## System Requirements

- Python 3.8+
- PyTorch (CUDA support is recommended)
- OpenAI Whisper library
- Sufficient disk space (the first run will download the model, which is ~3GB).

## Notes

- The Whisper `large-v3` model (~3GB) will be downloaded automatically on the first run. Ensure you have a stable internet connection.
- GPU acceleration will be used automatically if a compatible GPU is detected, significantly speeding up the process.
- The transcription language is set to English (`en`). If you need to change this, modify the `language="en"` parameter in the scripts. 