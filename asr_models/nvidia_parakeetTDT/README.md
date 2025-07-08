# NVIDIA Parakeet TDT 0.6B v2 Audio Transcription Tool

This tool uses the NVIDIA Parakeet TDT 0.6B v2 model to transcribe English audio files into text.

## Model Features

- **Model**: NVIDIA Parakeet TDT 0.6B v2
- **Architecture**: CTC (Connectionist Temporal Classification)
- **Language**: English
- **Parameters**: ~600M
- **Highlights**: A high-performance speech recognition model developed by NVIDIA, designed for enterprise-level applications.

## Dependencies Installation

Before use, please install the necessary dependencies:

```bash
pip install -r requirements.txt
```

**Note**: The NeMo toolkit is large, and installation may take some time. It is recommended to install it in an environment with a good network connection.

## Usage

### Method 1: Using the Simple Run Script (Recommended)

This script is designed for batch processing all subdirectories within a main source folder.

```bash
python run_transcription.py
```

### Method 2: Running the Main Transcription Script Directly

This provides the same functionality as the run script but can be called directly.

```bash
python parakeet_transcribe.py
```

### Method 3: Transcribing a Single Folder

If you need to process only one specific folder, use the `single_folder_transcribe.py` script. You will need to edit the `target_folder` variable inside the script.

```bash
python single_folder_transcribe.py
```

## Features

- Automatically traverses all subdirectories within `/media/meow/One Touch/ems_call/long_calls_filtered`.
- Finds all `.wav` files in each subdirectory.
- Uses the NVIDIA Parakeet TDT 0.6B v2 model for English transcription.
- Saves transcription results in the format `parakeet-tdt-0.6b-v2_{original_filename}.txt`.
- Automatically splits long audio files into 30-second chunks to manage memory usage.
- Skips already transcribed files.
- Supports GPU acceleration if available.
- Supports two model loading backends: Hugging Face Transformers and NVIDIA NeMo.
- Provides detailed progress logging.

## Output Format

Transcription files are saved in the same directory as the original audio files, with the following naming convention:
```
parakeet-tdt-0.6b-v2_{original_wav_filename}.txt
```

**Example:**
- **Original file**: `202412061019-707526-14744_call_6.wav`
- **Transcription file**: `parakeet-tdt-0.6b-v2_202412061019-707526-14744_call_6.txt`

## Processing Method

- Audio files are automatically resampled to 16kHz.
- Long audio files are split into 30-second chunks.
- Transcriptions from each chunk are automatically merged.
- Error recovery: Failure on a single chunk does not interrupt the entire transcription process.

## Model Loading

The tool supports two model loading backends:
1.  **Hugging Face Transformers** (Attempted first)
2.  **NVIDIA NeMo** (Fallback option)

The system will automatically select the available loading method.

## System Requirements

- Python 3.8+
- PyTorch (CUDA support is recommended)
- NVIDIA NeMo toolkit
- Transformers library
- Librosa for audio processing
- Sufficient disk space (the first run will download model files of about 2.4GB).

## Notes

- The NVIDIA Parakeet TDT model (~2.4GB) will be downloaded automatically on the first run.
- GPU acceleration will be used automatically if a compatible GPU is detected.
- The NeMo toolkit installation can be time-consuming.
- The model is optimized for English speech recognition.
- The script is designed to handle various audio formats by converting them as needed.

## Troubleshooting

### Model Loading Failure
If you encounter issues loading the model, please ensure:
1.  You have a stable internet connection.
2.  There is enough disk space.
3.  CUDA drivers are installed correctly (if using GPU).

### Out of Memory
If you run into memory issues:
1.  The script automatically uses chunking to manage memory.
2.  You can adjust the `chunk_length` parameter in the script.
3.  Consider running in CPU mode if GPU memory is insufficient. 