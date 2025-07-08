#!/usr/bin/env python3
"""
A simple wrapper script to run OpenAI Whisper transcription.
"""

import os
import subprocess
import argparse
import torch

def run_batch_transcription(source_dir, model_name="large-v3"):
    """
    Sets up the environment and calls the main Whisper transcription script.
    """
    print("="*60)
    print(f"OpenAI Whisper {model_name} English Audio Transcription Tool")
    print(f"Source Directory: {source_dir}")
    print("="*60)
    print()

    # Check for GPU
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name()}")
    else:
        print("⚠ No GPU detected, using CPU (processing will be slower).")
    
    print()
    print("Starting English transcription process...")
    print("Note: The model will be downloaded on the first run. Please be patient.")
    print()

    # The main logic is in `whisper_transcribe.py`. This script just calls it.
    # We are keeping the hardcoded source directory as the main script expects it.
    script_path = os.path.join(os.path.dirname(__file__), 'whisper_transcribe.py')

    try:
        # Since `whisper_transcribe.py` is hardcoded, we don't need to pass arguments.
        # If it were parameterized, we would pass them here.
        subprocess.run(['python', script_path], check=True)
    except FileNotFoundError:
        print(f"Error: Could not find the transcription script at {script_path}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the transcription script: {e}")

    print("\nTranscription process finished!")
    print("Results have been saved in their respective source folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch transcription using OpenAI Whisper.")
    
    # The underlying `whisper_transcribe.py` script has hardcoded paths and model names.
    # If it were parameterized, we would add arguments here.
    # For example:
    # parser.add_argument("source_dir", type=str, help="Directory containing sub-folders of .wav files.")
    # parser.add_argument("--model_name", type=str, default="large-v3", help="Name of the Whisper model to use.")
    
    args = parser.parse_args()
    
    # Using the hardcoded path from the main script.
    hardcoded_source_dir = "/media/meow/One Touch/ems_call/long_calls_filtered"
    run_batch_transcription(hardcoded_source_dir) 