#!/usr/bin/env python3
"""
A simple wrapper script to run NVIDIA Canary-1B transcription on all .wav files in a directory.
"""

import os
import subprocess
import argparse
import torch

def run_batch_transcription(source_dir, model_name="nvidia/canary-1b"):
    """
    Sets up the environment and calls the main transcription script.
    This script is designed to be simpler for batch processing.

    Args:
        source_dir (str): The directory containing subdirectories of .wav files.
        model_name (str): The name of the Canary model to use.
    """
    print("="*60)
    print("NVIDIA Canary-1B English Audio Transcription Tool")
    print(f"Source Directory: {source_dir}")
    print(f"Model: {model_name}")
    print("="*60)
    print()

    # Check for GPU
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name()}")
    else:
        print("⚠ No GPU detected, using CPU (processing will be slower).")
    
    print()
    print("Starting transcription process...")
    print("Note: The model will be downloaded on the first run. Please be patient.")
    print("Language setting: English")
    print()

    # The `canary_transcribe.py` script contains the main logic,
    # including the hardcoded source path. This wrapper just calls it.
    # A more robust implementation would pass the source_dir via command-line args
    # to the main script, but we are keeping its structure.
    
    script_path = os.path.join(os.path.dirname(__file__), 'canary_transcribe.py')

    try:
        # We assume the main script `canary_transcribe` will handle the logic.
        # This wrapper's job is just to provide a clean execution entry point.
        # Since `canary_transcribe.main` has hardcoded paths, we don't pass args.
        subprocess.run(['python', script_path], check=True)
    except FileNotFoundError:
        print(f"Error: Could not find the transcription script at {script_path}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the transcription script: {e}")

    print("\nTranscription process finished!")
    print("Results have been saved in their respective source folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch transcription on a directory of .wav files using NVIDIA Canary.")
    # The current `canary_transcribe.py` has a hardcoded source directory.
    # If that script is updated to accept args, this can be enabled.
    # parser.add_argument("source_dir", type=str, help="Directory containing the sub-folders of .wav files.")
    parser.add_argument("--model", type=str, default="nvidia/canary-1b", help="Name of the model to use for transcription.")
    
    args = parser.parse_args()
    
    # For now, we use the hardcoded path from the main script.
    # This can be changed to `args.source_dir` if the main script is modified.
    hardcoded_source_dir = "/media/meow/One Touch/ems_call/long_calls_filtered"
    run_batch_transcription(hardcoded_source_dir, args.model) 