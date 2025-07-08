#!/usr/bin/env python3
"""
A simple wrapper script to run NVIDIA Parakeet TDT transcription.
"""

import os
import subprocess
import argparse
import torch

def run_batch_transcription(source_dir, model_name="parakeet-tdt-0.6b-v2"):
    """
    Sets up the environment and calls the main transcription script.
    """
    print("="*60)
    print("NVIDIA Parakeet TDT 0.6B v2 English Audio Transcription Tool")
    print(f"Source Directory: {source_dir}")
    print(f"Model Name: {model_name}")
    print("="*60)
    print()

    # Check for GPU
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("⚠ No GPU detected, using CPU (processing will be slower).")
    
    print()
    print("Model Information:")
    print("  - Using Model: NVIDIA Parakeet TDT 0.6B v2")
    print("  - Model Type: CTC (Connectionist Temporal Classification)")
    print("  - Language Support: English")
    print("  - Model Size: ~600M parameters")
    print("  - Domain: General-purpose speech recognition")
    print()
    
    print("Starting English transcription process...")
    print("Note: The model will be downloaded on the first run. Please be patient.")
    print("Audio files will be processed in 30-second chunks.")
    print()

    # The main logic is in `parakeet_transcribe.py`. This script just calls it.
    # We are keeping the hardcoded source directory as the main script expects it.
    script_path = os.path.join(os.path.dirname(__file__), 'parakeet_transcribe.py')

    try:
        subprocess.run(['python', script_path], check=True)
    except FileNotFoundError:
        print(f"Error: Could not find the transcription script at {script_path}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the transcription script: {e}")

    print("\nTranscription process finished!")
    print("Results have been saved in their respective source folders.")
    print(f"Filename format: {model_name}_<original_filename>.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch transcription using NVIDIA Parakeet TDT.")
    
    # The underlying script has a hardcoded source directory.
    # If it were parameterized, we would add an argument for it here.
    # parser.add_argument("source_dir", type=str, help="Directory containing sub-folders of .wav files.")
    
    parser.add_argument("--model_name", type=str, default="parakeet-tdt-0.6b-v2", 
                        help="The name to use for the output files.")
    
    args = parser.parse_args()
    
    # Using the hardcoded path from the main script.
    hardcoded_source_dir = "/media/meow/One Touch/ems_call/long_calls_filtered"
    run_batch_transcription(hardcoded_source_dir, args.model_name) 