# -*- coding: utf-8 -*-
import os
import wave
import contextlib
from tqdm import tqdm

# --- Configuration ---
# The directory containing the merged call audio files to be analyzed.
TARGET_DIR = "/media/meow/One Touch/ems_call/merged_calls_by_timestamp"
# --- End of Configuration ---

def analyze_merged_calls(directory):
    """
    Scans a directory for .wav files, counts them, and categorizes them by duration.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    # Initialize counters
    total_files = 0
    less_than_1_min = 0
    between_1_and_2_min = 0
    more_than_2_min = 0

    print(f"Scanning directory: {directory}")

    # Use os.walk to recursively find all .wav files
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    if not wav_files:
        print("No .wav files found in the specified directory.")
        return
        
    total_files = len(wav_files)
    print(f"Found {total_files} .wav files. Analyzing durations...")

    # Process each file with a progress bar
    for file_path in tqdm(wav_files, desc="Analyzing files", unit="file"):
        try:
            with contextlib.closing(wave.open(file_path, 'r')) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration_s = frames / float(rate)

                # Categorize based on duration
                if duration_s < 60:
                    less_than_1_min += 1
                elif 60 <= duration_s < 120:
                    between_1_and_2_min += 1
                else:
                    more_than_2_min += 1
        except Exception as e:
            print(f"\nWarning: Could not process file '{file_path}'. Error: {e}")

    # Print the final statistics
    print("\n--- Analysis Complete ---")
    print(f"Total .wav files found: {total_files}")
    print("\nDuration Distribution:")
    print(f"  - Less than 1 minute (< 60s):    {less_than_1_min} files")
    print(f"  - 1 to 2 minutes (60s - 120s): {between_1_and_2_min} files")
    print(f"  - More than 2 minutes (>= 120s): {more_than_2_min} files")
    print("-------------------------")


if __name__ == "__main__":
    analyze_merged_calls(TARGET_DIR) 