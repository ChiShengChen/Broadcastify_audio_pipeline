# -*- coding: utf-8 -*-
import os
import torchaudio
from tqdm import tqdm

# --- Configuration ---
# The directory containing the .wav files to be analyzed.
TARGET_DIR = "/media/meow/One Touch/ems_call/final_wav_data_2024all_n3_speech_segments"
# --- End of Configuration ---

def calculate_average_duration(directory):
    """
    Scans a directory for .wav files, counts them, and calculates the average duration.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    # Initialize counters
    total_files = 0
    total_duration_s = 0.0

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
    print(f"Found {total_files} .wav files. Calculating total duration...")

    # Process each file with a progress bar
    for file_path in tqdm(wav_files, desc="Processing files", unit="file"):
        try:
            info = torchaudio.info(file_path)
            duration_s = info.num_frames / info.sample_rate
            total_duration_s += duration_s
        except Exception as e:
            print(f"\nWarning: Could not process file '{file_path}'. Error: {e}")

    # Calculate average duration
    average_duration_s = 0
    if total_files > 0:
        average_duration_s = total_duration_s / total_files

    # Print the final statistics
    print("\n--- Analysis Complete ---")
    print(f"Total .wav files found: {total_files}")
    print(f"Total combined duration: {total_duration_s:.2f} seconds ({total_duration_s / 3600:.2f} hours)")
    print(f"Average file duration: {average_duration_s:.2f} seconds")
    print("-------------------------")


if __name__ == "__main__":
    calculate_average_duration(TARGET_DIR) 