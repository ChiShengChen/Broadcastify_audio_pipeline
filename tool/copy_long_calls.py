# -*- coding: utf-8 -*-
import os
import shutil
import torchaudio
from tqdm import tqdm

# --- Configuration ---
# The source directory containing the merged call audio files.
SOURCE_DIR = "/media/meow/One Touch/ems_call/merged_calls_by_timestamp"
# The destination directory where long audio files will be copied.
DEST_DIR = "/media/meow/One Touch/ems_call/long_calls_filtered"
# The minimum duration in seconds for a file to be copied.
MIN_DURATION_S = 60
# --- End of Configuration ---

def copy_long_duration_files(source_dir, dest_dir, min_duration):
    """
    Scans a source directory for .wav files, and copies those longer than
    min_duration to a destination directory, preserving the folder structure.
    """
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Filtering files from: {source_dir}")
    print(f"Copying files >= {min_duration}s to: {dest_dir}")

    # Find all .wav files recursively
    wav_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    if not wav_files:
        print("No .wav files found in the source directory.")
        return

    copied_count = 0
    # Process each file with a progress bar
    for file_path in tqdm(wav_files, desc="Filtering files", unit="file"):
        try:
            info = torchaudio.info(file_path)
            duration_s = info.num_frames / info.sample_rate

            # Check if the duration meets the threshold
            if duration_s >= min_duration:
                # Construct the destination path, preserving subdirectory structure
                relative_path = os.path.relpath(file_path, source_dir)
                dest_path = os.path.join(dest_dir, relative_path)
                
                # Create the destination folder if it doesn't exist
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Copy the file
                shutil.copy2(file_path, dest_path)
                copied_count += 1

        except Exception as e:
            print(f"\nWarning: Could not process file '{file_path}'. Error: {e}")

    # Print the final summary
    print("\n--- Filtering Complete ---")
    print(f"Total files scanned: {len(wav_files)}")
    print(f"Copied {copied_count} files with duration >= {min_duration} seconds.")
    print("--------------------------")


if __name__ == "__main__":
    copy_long_duration_files(SOURCE_DIR, DEST_DIR, MIN_DURATION_S) 