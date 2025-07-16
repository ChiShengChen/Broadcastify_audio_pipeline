# -*- coding: utf-8 -*-
import os
import random
import shutil
from tqdm import tqdm

# --- Configuration ---
# The source directory containing the files to sample from.
SOURCE_DIR = "/media/meow/One Touch/ems_call/long_calls_filtered"

# The destination directories for the random samples.
DEST_DIR_1 = "/media/meow/One Touch/ems_call/random_samples_1"
DEST_DIR_2 = "/media/meow/One Touch/ems_call/random_samples_2"

# The number of random files to select for EACH directory.
NUM_SAMPLES = 25
# --- End of Configuration ---

def create_random_samples(source_dir, dest_dir1, dest_dir2, num_samples):
    """
    Selects two disjoint random samples of files from a source directory 
    and copies them to two separate destination directories.
    """
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    # Create destination directories
    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)
    print(f"Source directory: {source_dir}")
    print(f"Destination directories: \n1. {dest_dir1}\n2. {dest_dir2}")

    # Find all .wav files recursively
    all_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print("No .wav files found in the source directory.")
        return

    total_samples_needed = num_samples * 2

    # Adjust sample size if there are not enough files
    if len(all_files) < total_samples_needed:
        print(f"\nWarning: Found only {len(all_files)} files, which is less than the required {total_samples_needed} for two disjoint sets.")
        print("All available files will be split between the two directories.")
        total_samples_needed = len(all_files)
    
    # Randomly select files without replacement
    selected_files = random.sample(all_files, k=total_samples_needed)
    
    # Split the selected files into two lists
    midpoint = len(selected_files) // 2
    files_for_dest1 = selected_files[:midpoint]
    files_for_dest2 = selected_files[midpoint:]

    print(f"\nRandomly selected {len(files_for_dest1)} files for {os.path.basename(dest_dir1)} and {len(files_for_dest2)} for {os.path.basename(dest_dir2)}.")

    # Copy files to the first destination directory
    print(f"Copying {len(files_for_dest1)} files to {os.path.basename(dest_dir1)}...")
    for file_path in tqdm(files_for_dest1, desc=f"Copying to {os.path.basename(dest_dir1)}", unit="file"):
        try:
            shutil.copy2(file_path, os.path.join(dest_dir1, os.path.basename(file_path)))
        except Exception as e:
            print(f"\nError copying file '{file_path}' to '{dest_dir1}': {e}")

    # Copy files to the second destination directory
    print(f"\nCopying {len(files_for_dest2)} files to {os.path.basename(dest_dir2)}...")
    for file_path in tqdm(files_for_dest2, desc=f"Copying to {os.path.basename(dest_dir2)}", unit="file"):
        try:
            shutil.copy2(file_path, os.path.join(dest_dir2, os.path.basename(file_path)))
        except Exception as e:
            print(f"\nError copying file '{file_path}' to '{dest_dir2}': {e}")


    print("\n--- Process Complete ---")
    print(f"Successfully copied {len(files_for_dest1)} files to {os.path.basename(dest_dir1)}.")
    print(f"Successfully copied {len(files_for_dest2)} files to {os.path.basename(dest_dir2)}.")
    print("------------------------")

if __name__ == "__main__":
    create_random_samples(SOURCE_DIR, DEST_DIR_1, DEST_DIR_2, NUM_SAMPLES) 