# -*- coding: utf-8 -*-
import os
import csv
import torchaudio
from tqdm import tqdm
from collections import Counter

# --- Configuration ---
# The directory containing the audio files to be analyzed.
TARGET_DIR = "/media/meow/One Touch/ems_call/long_calls_filtered"
# The path for the output CSV log file.
OUTPUT_LOG_FILE = "/media/meow/One Touch/ems_call/long_calls_analysis_log.csv"
# --- End of Configuration ---

def analyze_audio_properties(directory, output_csv_path):
    """
    Scans a directory for .wav files, analyzes their properties,
    and saves the results to a CSV log file.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Analyzing files in: {directory}")
    print(f"Saving analysis log to: {output_csv_path}")

    # Find all .wav files recursively
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    if not wav_files:
        print("No .wav files found in the directory.")
        return

    # Lists to store data for summary
    durations = []
    sample_rates = []
    channel_counts = []

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Define CSV header
            fieldnames = ['relative_path', 'duration_s', 'sample_rate_hz', 'channels', 'bit_depth', 'file_size_kb']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Process each file
            for file_path in tqdm(wav_files, desc="Analyzing properties", unit="file"):
                try:
                    info = torchaudio.info(file_path)
                    
                    # Extract properties
                    duration_s = info.num_frames / info.sample_rate
                    file_size_kb = os.path.getsize(file_path) / 1024
                    
                    # Store data for summary
                    durations.append(duration_s)
                    sample_rates.append(info.sample_rate)
                    channel_counts.append(info.num_channels)
                    
                    # Write row to CSV
                    writer.writerow({
                        'relative_path': os.path.relpath(file_path, directory),
                        'duration_s': f"{duration_s:.2f}",
                        'sample_rate_hz': info.sample_rate,
                        'channels': info.num_channels,
                        'bit_depth': info.bits_per_sample,
                        'file_size_kb': f"{file_size_kb:.2f}"
                    })

                except Exception as e:
                    print(f"\nWarning: Could not process file '{file_path}'. Error: {e}")

    except IOError as e:
        print(f"\nError: Could not write to log file at '{output_csv_path}'. Error: {e}")
        return

    # --- Print Summary Report ---
    total_files = len(durations)
    if total_files == 0:
        print("\nNo files were successfully analyzed.")
        return
        
    avg_duration = sum(durations) / total_files
    sample_rate_dist = Counter(sample_rates)
    channel_dist = Counter(channel_counts)

    print("\n--- Analysis Summary ---")
    print(f"Total files analyzed: {total_files}")
    print(f"Average duration: {avg_duration:.2f} seconds")
    
    print("\nSample Rate Distribution:")
    for rate, count in sample_rate_dist.items():
        print(f"  - {rate} Hz: {count} files")
        
    print("\nChannel Distribution:")
    for channels, count in channel_dist.items():
        channel_str = "Mono" if channels == 1 else "Stereo" if channels == 2 else f"{channels} Channels"
        print(f"  - {channel_str}: {count} files")
        
    print("\n------------------------")
    print(f"Detailed analysis saved to '{output_csv_path}'")


if __name__ == "__main__":
    analyze_audio_properties(TARGET_DIR, OUTPUT_LOG_FILE) 