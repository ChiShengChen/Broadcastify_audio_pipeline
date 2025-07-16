#!/usr/bin/env python3
import os
import wave
import contextlib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def check_wav_folder(folder_path, save_path="wav_duration_histogram.png"):
    # 1. Count the total number of .wav files in the folder
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    print(f"Total number of .wav files in the folder {folder_path}: {len(wav_files)}")

    if not wav_files:
        print("No .wav files found in the folder. Exiting program.")
        return

    # 2. Extract the first 8 characters (assuming YYYYMMDD format) to find the earliest and latest dates
    dates = []
    for filename in wav_files:
        date_str = filename[:8]  # Assuming the first 8 characters are YYYYMMDD, e.g., '20241112'
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d").date()
            dates.append(file_date)
        except ValueError:
            # Skip files where the first 8 characters cannot be parsed as a date
            pass

    if not dates:
        print("None of the .wav files have a valid date format. Exiting program.")
        return

    oldest_date = min(dates)
    newest_date = max(dates)

    print(f"Earliest date: {oldest_date} (YYYY-MM-DD)")
    print(f"Latest date: {newest_date} (YYYY-MM-DD)")

    # 3. Check for missing dates within the date range
    missing_dates = []
    current_date = oldest_date
    date_set = set(dates)  # Convert list to set for fast lookup

    while current_date <= newest_date:
        if current_date not in date_set:
            missing_dates.append(current_date)
        current_date += timedelta(days=1)

    if missing_dates:
        print("The following dates have missing .wav files:")
        for d in missing_dates:
            print(d)
    else:
        print("No missing dates found within the date range.")

    # 4. Generate a histogram of file durations
    durations = []
    for filename in wav_files:
        filepath = os.path.join(folder_path, filename)
        with contextlib.closing(wave.open(filepath, 'r')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)  # Compute duration in seconds
            durations.append(duration)

    # Plot histogram and save image
    plt.figure(figsize=(8, 6))
    plt.hist(durations, bins=20, edgecolor='black')  # Adjust bins as needed
    plt.xlabel("WAV Duration (seconds)")
    plt.ylabel("Number of Files")
    plt.title("Distribution of WAV File Durations")
    
    # Save the figure instead of showing it
    plt.savefig(save_path, dpi=300)
    print(f"Histogram saved as {save_path}")

if __name__ == "__main__":
    # Specify the folder path containing the .wav files
    folder_path = "/media/meow/Elements/ems_call/data/a_data_2024all_n3_keep10s"
    check_wav_folder(folder_path, "/media/meow/Elements/ems_call/data/wav_duration_histogram.png")
