import os
import re
import numpy as np
from scipy.io import wavfile

# Define base directory
base_dir = "/media/meow/One Touch/ems_call/test_exp_data_200"
base_dir_0 = "/media/meow/One Touch/ems_call"

output_dir = os.path.join(base_dir_0, "merged_audio_test_exp_data_200_vad0_1")
os.makedirs(output_dir, exist_ok=True)

def parse_time_points(txt_file):
    """Parses the *_time_point.txt file and returns a list of segment time ranges."""
    segments = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"Segment \d+: (\d+\.\d+)s - (\d+\.\d+)s", line)
            if match:
                start, end = float(match.group(1)), float(match.group(2))
                segments.append((start, end))
    return segments

def find_audio_files(folder):
    """Finds all .wav files in a directory, ensuring they match segment order."""
    files = sorted(
        [f for f in os.listdir(folder) if f.startswith("segment_") and f.endswith(".wav")],
        key=lambda x: int(re.search(r"segment_(\d+)", x).group(1))  # Sort numerically
    )
    return [os.path.join(folder, f) for f in files]

def load_audio(file_path):
    """Reads a .wav audio file and returns the sample rate and audio data."""
    rate, data = wavfile.read(file_path)

    # Ensure audio is in int16 format (if it's in float format)
    if data.dtype != np.int16:
        data = (data * 32767).astype(np.int16)

    return rate, data

def save_audio(file_path, rate, data):
    """Saves the merged audio file."""
    wavfile.write(file_path, rate, data)

def process_segments(folder):
    """Processes segments in a folder, merges audio files based on timing, and saves the output."""
    # Find the time_point.txt file
    txt_file = None
    for file in os.listdir(folder):
        if file.endswith("_time_point.txt"):
            txt_file = os.path.join(folder, file)
            break

    if not txt_file:
        print(f"Skipping {folder}: No time_point.txt file found.")
        return

    segments = parse_time_points(txt_file)
    audio_files = find_audio_files(folder)

    if len(segments) != len(audio_files):
        print(f"Warning: Mismatch between segments and audio files in {folder}. Skipping...")
        return

    merged_audio = []
    last_end_time = None
    rate = None
    part_count = 1  # Counter for merged files
    output_prefix = os.path.basename(folder)  # Folder name as prefix

    for i, (start, end) in enumerate(segments):
        audio_file = audio_files[i]
        current_rate, audio_data = load_audio(audio_file)

        # Ensure consistent sample rate
        if rate is None:
            rate = current_rate
        elif rate != current_rate:
            print(f"Warning: Sample rate mismatch in {audio_file}. Skipping...")
            continue

        # Compute time gap between consecutive segments
        if last_end_time is not None:
            gap = start - last_end_time
            if gap < 60:  # If gap is less than 1 minute, insert silence
                silence = np.zeros(int(rate * gap), dtype=np.int16)
                merged_audio.append(silence)
            else:  # If gap is more than 1 minute, save current merged file and start a new one
                if merged_audio:
                    output_path = f"{output_dir}/merge_{output_prefix}_{part_count}.wav"
                    save_audio(output_path, rate, np.concatenate(merged_audio))
                    print(f"Saved: {output_path}")
                    part_count += 1
                merged_audio = []  # Reset audio list

        merged_audio.append(audio_data)
        last_end_time = end

    # Save the last merged file
    if merged_audio:
        output_path = f"{output_dir}/merge_{output_prefix}_{part_count}.wav"
        save_audio(output_path, rate, np.concatenate(merged_audio))
        print(f"Saved: {output_path}")

def main():
    """Main function that scans all experiment folders and processes them."""
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            process_segments(folder_path)

if __name__ == "__main__":
    main()
