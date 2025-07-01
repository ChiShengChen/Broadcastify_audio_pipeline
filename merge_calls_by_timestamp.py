# -*- coding: utf-8 -*-
import os
import re
from pydub import AudioSegment
from tqdm import tqdm

# --- Configuration ---
# The root directory containing the speech segment folders
INPUT_DIR = "/media/meow/One Touch/ems_call/final_wav_data_2024all_n3_speech_segments"
# The directory where the merged call audio files will be saved
OUTPUT_DIR = "/media/meow/One Touch/ems_call/merged_calls_by_timestamp"
# If the silence between two segments is longer than this (in seconds),
# consider it a new call. This is the most important parameter to tune.
CALL_BREAK_THRESHOLD_S = 15

# --- End of Configuration ---

def parse_time_points(txt_file_path):
    """Parses the _time_point.txt file and returns a sorted list of segment details."""
    segments = []
    with open(txt_file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"Segment (\d+): (\d+\.\d+)s - (\d+\.\d+)s", line)
            if match:
                seg_num = int(match.group(1))
                start_s = float(match.group(2))
                end_s = float(match.group(3))
                segments.append({"num": seg_num, "start": start_s, "end": end_s})
    
    # Sort by segment number to ensure chronological order
    segments.sort(key=lambda x: x["num"])
    return segments

def merge_audio_segments(segment_group, folder_path, output_filename):
    """
    Merges a group of audio segments into a single file,
    inserting silence based on their timestamps.
    """
    if not segment_group:
        return

    # Initialize with the first segment
    base_audio = AudioSegment.from_wav(os.path.join(folder_path, f"segment_{segment_group[0]['num']}.wav"))

    # Iterate through the rest of the segments in the group
    for i in range(len(segment_group) - 1):
        current_segment_info = segment_group[i]
        next_segment_info = segment_group[i+1]

        # Calculate silence duration in milliseconds
        silence_duration_ms = (next_segment_info['start'] - current_segment_info['end']) * 1000
        
        # Create silence segment
        silence = AudioSegment.silent(duration=silence_duration_ms)

        # Load next audio segment
        next_audio = AudioSegment.from_wav(os.path.join(folder_path, f"segment_{next_segment_info['num']}.wav"))

        # Append silence and the next audio segment
        base_audio += silence + next_audio

    # Export the final merged audio
    base_audio.export(output_filename, format="wav")
    print(f"  - Saved merged call: {os.path.basename(output_filename)}")

def process_folder(folder_path, output_dir_for_folder):
    """
    Processes a single folder of segments, identifies calls, and merges them.
    """
    folder_name = os.path.basename(folder_path)
    print(f"Processing folder: {folder_name}...")

    # Find the time_point.txt file
    time_point_file = None
    for file in os.listdir(folder_path):
        if file.endswith("_time_point.txt"):
            time_point_file = os.path.join(folder_path, file)
            break
    
    if not time_point_file:
        print(f"  - Warning: No time_point.txt file found in {folder_name}. Skipping.")
        return

    segments = parse_time_points(time_point_file)
    if not segments:
        print(f"  - Info: No segments found in {folder_name}. Skipping.")
        return

    os.makedirs(output_dir_for_folder, exist_ok=True)
    
    current_call_segments = []
    call_counter = 1

    for i, segment_info in enumerate(segments):
        if not current_call_segments:
            # Start of a new call
            current_call_segments.append(segment_info)
        else:
            # Compare with the previous segment to check for a call break
            last_segment_end_time = current_call_segments[-1]['end']
            gap_duration = segment_info['start'] - last_segment_end_time

            if gap_duration > CALL_BREAK_THRESHOLD_S:
                # A call break is detected, so merge the previous call
                output_filename = os.path.join(output_dir_for_folder, f"{folder_name}_call_{call_counter}.wav")
                merge_audio_segments(current_call_segments, folder_path, output_filename)
                
                # Start a new call
                current_call_segments = [segment_info]
                call_counter += 1
            else:
                # The segment belongs to the current call
                current_call_segments.append(segment_info)
    
    # Merge the last remaining call in the folder
    if current_call_segments:
        output_filename = os.path.join(output_dir_for_folder, f"{folder_name}_call_{call_counter}.wav")
        merge_audio_segments(current_call_segments, folder_path, output_filename)


def main():
    """Main function to iterate through all segment directories and process them."""
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory not found at '{INPUT_DIR}'")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    subfolders = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]

    for folder_path in tqdm(subfolders, desc="Processing all folders"):
        folder_name = os.path.basename(folder_path)
        output_subfolder = os.path.join(OUTPUT_DIR, folder_name)
        process_folder(folder_path, output_subfolder)

    print("\nProcessing complete!")
    print(f"Merged calls are saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main() 