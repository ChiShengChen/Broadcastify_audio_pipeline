# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def audio_filter(input_folder, output_folder, silence_threshold=-40.0, min_silence_len=1000, keep_duration=10000):
    """
    Scans all audio files in 'input_folder', detects segments that are above 'silence_threshold' dBFS,
    and for each segment found, keeps 'keep_duration' milliseconds from the segment's start (or until
    the end of the file), then exports these snippets to 'output_folder'. Ensures no overlapping regions.

    Parameters:
        input_folder (str)       : Path to the folder containing input audio files.
        output_folder (str)      : Path to the folder where the snippets will be saved.
        silence_threshold (float): Audio threshold in dBFS. Any volume above this value is considered ???non-silent.???
        min_silence_len (int)    : The minimum length (in ms) of audio above the threshold to consider it a valid segment.
        keep_duration (int)      : How many milliseconds to keep once a non-silent segment is detected.
    """

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all audio files in the input folder
    audio_files = [file_name for file_name in os.listdir(input_folder) if file_name.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    # Use tqdm to iterate through the files
    for file_name in tqdm(audio_files, desc="Processing audio files"):
        input_path = os.path.join(input_folder, file_name)
        
        # Load the audio file using pydub
        audio = AudioSegment.from_file(input_path)

        # Detect non-silent ranges (i.e., segments above the specified threshold)
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_threshold
        )

        # Initialize the end of the last processed snippet to avoid overlap
        last_end_time = 0

        # For each detected non-silent range
        for idx, (start_ms, end_ms) in enumerate(nonsilent_ranges):
            # Skip this range if it overlaps with the last processed snippet
            if start_ms < last_end_time:
                continue

            # Calculate the snippet end time
            snippet_end = start_ms + keep_duration

            # If it exceeds the audio length, limit it to the end of the file
            if snippet_end > len(audio):
                snippet_end = len(audio)

            # Extract the snippet
            snippet = audio[start_ms:snippet_end]

            # Update last_end_time to the end of the current snippet
            last_end_time = snippet_end

            # Form the output file name
            base_name, _ = os.path.splitext(file_name)
            output_file_name = f"{base_name}_snippet_{idx}.wav"
            output_path = os.path.join(output_folder, output_file_name)

            # Export the snippet as WAV
            snippet.export(output_path, format="wav")

def main():
     # Example paths (adjust these to your actual input and output folders)
    in_folder = "/media/meow/Elements/ems_call/data/data_2024all_n3"
    out_folder = "/media/meow/Elements/ems_call/data/20250123_data_2024all_n3_keep10s"

    # Ensure the download folder exists
    os.makedirs(out_folder, exist_ok=True)

    # Run the audio filter
    audio_filter(
        input_folder=in_folder,
        output_folder=out_folder,
        silence_threshold=-40.0, # Detect segments above -40dBFS
        min_silence_len=1000,    # At least 1 second of audio above the threshold
        keep_duration=10000      # Once detected, keep 10 seconds (10000 ms)
    )

if __name__ == "__main__":
    main()
