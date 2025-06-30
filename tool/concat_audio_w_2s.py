# -*- coding: utf-8 -*-
import os
from pydub import AudioSegment

def concatenate_audio(folder_path, output_dir):
    """ Concatenates all audio files in a subfolder, sorts them numerically, adds 2 seconds of silence, and saves to the output directory. """
    audio_segments = []
    silence_gap = AudioSegment.silent(duration=2000)  # 2 seconds of silence

    # Retrieve and sort all .wav files numerically
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.wav')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    if not files:
        print(f"Warning: No usable audio files found in {folder_path}")
        return

    # Concatenate audio files with silence in between
    for file in files:
        file_path = os.path.join(folder_path, file)
        audio_segment = AudioSegment.from_wav(file_path)
        audio_segments.append(audio_segment)
        audio_segments.append(silence_gap)  # Insert silence

    # Remove the last silence gap
    final_audio = sum(audio_segments[:-1])

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set output filename
    folder_name = os.path.basename(folder_path)
    output_path = os.path.join(output_dir, f"concat_segment_{folder_name}.wav")

    # Export the concatenated audio
    final_audio.export(output_path, format="wav")
    print(f"Saved: {output_path}")

def process_all_folders(root_folder, output_root):
    """ Iterates through the main directory, processes all subfolders, and saves concatenated audio files in the output directory. """
    for sub_folder in os.listdir(root_folder):
        sub_folder_path = os.path.join(root_folder, sub_folder)
        if os.path.isdir(sub_folder_path):
            concatenate_audio(sub_folder_path, output_root)

if __name__ == "__main__":
    root_dir = "/media/meow/One Touch/ems_call/final_wav_data_2024all_n3_speech_segments"
    output_dir = "/media/meow/One Touch/ems_call/final_wav_concated_2s_processed_audio"  # Change this to your desired output directory
    process_all_folders(root_dir, output_dir)
