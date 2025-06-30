# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
from pydub import AudioSegment
from tqdm import tqdm

def convert_mp3_to_wav(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    mp3_files = [os.path.join(root, file)
                 for root, _, files in os.walk(input_folder)
                 for file in files if file.lower().endswith('.mp3')]

    print(f"In '{input_folder}' find {len(mp3_files)} MP3 files")

    for mp3_path in tqdm(mp3_files, desc="Converting MP3 to WAV", unit="file"):
        wav_filename = os.path.splitext(os.path.basename(mp3_path))[0] + '.wav'
        wav_path = os.path.join(output_folder, wav_filename)

        # Check if the .wav file already exists
        if os.path.exists(wav_path):
            print(f"Skipped {wav_path} as it already exists.")
            continue

        try:
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
            print(f"Done transfer {mp3_path} to {wav_path}")
        except Exception as e:
            print(f"Skip {mp3_path}: MP3 file may be broken, error: {e}")

# Usage example
input_folder = "/media/meow/Elements/ems_call/data/raw_wav/data_2024all_n3"
output_folder = "/media/meow/One Touch/ems_call/final_wav_data_2024all_n3"

convert_mp3_to_wav(input_folder, output_folder)
