# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import torch
import torchaudio
from tqdm import tqdm

def load_vad_model():
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)
    return model

def detect_speech_segments(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    vad_model = load_vad_model()

    wav_files = [os.path.join(root, file)
                 for root, _, files in os.walk(input_folder)
                 for file in files if file.lower().endswith('.wav')]

    print(f"Found {len(wav_files)} WAV files in '{input_folder}'.")

    for wav_path in tqdm(wav_files, desc="Processing WAV files", unit="file"):
        file_name = os.path.splitext(os.path.basename(wav_path))[0]
        file_output_folder = os.path.join(output_folder, file_name)

        if os.path.exists(file_output_folder) and os.listdir(file_output_folder):
            print(f"Skipping already processed file: {wav_path}")
            continue  # Skip processing if the output folder already has files

        try:
            process_single_wav(wav_path, output_folder, vad_model)
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

def process_single_wav(wav_path, output_folder, vad_model):
    print(f"Processing {wav_path}...")

    waveform, sample_rate = torchaudio.load(wav_path)

    if waveform.shape[0] != 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 16000:
        print(f"Resampling from {sample_rate} Hz to 16000 Hz...")
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    print(f"Audio loaded: {waveform.shape}, Sample Rate: {sample_rate}")

    chunk_size = 512
    speech_timestamps = []
    current_start = None

    for frame_idx in range(0, waveform.shape[1] // chunk_size):
        start_sample = frame_idx * chunk_size
        end_sample = start_sample + chunk_size
        chunk = waveform[:, start_sample:end_sample]

        if chunk.shape[1] < chunk_size:
            padding = torch.zeros((1, chunk_size - chunk.shape[1]))
            chunk = torch.cat((chunk, padding), dim=1)

        try:
            is_speech = vad_model(chunk, sample_rate).item() > 0.1
        except Exception as e:
            print(f"Error during VAD: {e}")
            return

        if is_speech:
            if current_start is None:
                current_start = start_sample
        else:
            if current_start is not None:
                end_sample = start_sample
                if end_sample > current_start + 512:
                    speech_timestamps.append((current_start, end_sample))
                current_start = None

    print(f"Detected {len(speech_timestamps)} speech segments in {wav_path}.")

    if not speech_timestamps:
        print(f"No speech detected in {wav_path}. Skipping.")
        return

    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    file_output_folder = os.path.join(output_folder, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    for idx, (start_sample, end_sample) in enumerate(speech_timestamps):
        if start_sample >= end_sample:
            print(f"WARNING: Skipping invalid segment {idx + 1} (start_sample >= end_sample)")
            continue

        segment = waveform[:, start_sample:end_sample]
        save_path = os.path.join(file_output_folder, f"segment_{idx + 1}.wav")
        torchaudio.save(save_path, segment, sample_rate)
        print(f"Saved: {save_path}")

# Example usage
# input_folder = "/media/meow/Elements/ems_call/data/raw_wav/wav_data_2024all_n3"
input_folder = "/media/meow/One Touch/ems_call/final_wav_data_2024all_n3"

# output_folder = "/media/meow/Elements/ems_call/data/wav_data_2024all_n3_speech_segments"
output_folder = "/media/meow/One Touch/ems_call/final_wav_data_2024all_n3_speech_segments"

detect_speech_segments(input_folder, output_folder)
