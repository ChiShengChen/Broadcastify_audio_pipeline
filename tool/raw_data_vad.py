# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import torch
import torchaudio
import torchaudio.transforms as T
import os
from tqdm import tqdm
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_path, wav_path):
    """ Converts an MP3 file to WAV format, handling errors. """
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Skipping {mp3_path}: MP3 is corrupt. Error: {e}")
        return None

def detect_speech_segments(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    mp3_files = [os.path.join(root, file) 
                 for root, _, files in os.walk(input_folder) 
                 for file in files if file.lower().endswith('.mp3')]

    print(f"Found {len(mp3_files)} MP3 files in '{input_folder}'.")

    if not mp3_files:
        print("No MP3 files found. Exiting.")
        return

    # **確保只載入 VAD 模型一次**
    global vad_model
    if "vad_model" not in globals() or not isinstance(vad_model, torch.nn.Module):
        print("Loading Silero VAD model...")
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          onnx=False)

    for mp3_path in tqdm(mp3_files, desc="Processing MP3 files", unit="file"):
        wav_path = mp3_path.replace(".mp3", ".wav")
        wav_path = convert_mp3_to_wav(mp3_path, wav_path)

        if wav_path is None:
            continue  # 跳過轉檔失敗的文件

        try:
            process_single_wav(wav_path, output_folder)
            os.remove(wav_path)  # 只有在成功處理後才刪除
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

def process_single_wav(wav_path, output_folder):
    #"""處理 WAV 檔案，擷取人聲片段並輸出"""
    global vad_model  

    try:
        print(f"Processing {wav_path}...")

        waveform, sample_rate = torchaudio.load(wav_path)

        # 確保是單聲道
        if waveform.shape[0] != 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 確保採樣率為 16kHz
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate} Hz to 16000 Hz...")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        print(f"Audio loaded: {waveform.shape}, Sample Rate: {sample_rate}")

        if "vad_model" not in globals() or not isinstance(vad_model, torch.nn.Module):
            print("Error: vad_model is not available! Skipping this file.")
            return

        # 設定 VAD 分析的 chunk size
        chunk_size = 512  # Silero VAD 需要 512 采樣點 (16kHz)
        speech_timestamps = []
        current_start = None

        # **迭代音訊並進行 VAD**
        for frame_idx in range(0, waveform.shape[1] // chunk_size):
            start_sample = frame_idx * chunk_size
            end_sample = start_sample + chunk_size
            chunk = waveform[:, start_sample:end_sample]

            # 如果 chunk 長度不足 512，補零
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
                    current_start = frame_idx  # **開始記錄人聲**
            else:
                if current_start is not None:
                    end_frame = frame_idx
                    start_sample = current_start * chunk_size
                    end_sample = end_frame * chunk_size

                    if end_sample > start_sample + 512:  # **確保最小片段長度**
                        speech_timestamps.append((start_sample, end_sample))
                    current_start = None

        print(f"Detected {len(speech_timestamps)} speech segments in {wav_path}.")

        if not speech_timestamps:
            print(f"No speech detected in {wav_path}. Skipping.")
            return  

        # **建立輸出資料夾**
        file_name = os.path.splitext(os.path.basename(wav_path))[0]
        file_output_folder = os.path.join(output_folder, file_name)
        os.makedirs(file_output_folder, exist_ok=True)

        # **儲存所有人聲段落**
        for idx, (start_sample, end_sample) in enumerate(speech_timestamps):
            if start_sample >= end_sample:  
                print(f"WARNING: Skipping invalid segment {idx + 1} (start_sample >= end_sample)")
                continue

            segment = waveform[:, start_sample:end_sample]
            save_path = os.path.join(file_output_folder, f"segment_{idx + 1}.wav")
            torchaudio.save(save_path, segment, sample_rate)
            print(f"Saved: {save_path}")

    except Exception as e:
        print(f"Error processing {wav_path}: {e}")


# **範例用法**
input_folder = "/media/meow/Elements/ems_call/data/raw_wav/data_2024all_n3"
output_folder = "/media/meow/Elements/ems_call/data/raw_data_2024all_n3_processed_segments"
detect_speech_segments(input_folder, output_folder)

