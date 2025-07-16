import os
import glob
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from tqdm import tqdm

# --- Configuration ---
# 輸入資料夾：包含您原始音檔的路徑
# INPUT_DIR = '/media/meow/One Touch/ems_call/random_samples_1'
INPUT_DIR = '/media/meow/One Touch/ems_call/random_samples_2'

# 輸出資料夾：儲存處理後音檔的路徑
# OUTPUT_DIR = '/media/meow/One Touch/ems_call/random_samples_1_preprocessed'
OUTPUT_DIR = '/media/meow/One Touch/ems_call/random_samples_2_preprocessed'
# 目標取樣率 (ASR 模型常用 16000 Hz)
TARGET_SR = 16000

# 支援的音檔格式
SUPPORTED_EXTENSIONS = ['*.wav', '*.mp3', '*.flac', '*.m4a']
# ---------------------

def preprocess_audio(input_path, output_path, target_sr):
    """
    對單一音檔進行前處理：
    1. 載入音檔並重新取樣至 target_sr
    2. 執行降噪
    3. 音量標準化
    4. 存成 WAV 格式
    """
    try:
        # 1. 載入音檔並重新取樣
        # librosa.load 會自動將音檔轉為單聲道與浮點數格式
        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

        # 2. 執行降噪
        # 我們只對非靜音部分進行降噪，效果較好
        reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr)

        # 3. 音量標準化 (Peak normalization)
        # 將音訊的最大絕對值調整為 1.0
        normalized_audio = librosa.util.normalize(reduced_noise_audio)

        # 4. 儲存處理後的音檔
        sf.write(output_path, normalized_audio, target_sr, 'PCM_16')

    except Exception as e:
        print(f"  - Error processing {os.path.basename(input_path)}: {e}")

def main():
    """主函式，遍歷資料夾並處理所有音檔。"""
    print(f"開始進行音檔前處理...")
    print(f"輸入資料夾: {INPUT_DIR}")
    print(f"輸出資料夾: {OUTPUT_DIR}")
    print(f"目標取樣率: {TARGET_SR} Hz")

    if not os.path.exists(INPUT_DIR):
        print(f"錯誤：輸入資料夾 {INPUT_DIR} 不存在。")
        return

    # 建立輸出資料夾
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 取得所有支援的音檔
    audio_files = []
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not audio_files:
        print("在輸入資料夾中找不到任何支援的音檔。")
        return

    print(f"找到 {len(audio_files)} 個音檔，開始處理...")

    # 使用 tqdm 顯示進度條
    for file_path in tqdm(audio_files, desc="Preprocessing audio files"):
        basename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(basename)[0]
        output_filename = f"{filename_no_ext}.wav" # 統一輸出為 .wav
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        preprocess_audio(file_path, output_path, TARGET_SR)

    print("\n所有音檔處理完成！")
    print(f"處理後的檔案已儲存至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main() 