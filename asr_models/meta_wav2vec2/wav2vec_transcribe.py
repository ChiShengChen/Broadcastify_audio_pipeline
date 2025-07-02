#!/usr/bin/env python3
"""
Meta Wav2Vec2 英文音頻轉錄腳本
用於將 long_calls_filtered 目錄中的音頻文件轉換為文字
"""

import os
import torch
import librosa
import numpy as np
from pathlib import Path
import logging
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings

# 忽略一些警告信息
warnings.filterwarnings("ignore")

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_audio_files(source_dir, model_name="wav2vec-xls-r"):
    """
    使用 Wav2Vec2 英文模型轉錄音頻文件
    
    Args:
        source_dir (str): 包含音頻文件的源目錄路徑
        model_name (str): 模型名稱，預設為 "wav2vec-xls-r"
    """
    
    # 檢查 CUDA 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用設備: {device}")
    
    # 載入 Wav2Vec2 模型和處理器
    model_id = "facebook/wav2vec2-large-960h-lv60-self"  # 英文模型
    logger.info(f"載入 Wav2Vec2 英文模型: {model_id}")
    
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        model.to(device)
        model.eval()
        logger.info("模型載入成功")
    except Exception as e:
        logger.error(f"載入模型失敗: {e}")
        return
    
    # 取得源目錄路徑
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"源目錄不存在: {source_dir}")
        return
    
    # 統計變數
    total_files = 0
    transcribed_files = 0
    skipped_files = 0
    
    # 遍歷所有子目錄
    for subdir in source_path.iterdir():
        if subdir.is_dir():
            logger.info(f"處理目錄: {subdir.name}")
            
            # 在子目錄中尋找 .wav 文件
            wav_files = list(subdir.glob("*.wav"))
            
            for wav_file in wav_files:
                total_files += 1
                
                # 生成轉錄文件名
                transcript_filename = f"{model_name}_{wav_file.stem}.txt"
                transcript_path = wav_file.parent / transcript_filename
                
                # 檢查轉錄文件是否已存在
                if transcript_path.exists():
                    logger.info(f"跳過已存在的轉錄文件: {transcript_filename}")
                    skipped_files += 1
                    continue
                
                try:
                    logger.info(f"轉錄文件: {wav_file.name}")
                    
                    # 讀取音頻文件
                    audio, sample_rate = librosa.load(str(wav_file), sr=16000)
                    
                    # 將音頻分割成較小的片段以避免記憶體問題
                    chunk_length = 16000 * 30  # 30秒片段
                    chunks = []
                    
                    for i in range(0, len(audio), chunk_length):
                        chunk = audio[i:i + chunk_length]
                        if len(chunk) > 0:
                            chunks.append(chunk)
                    
                    # 處理每個片段並合併結果
                    transcripts = []
                    
                    for i, chunk in enumerate(chunks):
                        logger.info(f"處理片段 {i+1}/{len(chunks)}")
                        
                        # 預處理音頻
                        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # 執行推理
                        with torch.no_grad():
                            logits = model(**inputs).logits
                        
                        # 解碼結果
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = processor.batch_decode(predicted_ids)[0]
                        
                        if transcription.strip():
                            transcripts.append(transcription.strip())
                    
                    # 合併所有片段的轉錄結果
                    final_transcript = " ".join(transcripts)
                    
                    # 保存轉錄結果
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(final_transcript)
                    
                    logger.info(f"轉錄完成，保存為: {transcript_filename}")
                    transcribed_files += 1
                    
                except Exception as e:
                    logger.error(f"轉錄文件 {wav_file.name} 時發生錯誤: {e}")
                    continue
    
    # 輸出統計結果
    logger.info("="*50)
    logger.info("轉錄完成統計:")
    logger.info(f"總共找到音頻文件: {total_files}")
    logger.info(f"成功轉錄文件: {transcribed_files}")
    logger.info(f"跳過的文件: {skipped_files}")
    logger.info("="*50)

def main():
    # 設定路徑
    source_directory = "/media/meow/One Touch/ems_call/long_calls_filtered"
    model_name = "wav2vec-xls-r"
    
    logger.info("開始 Wav2Vec2 英文音頻轉錄程序")
    logger.info(f"源目錄: {source_directory}")
    logger.info(f"使用模型: {model_name}")
    
    # 檢查源目錄是否存在
    if not os.path.exists(source_directory):
        logger.error(f"源目錄不存在: {source_directory}")
        return
    
    # 開始轉錄
    transcribe_audio_files(source_directory, model_name)

if __name__ == "__main__":
    main() 