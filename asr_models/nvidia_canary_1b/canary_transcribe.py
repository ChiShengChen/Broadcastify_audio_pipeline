#!/usr/bin/env python3
"""
NVIDIA Canary-1B 英文音頻轉錄腳本
用於將 long_calls_filtered 目錄中的音頻文件轉換為英文文字
"""

import os
import torch
from pathlib import Path
import logging
from nemo.collections.asr.models import EncDecMultiTaskModel
import json

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_audio_files(source_dir, model_name="nvidia/canary-1b"):
    """
    使用 NVIDIA Canary 模型轉錄音頻文件
    
    Args:
        source_dir (str): 包含音頻文件的源目錄路徑
        model_name (str): Canary 模型名稱，預設為 "nvidia/canary-1b"
    """
    
    # 檢查 CUDA 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用設備: {device}")
    
    # 載入 Canary 模型
    logger.info(f"載入 Canary {model_name} 模型...")
    try:
        canary_model = EncDecMultiTaskModel.from_pretrained(model_name)
        canary_model.to(device)
        
        # 更新解碼參數
        decode_cfg = canary_model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        canary_model.change_decoding_strategy(decode_cfg)
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
            
            # 尋找所有 .wav 文件
            wav_files_to_process = []
            file_paths_map = {}

            for wav_file in subdir.glob("*.wav"):
                total_files += 1
                model_identifier = model_name.split('/')[-1]
                transcript_filename = f"{model_identifier}_{wav_file.stem}.txt"
                transcript_path = wav_file.parent / transcript_filename
                
                if transcript_path.exists():
                    # 如果文件存在但為空，則重新轉錄
                    if transcript_path.stat().st_size == 0:
                        logger.info(f"發現空的轉錄文件，將重新轉錄: {transcript_filename}")
                    else:
                        logger.info(f"跳過已存在的轉錄文件: {transcript_filename}")
                        skipped_files += 1
                        continue
                
                wav_files_to_process.append(str(wav_file))
                file_paths_map[str(wav_file)] = transcript_path

            if not wav_files_to_process:
                logger.info(f"目錄 {subdir.name} 中沒有需要處理的新文件。")
                continue

            try:
                logger.info(f"開始轉錄目錄 {subdir.name} 中的 {len(wav_files_to_process)} 個文件...")
                # 執行批次轉錄
                transcriptions = canary_model.transcribe(
                    audio=wav_files_to_process,
                    batch_size=16,
                    return_hypotheses=True,  # 強制回傳 Hypothesis 物件
                )

                # 處理回傳的 Hypothesis 物件
                for i, wav_path_str in enumerate(wav_files_to_process):
                    transcript_path = file_paths_map[wav_path_str]
                    try:
                        # 從 Hypothesis 物件中提取文字
                        transcription_text = transcriptions[i].text
                        
                        # 保存轉錄結果
                        with open(transcript_path, 'w', encoding='utf-8') as f:
                            f.write(transcription_text)
                        
                        logger.info(f"轉錄完成，保存為: {transcript_path.name}")
                        transcribed_files += 1
                    except Exception as e:
                        logger.error(f"保存文件 {transcript_path.name} 時發生錯誤: {e}")

            except Exception as e:
                logger.error(f"轉錄目錄 {subdir.name} 時發生錯誤: {e}")
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
    model_name = "nvidia/canary-1b"
    
    logger.info("開始英文音頻轉錄程序 (NVIDIA Canary)")
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