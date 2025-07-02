#!/usr/bin/env python3
"""
專門處理單個目錄的 Whisper 轉錄腳本
"""

import os
import whisper
import torch
from pathlib import Path
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_single_folder(folder_path, model_name="large-v3"):
    """
    轉錄單個目錄中的音頻文件
    """
    
    # 檢查 CUDA 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用設備: {device}")
    
    # 載入 Whisper 模型
    logger.info(f"載入 Whisper {model_name} 模型...")
    try:
        model = whisper.load_model(model_name, device=device)
        logger.info("模型載入成功")
    except Exception as e:
        logger.error(f"載入模型失敗: {e}")
        return
    
    # 取得目錄路徑
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"目錄不存在: {folder_path}")
        return
    
    # 找到所有 .wav 文件
    wav_files = list(folder.glob("*.wav"))
    logger.info(f"在 {folder.name} 中找到 {len(wav_files)} 個音頻文件")
    
    for wav_file in wav_files:
        # 生成轉錄文件名
        transcript_filename = f"{model_name}_{wav_file.stem}.txt"
        transcript_path = wav_file.parent / transcript_filename
        
        try:
            logger.info(f"轉錄文件: {wav_file.name}")
            
            # 執行轉錄（強制覆蓋現有文件）
            result = model.transcribe(str(wav_file), language="en")
            
            # 保存轉錄結果
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            logger.info(f"轉錄完成，保存為: {transcript_filename}")
            logger.info(f"轉錄內容（前100字符）: {result['text'][:100]}...")
            
        except Exception as e:
            logger.error(f"轉錄文件 {wav_file.name} 時發生錯誤: {e}")

if __name__ == "__main__":
    target_folder = "/media/meow/One Touch/ems_call/long_calls_filtered/202412010033-478455-14744"
    
    print("="*60)
    print("單個目錄英文轉錄工具")
    print("="*60)
    print(f"目標目錄: {target_folder}")
    print()
    
    transcribe_single_folder(target_folder) 