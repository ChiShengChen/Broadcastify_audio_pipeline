#!/usr/bin/env python3
"""
專門處理單個目錄的 NVIDIA Parakeet TDT 轉錄腳本
"""

import os
import torch
import librosa
from pathlib import Path
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_single_folder(folder_path, model_name="parakeet-tdt-0.6b-v2"):
    """
    轉錄單個目錄中的音頻文件
    """
    
    # 檢查 CUDA 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用設備: {device}")
    
    # 載入 Parakeet TDT 模型
    logger.info(f"載入 NVIDIA Parakeet TDT 模型...")
    try:
        # 首先嘗試 NeMo 方式
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.EncDecCTCModel.from_pretrained("nvidia/parakeet-ctc-0.6b")
        logger.info("使用 NeMo 方式載入模型成功")
        use_nemo = True
    except Exception as e:
        logger.error(f"NeMo 方式載入失敗: {e}")
        try:
            # 嘗試 Hugging Face 方式
            from transformers import AutoProcessor, AutoModelForCTC
            processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-0.6b")
            model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-0.6b")
            model.to(device)
            model.eval()
            logger.info("使用 Hugging Face 方式載入模型成功")
            use_nemo = False
        except Exception as e2:
            logger.error(f"兩種方式都失敗: {e2}")
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
            
            # 讀取音頻文件
            audio, sample_rate = librosa.load(str(wav_file), sr=16000)
            
            # 執行轉錄（強制覆蓋現有文件）
            if use_nemo:
                # 使用 NeMo 方式
                transcription = model.transcribe([str(wav_file)])[0]
                if hasattr(transcription, 'text'):
                    result = transcription.text
                else:
                    result = transcription
            else:
                # 使用 Hugging Face 方式
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                
                predicted_ids = torch.argmax(logits, dim=-1)
                result = processor.batch_decode(predicted_ids)[0]
            
            # 保存轉錄結果
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            logger.info(f"轉錄完成，保存為: {transcript_filename}")
            logger.info(f"轉錄內容（前100字符）: {result[:100]}...")
            
        except Exception as e:
            logger.error(f"轉錄文件 {wav_file.name} 時發生錯誤: {e}")

if __name__ == "__main__":
    target_folder = "/media/meow/One Touch/ems_call/long_calls_filtered/202412010033-478455-14744"
    
    print("="*60)
    print("單個目錄 NVIDIA Parakeet TDT 轉錄工具")
    print("="*60)
    print(f"目標目錄: {target_folder}")
    print()
    
    transcribe_single_folder(target_folder) 