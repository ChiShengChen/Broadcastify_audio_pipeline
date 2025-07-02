#!/usr/bin/env python3
"""
Kimi-Audio 英文音頻轉錄腳本
用於將 long_calls_filtered 目錄中的音頻文件轉換為英文文字
"""

import os
import torch
from pathlib import Path
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import librosa

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_audio_files(source_dir, model_id="moonshotai/Kimi-Audio-7B-Instruct"):
    """
    使用 Kimi-Audio 模型轉錄音頻文件
    
    Args:
        source_dir (str): 包含音頻文件的源目錄路徑
        model_id (str): Hugging Face上的模型ID
    """
    
    # 檢查 CUDA 是否可用
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"使用設備: {device}, 資料類型: {torch_dtype}")
    
    # 載入 Kimi-Audio 模型和處理器
    logger.info(f"載入 Kimi-Audio 模型: {model_id}...")
    try:
        # 由於此模型的自訂程式碼將處理功能整合到 Tokenizer 中，
        # 我們直接使用 AutoTokenizer 來載入。
        processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="sdpa" # a more efficient attention mechanism
        )
        model.to(device)
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
    for subdir in sorted(source_path.iterdir()):
        if subdir.is_dir():
            logger.info(f"處理目錄: {subdir.name}")
            
            # 在子目錄中尋找 .wav 文件
            wav_files = list(subdir.glob("*.wav"))
            
            for wav_file in sorted(wav_files):
                total_files += 1
                
                # 生成轉錄文件名
                transcript_filename = f"{model_id.split('/')[-1]}_{wav_file.stem}.txt"
                transcript_path = wav_file.parent / transcript_filename
                
                # 檢查轉錄文件是否已存在
                if transcript_path.exists():
                    logger.info(f"跳過已存在的轉錄文件: {transcript_filename}")
                    skipped_files += 1
                    continue
                
                try:
                    logger.info(f"轉錄文件: {wav_file.name}")

                    # 讀取音頻文件
                    audio_array, sampling_rate = librosa.load(str(wav_file), sr=16000)

                    # 準備模型輸入
                    messages = [
                        {"role": "user", "content": "<audio>"},
                    ]
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(prompt, audios=audio_array, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device=device, dtype=torch_dtype if k == 'audio_values' else 'auto') for k, v in inputs.items()}
                    
                    # 執行轉錄
                    generate_ids = model.generate(**inputs, max_new_tokens=1024)
                    
                    # 解碼並移除輸入部分
                    outputs = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
                    transcription = outputs.split("ASSISTANT: ")[-1].strip()

                    # 保存轉錄結果
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(transcription)
                    
                    logger.info(f"轉錄完成，保存為: {transcript_filename}")
                    transcribed_files += 1
                    
                except Exception as e:
                    logger.error(f"轉錄文件 {wav_file.name} 時發生錯誤: {e}")
                    continue
    
    # 清理模型
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("模型已從記憶體中卸載")

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
    model_id = "moonshotai/Kimi-Audio-7B-Instruct"
    
    logger.info("開始 Kimi-Audio 英文音頻轉錄程序")
    logger.info(f"源目錄: {source_directory}")
    logger.info(f"使用模型: {model_id}")
    
    # 檢查源目錄是否存在
    if not os.path.exists(source_directory):
        logger.error(f"源目錄不存在: {source_directory}")
        return
    
    # 開始轉錄
    transcribe_audio_files(source_directory, model_id)

if __name__ == "__main__":
    main() 