#!/usr/bin/env python3
"""
Whisper transcription script for a single directory.
"""

import os
import whisper
import torch
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_single_folder(folder_path, model_name="large-v3"):
    """
    Transcribes audio files in a single directory.
    """
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load the Whisper model
    logger.info(f"Loading Whisper model: {model_name}...")
    try:
        model = whisper.load_model(model_name, device=device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Get the directory path
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Directory not found: {folder_path}")
        return
    
    # Find all .wav files
    wav_files = list(folder.glob("*.wav"))
    logger.info(f"Found {len(wav_files)} audio files in {folder.name}")
    
    for wav_file in wav_files:
        # Generate transcript filename
        transcript_filename = f"{model_name}_{wav_file.stem}.txt"
        transcript_path = wav_file.parent / transcript_filename
        
        try:
            logger.info(f"Transcribing file: {wav_file.name}")
            
            # Perform transcription (force overwrite of existing files)
            result = model.transcribe(str(wav_file), language="en")
            
            # Save the transcription result
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            logger.info(f"Transcription complete, saved as: {transcript_filename}")
            logger.info(f"Transcription content (first 100 chars): {result['text'][:100]}...")
            
        except Exception as e:
            logger.error(f"Error transcribing file {wav_file.name}: {e}")

if __name__ == "__main__":
    target_folder = "/media/meow/One Touch/ems_call/long_calls_filtered/202412010033-478455-14744"
    
    print("="*60)
    print("Single Directory English Transcription Tool")
    print("="*60)
    print(f"Target Directory: {target_folder}")
    print()
    
    transcribe_single_folder(target_folder) 