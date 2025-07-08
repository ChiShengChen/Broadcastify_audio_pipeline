#!/usr/bin/env python3
"""
OpenAI Whisper Large-v3 English Audio Transcription Script
Used to transcribe audio files from the long_calls_filtered directory to English text.
"""

import os
import whisper
import torch
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_audio_files(source_dir, model_name="large-v3"):
    """
    Transcribes audio files using the Whisper model.
    
    Args:
        source_dir (str): Path to the source directory containing audio files.
        model_name (str): Name of the Whisper model, defaults to "large-v3".
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
    
    # Get the source directory path
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return
    
    # Statistics variables
    total_files = 0
    transcribed_files = 0
    skipped_files = 0
    
    # Iterate over all subdirectories
    for subdir in source_path.iterdir():
        if subdir.is_dir():
            logger.info(f"Processing directory: {subdir.name}")
            
            # Find .wav files in the subdirectory
            wav_files = list(subdir.glob("*.wav"))
            
            for wav_file in wav_files:
                total_files += 1
                
                # Generate transcript filename
                transcript_filename = f"{model_name}_{wav_file.stem}.txt"
                transcript_path = wav_file.parent / transcript_filename
                
                # Check if transcript file already exists
                if transcript_path.exists():
                    logger.info(f"Skipping existing transcript file: {transcript_filename}")
                    skipped_files += 1
                    continue
                
                try:
                    logger.info(f"Transcribing file: {wav_file.name}")
                    
                    # Perform transcription
                    result = model.transcribe(str(wav_file), language="en")
                    
                    # Save the transcription result
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(result["text"])
                    
                    logger.info(f"Transcription complete, saved as: {transcript_filename}")
                    transcribed_files += 1
                    
                except Exception as e:
                    logger.error(f"Error transcribing file {wav_file.name}: {e}")
                    continue
    
    # Output statistics
    logger.info("="*50)
    logger.info("Transcription Statistics:")
    logger.info(f"Total audio files found: {total_files}")
    logger.info(f"Successfully transcribed files: {transcribed_files}")
    logger.info(f"Skipped files: {skipped_files}")
    logger.info("="*50)

def main():
    # Set paths
    source_directory = "/media/meow/One Touch/ems_call/long_calls_filtered"
    model_name = "large-v3"
    
    logger.info("Starting English audio transcription process.")
    logger.info(f"Source directory: {source_directory}")
    logger.info(f"Using model: {model_name}")
    
    # Check if the source directory exists
    if not os.path.exists(source_directory):
        logger.error(f"Source directory does not exist: {source_directory}")
        return
    
    # Start transcription
    transcribe_audio_files(source_directory, model_name)

if __name__ == "__main__":
    main() 