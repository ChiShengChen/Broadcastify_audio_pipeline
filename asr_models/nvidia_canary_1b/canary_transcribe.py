#!/usr/bin/env python3
"""
NVIDIA Canary-1B English Audio Transcription Script
Used to transcribe audio files from the long_calls_filtered directory into English text.
"""

import os
import torch
from pathlib import Path
import logging
from nemo.collections.asr.models import EncDecMultiTaskModel
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_audio_files(source_dir, model_name="nvidia/canary-1b"):
    """
    Transcribes audio files using the NVIDIA Canary model.
    
    Args:
        source_dir (str): The path to the source directory containing audio files.
        model_name (str): The Canary model name, default is "nvidia/canary-1b".
    """
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load the Canary model
    logger.info(f"Loading Canary model: {model_name}...")
    try:
        canary_model = EncDecMultiTaskModel.from_pretrained(model_name)
        canary_model.to(device)
        
        # Update decoding parameters
        decode_cfg = canary_model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        canary_model.change_decoding_strategy(decode_cfg)
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
    
    # Iterate through all subdirectories
    for subdir in source_path.iterdir():
        if subdir.is_dir():
            logger.info(f"Processing directory: {subdir.name}")
            
            # Find all .wav files to process
            wav_files_to_process = []
            file_paths_map = {}

            for wav_file in subdir.glob("*.wav"):
                total_files += 1
                model_identifier = model_name.split('/')[-1]
                transcript_filename = f"{model_identifier}_{wav_file.stem}.txt"
                transcript_path = wav_file.parent / transcript_filename
                
                if transcript_path.exists():
                    # If the file exists but is empty, re-transcribe it
                    if transcript_path.stat().st_size == 0:
                        logger.info(f"Found an empty transcript file, will re-transcribe: {transcript_filename}")
                    else:
                        logger.info(f"Skipping existing transcript file: {transcript_filename}")
                        skipped_files += 1
                        continue
                
                wav_files_to_process.append(str(wav_file))
                file_paths_map[str(wav_file)] = transcript_path

            if not wav_files_to_process:
                logger.info(f"No new files to process in directory {subdir.name}.")
                continue

            try:
                logger.info(f"Starting transcription for {len(wav_files_to_process)} files in directory {subdir.name}...")
                # Perform batch transcription
                transcriptions = canary_model.transcribe(
                    audio=wav_files_to_process,
                    batch_size=16,
                    return_hypotheses=True,  # Force return of Hypothesis objects
                )

                # Process the returned Hypothesis objects
                for i, wav_path_str in enumerate(wav_files_to_process):
                    transcript_path = file_paths_map[wav_path_str]
                    try:
                        # Extract text from the Hypothesis object
                        transcription_text = transcriptions[i].text
                        
                        # Save the transcription result
                        with open(transcript_path, 'w', encoding='utf-8') as f:
                            f.write(transcription_text)
                        
                        logger.info(f"Transcription complete, saved as: {transcript_path.name}")
                        transcribed_files += 1
                    except Exception as e:
                        logger.error(f"Error saving file {transcript_path.name}: {e}")

            except Exception as e:
                logger.error(f"Error during transcription for directory {subdir.name}: {e}")
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
    model_name = "nvidia/canary-1b"
    
    logger.info("Starting English audio transcription process (NVIDIA Canary)")
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