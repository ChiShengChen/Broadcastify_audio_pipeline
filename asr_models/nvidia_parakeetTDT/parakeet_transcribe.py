#!/usr/bin/env python3
"""
NVIDIA Parakeet TDT 0.6B v2 English Audio Transcription Script
Used to transcribe audio files in the long_calls_filtered directory into English text.
"""

import os
import torch
import librosa
import numpy as np
from pathlib import Path
import logging
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import warnings

# Ignore some warning messages
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_audio_files(source_dir, model_name="parakeet-tdt-0.6b-v2"):
    """
    Transcribes audio files using the NVIDIA Parakeet TDT 0.6B v2 model.
    
    Args:
        source_dir (str): Path to the source directory containing audio files.
        model_name (str): Model name, defaults to "parakeet-tdt-0.6b-v2".
    """
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load Parakeet TDT model and processor
    model_id = "nvidia/parakeet-ctc-0.6b"  # NVIDIA Parakeet CTC model
    logger.info(f"Loading NVIDIA Parakeet TDT English model: {model_id}")
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model, trying with NeMo: {e}")
        # Try loading with NeMo
        try:
            import nemo.collections.asr as nemo_asr
            model = nemo_asr.models.EncDecCTCModel.from_pretrained("nvidia/parakeet-ctc-0.6b")
            processor = None
            logger.info("Successfully loaded model using NeMo.")
        except Exception as e2:
            logger.error(f"NeMo method also failed: {e2}")
            return
    
    # Get source directory path
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
                    
                    # Load audio file
                    audio, sample_rate = librosa.load(str(wav_file), sr=16000)
                    
                    # Split audio into smaller chunks to avoid memory issues
                    chunk_length = 16000 * 30  # 30-second chunks
                    chunks = []
                    
                    for i in range(0, len(audio), chunk_length):
                        chunk = audio[i:i + chunk_length]
                        if len(chunk) > 0:
                            chunks.append(chunk)
                    
                    # Process each chunk and combine the results
                    transcripts = []
                    
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                        
                        try:
                            if processor is not None:
                                # Using HuggingFace model
                                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                                inputs = {k: v.to(device) for k, v in inputs.items()}
                                with torch.no_grad():
                                    outputs = model.generate(**inputs)
                                transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                            else:
                                # Using NeMo model
                                result = model.transcribe([chunk])[0]
                                transcription = result.text if hasattr(result, "text") else str(result)

                            if transcription.strip():
                                transcripts.append(transcription.strip())
                                
                        except Exception as chunk_error:
                            logger.warning(f"Error processing chunk {i+1}: {chunk_error}")
                            continue
                    
                    # Combine transcripts from all chunks
                    final_transcript = " ".join(transcripts)
                    
                    if not final_transcript.strip():
                        logger.warning(f"Audio file {wav_file.name} did not produce a valid transcript.")
                        continue
                    
                    # Save the transcription result
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(final_transcript)
                    
                    logger.info(f"Transcription complete, saved as: {transcript_filename}")
                    logger.info(f"Transcription content (first 100 chars): {final_transcript[:100]}...")
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
    model_name = "parakeet-tdt-0.6b-v2"
    
    logger.info("Starting NVIDIA Parakeet TDT English audio transcription process")
    logger.info(f"Source directory: {source_directory}")
    logger.info(f"Using model: {model_name}")
    
    # Check if source directory exists
    if not os.path.exists(source_directory):
        logger.error(f"Source directory not found: {source_directory}")
        return
    
    # Start transcription
    transcribe_audio_files(source_directory, model_name)

if __name__ == "__main__":
    main() 