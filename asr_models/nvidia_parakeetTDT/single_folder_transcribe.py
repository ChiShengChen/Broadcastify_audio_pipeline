#!/usr/bin/env python3
"""
NVIDIA Parakeet TDT transcription script for a single directory.
"""

import os
import torch
import librosa
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_single_folder(folder_path, model_name="parakeet-tdt-0.6b-v2"):
    """
    Transcribes audio files in a single directory.
    """
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load Parakeet TDT model
    logger.info(f"Loading NVIDIA Parakeet TDT model...")
    try:
        # First, try the NeMo method
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.EncDecCTCModel.from_pretrained("nvidia/parakeet-ctc-0.6b")
        logger.info("Successfully loaded model using NeMo.")
        use_nemo = True
    except Exception as e:
        logger.error(f"NeMo method failed: {e}")
        try:
            # Try the Hugging Face method
            from transformers import AutoProcessor, AutoModelForCTC
            processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-0.6b")
            model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-0.6b")
            model.to(device)
            model.eval()
            logger.info("Successfully loaded model using Hugging Face.")
            use_nemo = False
        except Exception as e2:
            logger.error(f"Both methods failed: {e2}")
            return
    
    # Get directory path
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
            
            # Load audio file
            audio, sample_rate = librosa.load(str(wav_file), sr=16000)
            
            # Perform transcription (force overwrite of existing files)
            if use_nemo:
                # Use NeMo method
                transcription = model.transcribe([str(wav_file)])[0]
                if hasattr(transcription, 'text'):
                    result = transcription.text
                else:
                    result = transcription
            else:
                # Use Hugging Face method
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                
                predicted_ids = torch.argmax(logits, dim=-1)
                result = processor.batch_decode(predicted_ids)[0]
            
            # Save transcription result
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            logger.info(f"Transcription complete, saved as: {transcript_filename}")
            logger.info(f"Transcription content (first 100 chars): {result[:100]}...")
            
        except Exception as e:
            logger.error(f"Error transcribing file {wav_file.name}: {e}")

if __name__ == "__main__":
    target_folder = "/media/meow/One Touch/ems_call/long_calls_filtered/202412010033-478455-14744"
    
    print("="*60)
    print("Single Directory NVIDIA Parakeet TDT Transcription Tool")
    print("="*60)
    print(f"Target Directory: {target_folder}")
    print()
    
    transcribe_single_folder(target_folder) 