#!/usr/bin/env python3
"""
Meta Wav2Vec2 English Audio Transcription Script
Used to transcribe audio files in the long_calls_filtered directory into text
"""

import os
import torch
import torchaudio
import torchaudio.transforms as T
import argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def transcribe_audio(model_name, audio_path, hf_token=None):
    """
    Transcribe an audio file using a specified Wav2Vec 2.0 model.
    
    Args:
        model_name (str): The name of the model on Hugging Face Hub or a local path.
        audio_path (str): Path to the audio file.
        hf_token (str, optional): Hugging Face Hub token for private models.
    """
    # Load model and processor
    print(f"Loading model: {model_name}...")
    try:
        # If a local path is provided and exists, load from there
        if os.path.isdir(model_name):
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            processor = Wav2Vec2Processor.from_pretrained(model_name)
        else:
            # Otherwise, load from Hugging Face Hub
            model = Wav2Vec2ForCTC.from_pretrained(model_name, use_auth_token=hf_token)
            processor = Wav2Vec2Processor.from_pretrained(model_name, use_auth_token=hf_token)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded successfully on {device}.")

    # Load and resample audio file
    try:
        print(f"Loading audio file: {audio_path}...")
        speech_array, sampling_rate = torchaudio.load(audio_path)
        
        # Resample if sampling rate is not 16kHz
        if sampling_rate != 16000:
            resampler = T.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech_array = resampler(speech_array)
            sampling_rate = 16000

    except Exception as e:
        print(f"Error loading or resampling audio file {audio_path}: {e}")
        return None

    try:
        print("Processing audio...")
        input_values = processor(speech_array.squeeze(), sampling_rate=sampling_rate, return_tensors="pt").input_values
        
        # Move input to GPU if available
        input_values = input_values.to(device)

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode to get predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode token IDs to text
        transcription = processor.decode(predicted_ids[0])
        
        return transcription.strip()
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def main(args):
    """Main function to handle argument parsing and transcription."""
    
    # Perform transcription
    transcribed_text = transcribe_audio(args.model, args.audio_file, args.hf_token)
    
    if transcribed_text is not None:
        print("\n--- Transcription Result ---")
        print(transcribed_text)
        print("--------------------------\n")
        
        # Determine the output filename
        if args.output_file:
            output_path = args.output_file
        else:
            # Auto-generate output filename in the format: <model_name>_<original_filename>.txt
            audio_basename = os.path.splitext(os.path.basename(args.audio_file))[0]
            model_basename = args.model.split('/')[-1] # Get model name from path
            output_filename = f"{model_basename}_{audio_basename}.txt"
            output_path = os.path.join(os.getcwd(), output_filename)
        
        # Write transcription result to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcribed_text)
            print(f"Transcription successfully saved to: {output_path}")
        except IOError as e:
            print(f"Error writing to output file {output_path}: {e}")

if __name__ == '__main__':
    # Setup ArgumentParser
    parser = argparse.ArgumentParser(description="Transcribe an audio file using a specified Wav2Vec 2.0 model.")
    parser.add_argument("model", type=str, help="Name of the Wav2Vec 2.0 model from Hugging Face Hub or a local path.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file to be transcribed.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional. Path to save the transcription text file. If not provided, it will be saved in the current directory.")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional. Hugging Face Hub token for models that require authentication.")
    
    args = parser.parse_args()
    main(args)