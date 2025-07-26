#!/usr/bin/env python3

import os
import torch
import torchaudio
import numpy as np
import tempfile
import time
from pathlib import Path

def create_test_audio(duration_seconds=300, sample_rate=16000):
    """Create a test audio file with specified duration"""
    print(f"Creating test audio file: {duration_seconds}s at {sample_rate}Hz")
    
    # Generate white noise for testing
    num_samples = int(duration_seconds * sample_rate)
    audio_data = np.random.randn(num_samples).astype(np.float32) * 0.1  # Low volume
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Save as WAV
    torchaudio.save(temp_path, torch.tensor(audio_data).unsqueeze(0), sample_rate)
    
    print(f"Test audio saved to: {temp_path}")
    print(f"File size: {os.path.getsize(temp_path) / (1024*1024):.2f} MB")
    
    return temp_path

def test_whisper_long_audio(audio_path):
    """Test Whisper with long audio"""
    print("\n=== Testing Whisper (large-v3) ===")
    
    try:
        import whisper
        
        # Load model
        print("Loading Whisper model...")
        model = whisper.load_model('large-v3')
        
        # Check audio duration
        import librosa
        duration = librosa.get_duration(path=audio_path)
        print(f"Audio duration: {duration:.2f}s")
        
        # Transcribe
        print("Starting transcription...")
        start_time = time.time()
        
        result = model.transcribe(audio_path, language="en")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Transcription completed in {processing_time:.2f}s")
        print(f"Transcription length: {len(result['text'])} characters")
        print(f"Processing speed: {duration/processing_time:.2f}x real-time")
        
        # Check if there are segments
        if 'segments' in result:
            print(f"Number of segments: {len(result['segments'])}")
            for i, segment in enumerate(result['segments'][:3]):  # Show first 3 segments
                print(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
        else:
            print("No segments found in result")
            
        return result
        
    except Exception as e:
        print(f"Error testing Whisper: {e}")
        return None

def test_nemo_long_audio(audio_path, model_name="nvidia/canary-1b"):
    """Test NeMo model with long audio"""
    print(f"\n=== Testing NeMo ({model_name}) ===")
    
    try:
        import nemo.collections.asr as nemo_asr
        
        # Load model
        print("Loading NeMo model...")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        
        # Check audio duration
        import librosa
        duration = librosa.get_duration(path=audio_path)
        print(f"Audio duration: {duration:.2f}s")
        
        # Transcribe
        print("Starting transcription...")
        start_time = time.time()
        
        result = model.transcribe(audio=[audio_path], batch_size=1, return_hypotheses=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Transcription completed in {processing_time:.2f}s")
        
        if result and len(result) > 0:
            hypothesis = result[0]
            text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)
            print(f"Transcription length: {len(text)} characters")
            print(f"Processing speed: {duration/processing_time:.2f}x real-time")
            
            # Check if there are timestamps or segments
            if hasattr(hypothesis, 'timestep') and hypothesis.timestep:
                print(f"Number of timesteps: {len(hypothesis.timestep)}")
            else:
                print("No timesteps found in result")
                
            return result
        else:
            print("No transcription result")
            return None
            
    except Exception as e:
        print(f"Error testing NeMo: {e}")
        return None

def test_transformers_long_audio(audio_path):
    """Test Transformers (Wav2Vec2) with long audio"""
    print("\n=== Testing Transformers (Wav2Vec2) ===")
    
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torchaudio.transforms as T
        
        # Load model
        print("Loading Wav2Vec2 model...")
        model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        
        # Load audio
        print("Loading audio...")
        speech_array, sampling_rate = torchaudio.load(audio_path)
        
        # Check duration
        duration = speech_array.shape[1] / sampling_rate
        print(f"Audio duration: {duration:.2f}s")
        
        # Resample if needed
        if sampling_rate != 16000:
            resampler = T.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech_array = resampler(speech_array)
        
        # Transcribe
        print("Starting transcription...")
        start_time = time.time()
        
        inputs = processor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Transcription completed in {processing_time:.2f}s")
        print(f"Transcription length: {len(transcription)} characters")
        print(f"Processing speed: {duration/processing_time:.2f}x real-time")
        
        # Check input shape to see if it was processed as one chunk
        print(f"Input tensor shape: {inputs.input_values.shape}")
        
        return transcription
        
    except Exception as e:
        print(f"Error testing Transformers: {e}")
        return None

def main():
    """Main test function"""
    print("=== ASR Long Audio Processing Test ===")
    print("This test checks if ASR models automatically split long audio files")
    print()
    
    # Create a 5-minute test audio file
    test_audio_path = create_test_audio(duration_seconds=300)  # 5 minutes
    
    try:
        # Test each model
        whisper_result = test_whisper_long_audio(test_audio_path)
        nemo_result = test_nemo_long_audio(test_audio_path)
        transformers_result = test_transformers_long_audio(test_audio_path)
        
        print("\n=== Summary ===")
        print("Based on the test results:")
        print("- If models process the entire file without errors, they likely handle long audio internally")
        print("- If models fail or show memory issues, they may need external splitting")
        print("- Check the processing speed to see if it's reasonable for the audio length")
        
    finally:
        # Clean up
        if os.path.exists(test_audio_path):
            os.unlink(test_audio_path)
            print(f"\nCleaned up test file: {test_audio_path}")

if __name__ == '__main__':
    main() 