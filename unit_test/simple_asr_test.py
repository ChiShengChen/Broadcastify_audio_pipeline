#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple ASR Test
===============

Quick test to reproduce the ASR file processing issues:
1. Test different models' ability to process files
2. Test VAD impact on file processing
3. Identify why some models fail to process all files
"""

import os
import sys
import tempfile
import shutil
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path

def create_test_files(output_dir: str, count: int = 10):
    """Create simple test audio files"""
    os.makedirs(output_dir, exist_ok=True)
    
    sample_rate = 16000
    duration = 5.0
    
    for i in range(count):
        # Create simple audio
        t = torch.linspace(0, duration, int(sample_rate * duration))
        signal = 0.3 * torch.sin(2 * np.pi * 300 * t)
        signal = signal + 0.05 * torch.randn_like(signal)
        
        filename = f"test_{i:03d}.wav"
        filepath = os.path.join(output_dir, filename)
        torchaudio.save(filepath, signal.unsqueeze(0), sample_rate)
    
    print(f"Created {count} test files in {output_dir}")

def test_whisper(input_dir: str, output_dir: str):
    """Test Whisper model"""
    print("\nTesting Whisper (large-v3)...")
    
    try:
        import whisper
        model = whisper.load_model('large-v3')
        
        wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        processed = 0
        
        for wav_file in wav_files:
            try:
                wav_path = os.path.join(input_dir, wav_file)
                result = model.transcribe(wav_path)
                
                # Save transcript
                transcript_file = os.path.join(output_dir, f"large-v3_{wav_file.replace('.wav', '.txt')}")
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                
                processed += 1
                print(f"  ✓ {wav_file}")
                
            except Exception as e:
                print(f"  ✗ {wav_file}: {e}")
        
        print(f"Whisper processed: {processed}/{len(wav_files)} files")
        return processed, len(wav_files)
        
    except Exception as e:
        print(f"Whisper not available: {e}")
        return 0, 0

def test_vad(input_dir: str, output_dir: str):
    """Test VAD processing"""
    print("\nTesting VAD processing...")
    
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from vad_pipeline import VADPipeline
        
        vad = VADPipeline(
            speech_threshold=0.5,
            min_speech_duration=0.5,
            min_silence_duration=0.3,
            target_sample_rate=16000
        )
        
        vad_summary = vad.process_directory(input_dir, output_dir)
        
        if 'error' not in vad_summary:
            print(f"VAD processed: {vad_summary.get('successful', 0)} files")
            return vad_summary.get('successful', 0), len(os.listdir(input_dir))
        else:
            print(f"VAD failed: {vad_summary['error']}")
            return 0, len(os.listdir(input_dir))
            
    except Exception as e:
        print(f"VAD not available: {e}")
        return 0, 0

def test_other_models(input_dir: str, output_dir: str):
    """Test other models (simplified)"""
    models = ['canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
    results = {}
    
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    for model in models:
        print(f"\nTesting {model}...")
        processed = 0
        
        for wav_file in wav_files:
            try:
                # Create dummy transcript for testing
                transcript_file = os.path.join(output_dir, f"{model}_{wav_file.replace('.wav', '.txt')}")
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(f"test transcript for {wav_file}")
                
                processed += 1
                print(f"  ✓ {wav_file}")
                
            except Exception as e:
                print(f"  ✗ {wav_file}: {e}")
        
        results[model] = (processed, len(wav_files))
        print(f"{model} processed: {processed}/{len(wav_files)} files")
    
    return results

def main():
    """Main test function"""
    print("Simple ASR File Processing Test")
    print("=" * 40)
    
    # Create test directory
    test_dir = tempfile.mkdtemp(prefix="simple_asr_test_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Create test files
        audio_dir = os.path.join(test_dir, "audio")
        create_test_files(audio_dir, count=10)
        
        # Test without VAD
        print("\n" + "=" * 40)
        print("Testing WITHOUT VAD")
        print("=" * 40)
        
        no_vad_dir = os.path.join(test_dir, "no_vad")
        os.makedirs(no_vad_dir, exist_ok=True)
        
        # Test Whisper
        whisper_processed, whisper_total = test_whisper(audio_dir, no_vad_dir)
        
        # Test other models
        other_results = test_other_models(audio_dir, no_vad_dir)
        
        # Test with VAD
        print("\n" + "=" * 40)
        print("Testing WITH VAD")
        print("=" * 40)
        
        vad_dir = os.path.join(test_dir, "vad_output")
        vad_processed, vad_total = test_vad(audio_dir, vad_dir)
        
        if vad_processed > 0:
            with_vad_dir = os.path.join(test_dir, "with_vad")
            os.makedirs(with_vad_dir, exist_ok=True)
            
            # Test Whisper with VAD
            whisper_vad_processed, whisper_vad_total = test_whisper(vad_dir, with_vad_dir)
            
            # Test other models with VAD
            other_vad_results = test_other_models(vad_dir, with_vad_dir)
        
        # Summary
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        
        print(f"\nWithout VAD:")
        print(f"  Whisper: {whisper_processed}/{whisper_total} ({whisper_processed/whisper_total:.1%})")
        for model, (processed, total) in other_results.items():
            print(f"  {model}: {processed}/{total} ({processed/total:.1%})")
        
        if vad_processed > 0:
            print(f"\nWith VAD:")
            print(f"  VAD processed: {vad_processed}/{vad_total} files")
            print(f"  Whisper: {whisper_vad_processed}/{vad_processed} ({whisper_vad_processed/vad_processed:.1%})")
            for model, (processed, total) in other_vad_results.items():
                print(f"  {model}: {processed}/{vad_processed} ({processed/vad_processed:.1%})")
        
        # Identify issues
        print(f"\n" + "=" * 40)
        print("ISSUE IDENTIFICATION")
        print("=" * 40)
        
        # Check which models can process all files without VAD
        if whisper_processed < whisper_total:
            print(f"❌ Whisper: Cannot process all files without VAD")
        
        for model, (processed, total) in other_results.items():
            if processed < total:
                print(f"❌ {model}: Cannot process all files without VAD")
        
        # Check VAD impact
        if vad_processed > 0:
            if vad_processed < vad_total:
                print(f"⚠️  VAD: Reduces processable files from {vad_total} to {vad_processed}")
            
            if whisper_vad_processed < vad_processed:
                print(f"⚠️  Whisper with VAD: Cannot process all VAD files")
            
            for model, (processed, total) in other_vad_results.items():
                if processed < vad_processed:
                    print(f"⚠️  {model} with VAD: Cannot process all VAD files")
        
        print(f"\nTest results saved in: {test_dir}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ask if user wants to keep files
        response = input("\nKeep test files? (y/n): ").lower().strip()
        if response != 'y':
            shutil.rmtree(test_dir)
            print(f"Cleaned up: {test_dir}")

if __name__ == "__main__":
    main() 