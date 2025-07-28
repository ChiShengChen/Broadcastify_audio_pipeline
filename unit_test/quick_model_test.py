#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Model Test
===============

Quick test to reproduce model differences using shorter audio files.
This script creates various short audio files to test model compatibility.
"""

import os
import sys
import tempfile
import shutil
import json
import time
import numpy as np
import torch
import torchaudio
from pathlib import Path

def create_quick_test_files(output_dir: str):
    """Create quick test files with various characteristics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Quick test scenarios (all short duration)
    test_cases = [
        # Duration variations
        {
            'name': 'very_short_0.5s',
            'duration': 0.5,
            'description': 'Very short audio (0.5s)'
        },
        {
            'name': 'short_5s',
            'duration': 5.0,
            'description': 'Short audio (5s)'
        },
        {
            'name': 'normal_30s',
            'duration': 30.0,
            'description': 'Normal audio (30s)'
        },
        
        # Volume variations
        {
            'name': 'low_volume',
            'duration': 10.0,
            'amplitude': 0.05,
            'description': 'Low volume audio'
        },
        {
            'name': 'high_volume',
            'duration': 10.0,
            'amplitude': 0.8,
            'description': 'High volume audio'
        },
        
        # Content variations
        {
            'name': 'silence_only',
            'duration': 10.0,
            'silence_only': True,
            'description': 'Silence only'
        },
        {
            'name': 'noise_only',
            'duration': 10.0,
            'noise_only': True,
            'description': 'Noise only'
        },
        {
            'name': 'speech_like',
            'duration': 10.0,
            'description': 'Speech-like audio'
        },
        
        # Format variations
        {
            'name': 'stereo_audio',
            'duration': 10.0,
            'channels': 2,
            'description': 'Stereo audio'
        },
        {
            'name': 'low_sample_rate',
            'duration': 10.0,
            'sample_rate': 8000,
            'description': 'Low sample rate (8kHz)'
        }
    ]
    
    created_files = []
    
    for test_case in test_cases:
        print(f"Creating {test_case['name']}: {test_case['description']}")
        
        try:
            file_path = create_quick_audio(test_case, output_dir)
            created_files.append({
                'path': file_path,
                'metadata': test_case
            })
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    return created_files

def create_quick_audio(test_case, output_dir):
    """Create a quick test audio file"""
    name = test_case['name']
    duration = test_case['duration']
    sample_rate = test_case.get('sample_rate', 16000)
    channels = test_case.get('channels', 1)
    
    # Create base signal
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    if test_case.get('silence_only'):
        signal = 0.001 * torch.randn_like(t)
    
    elif test_case.get('noise_only'):
        signal = 0.3 * torch.randn_like(t)
    
    else:
        # Default speech-like signal
        amplitude = test_case.get('amplitude', 0.3)
        signal = amplitude * torch.sin(2 * np.pi * 300 * t)
        
        # Add harmonics for more realistic speech
        signal = signal + 0.15 * torch.sin(2 * np.pi * 600 * t)
        signal = signal + 0.1 * torch.sin(2 * np.pi * 900 * t)
    
    # Handle channels
    if channels == 1:
        signal = signal.unsqueeze(0)
    else:
        # Create stereo
        signal = torch.stack([signal, signal * 0.9])
    
    # Save file
    filename = f"{name}.wav"
    filepath = os.path.join(output_dir, filename)
    torchaudio.save(filepath, signal, sample_rate)
    
    return filepath

def test_model_quick(audio_file: str, model_name: str):
    """Quick test of model processing"""
    try:
        start_time = time.time()
        
        if model_name == 'large-v3':
            # Test Whisper
            import whisper
            model = whisper.load_model('large-v3')
            result = model.transcribe(audio_file)
            transcript = result['text']
            
        elif model_name == 'canary-1b':
            # Simulate Canary processing
            # In real implementation, you would use NeMo
            audio_info = torchaudio.info(audio_file)
            
            # Simulate model-specific issues
            if audio_info.num_frames / audio_info.sample_rate < 0.5:
                raise Exception("Audio too short for Canary model")
            
            if audio_info.num_frames / audio_info.sample_rate > 60:
                raise Exception("Audio too long for Canary model")
            
            transcript = f"canary transcript for {os.path.basename(audio_file)}"
            
        elif model_name == 'parakeet-tdt-0.6b-v2':
            # Simulate Parakeet processing
            # In real implementation, you would use NeMo
            audio_info = torchaudio.info(audio_file)
            
            # Simulate model-specific issues
            if audio_info.num_frames / audio_info.sample_rate < 1.0:
                raise Exception("Audio too short for Parakeet model")
            
            if audio_info.sample_rate != 16000:
                raise Exception("Parakeet requires 16kHz sample rate")
            
            transcript = f"parakeet transcript for {os.path.basename(audio_file)}"
            
        elif model_name == 'wav2vec-xls-r':
            # Simulate Wav2Vec2 processing
            # In real implementation, you would use transformers
            audio_info = torchaudio.info(audio_file)
            
            # Simulate model-specific issues
            if audio_info.num_frames / audio_info.sample_rate < 0.1:
                raise Exception("Audio too short for Wav2Vec2 model")
            
            waveform, _ = torchaudio.load(audio_file)
            if torch.abs(waveform).mean() < 0.01:
                raise Exception("Audio too quiet for Wav2Vec2 model")
            
            transcript = f"wav2vec transcript for {os.path.basename(audio_file)}"
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'transcript': transcript,
            'processing_time': processing_time,
            'transcript_length': len(transcript)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'processing_time': 0,
            'transcript_length': 0
        }

def run_quick_model_test():
    """Run quick model compatibility test"""
    print("Quick Model Compatibility Test")
    print("=" * 40)
    
    # Create quick test files
    temp_dir = tempfile.mkdtemp(prefix="quick_model_test_")
    audio_files = create_quick_test_files(temp_dir)
    
    # Models to test
    models = ['large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
    
    results = {}
    
    try:
        for audio_file in audio_files:
            file_path = audio_file['path']
            filename = os.path.basename(file_path)
            metadata = audio_file['metadata']
            
            print(f"\nTesting {filename}: {metadata['description']}")
            
            file_results = {}
            
            for model_name in models:
                print(f"  Testing {model_name}...")
                
                result = test_model_quick(file_path, model_name)
                file_results[model_name] = result
                
                if result['success']:
                    print(f"    ✓ {model_name}: {result['transcript_length']} chars, {result['processing_time']:.2f}s")
                else:
                    print(f"    ❌ {model_name}: {result['error']}")
            
            results[filename] = {
                'metadata': metadata,
                'model_results': file_results
            }
        
        # Analyze results
        print(f"\n" + "=" * 40)
        print("QUICK TEST RESULTS")
        print("=" * 40)
        
        # Count successes per model
        model_success_counts = {model: 0 for model in models}
        total_files = len(audio_files)
        
        for filename, file_data in results.items():
            for model_name, model_result in file_data['model_results'].items():
                if model_result['success']:
                    model_success_counts[model_name] += 1
        
        print(f"\nSuccess Rates:")
        for model_name, success_count in model_success_counts.items():
            success_rate = success_count / total_files * 100
            print(f"  {model_name}: {success_count}/{total_files} ({success_rate:.1f}%)")
        
        # Identify problematic files
        print(f"\nProblematic Files (by model):")
        for model_name in models:
            problematic_files = []
            for filename, file_data in results.items():
                if not file_data['model_results'][model_name]['success']:
                    problematic_files.append(filename)
            
            if problematic_files:
                print(f"  {model_name} failed on:")
                for file in problematic_files:
                    metadata = results[file]['metadata']
                    print(f"    - {file}: {metadata['description']}")
            else:
                print(f"  {model_name}: No failures")
        
        # Save results
        results_file = os.path.join(temp_dir, "quick_test_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_files': total_files,
                    'model_success_counts': model_success_counts,
                    'success_rates': {model: count/total_files for model, count in model_success_counts.items()}
                },
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return results
        
    finally:
        shutil.rmtree(temp_dir)

def test_with_short_real_files():
    """Test with short real audio files"""
    print(f"\n" + "=" * 40)
    print("TESTING WITH SHORT REAL FILES")
    print("=" * 40)
    
    # Look for shorter audio files
    real_audio_dirs = [
        "/media/meow/One Touch/ems_call/random_samples_1_preprocessed"
    ]
    
    short_audio_files = []
    for audio_dir in real_audio_dirs:
        if os.path.exists(audio_dir):
            wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            if wav_files:
                for wav_file in wav_files[:5]:  # Take first 5 files
                    file_path = os.path.join(audio_dir, wav_file)
                    
                    # Check file duration
                    try:
                        info = torchaudio.info(file_path)
                        duration = info.num_frames / info.sample_rate
                        
                        # Only use files shorter than 2 minutes
                        if duration < 120:
                            short_audio_files.append(file_path)
                            print(f"Found short file: {wav_file} ({duration:.1f}s)")
                    except Exception as e:
                        print(f"Error checking {wav_file}: {e}")
                break
    
    if not short_audio_files:
        print("No short real audio files found")
        return {}
    
    print(f"Testing with {len(short_audio_files)} short real audio files")
    
    models = ['large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
    results = {}
    
    for audio_file in short_audio_files:
        filename = os.path.basename(audio_file)
        print(f"\nTesting {filename}...")
        
        file_results = {}
        for model_name in models:
            result = test_model_quick(audio_file, model_name)
            file_results[model_name] = result
            
            if result['success']:
                print(f"  ✓ {model_name}: {result['transcript_length']} chars, {result['processing_time']:.2f}s")
            else:
                print(f"  ❌ {model_name}: {result['error']}")
        
        results[filename] = file_results
    
    # Analyze results
    print(f"\nReal Files Analysis:")
    for model_name in models:
        success_count = sum(1 for file_results in results.values() if file_results[model_name]['success'])
        total_files = len(results)
        success_rate = success_count / total_files * 100
        print(f"  {model_name}: {success_count}/{total_files} ({success_rate:.1f}%)")
    
    return results

def main():
    """Main function"""
    print("Quick Model Test")
    print("=" * 40)
    
    try:
        # Run quick test with synthetic files
        synthetic_results = run_quick_model_test()
        
        # Test with short real files
        real_results = test_with_short_real_files()
        
        # Summary
        print(f"\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        
        if synthetic_results:
            print(f"Synthetic files tested: {len(synthetic_results)}")
        
        if real_results:
            print(f"Real files tested: {len(real_results)}")
        
        print(f"\n✅ Quick model test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 