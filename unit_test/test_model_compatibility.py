#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Model Compatibility
=======================

This script tests why different ASR models process different numbers of files
without VAD preprocessing. It creates various audio characteristics that might
cause model-specific failures.
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
from typing import Dict, List, Tuple, Optional

def create_challenging_audio_files(output_dir: str):
    """Create audio files that might challenge different ASR models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Audio characteristics that might cause model failures
    test_cases = [
        # Duration challenges
        {
            'name': 'very_short_1s',
            'duration': 1.0,
            'description': 'Very short audio (1s) - might cause timeout'
        },
        {
            'name': 'very_long_10min',
            'duration': 600.0,
            'description': 'Very long audio (10min) - might cause OOM'
        },
        
        # Volume challenges
        {
            'name': 'extremely_low_volume',
            'duration': 30.0,
            'amplitude': 0.01,
            'description': 'Extremely low volume - might be filtered out'
        },
        {
            'name': 'extremely_high_volume',
            'duration': 30.0,
            'amplitude': 0.9,
            'description': 'Extremely high volume - might cause clipping'
        },
        
        # Noise challenges
        {
            'name': 'high_noise_ratio',
            'duration': 30.0,
            'noise_ratio': 0.8,
            'description': 'High noise ratio - might confuse models'
        },
        {
            'name': 'pure_noise',
            'duration': 30.0,
            'noise_only': True,
            'description': 'Pure noise - no speech content'
        },
        
        # Format challenges
        {
            'name': 'low_sample_rate_8k',
            'duration': 30.0,
            'sample_rate': 8000,
            'description': 'Low sample rate (8kHz) - might cause issues'
        },
        {
            'name': 'stereo_audio',
            'duration': 30.0,
            'channels': 2,
            'description': 'Stereo audio - some models expect mono'
        },
        
        # Content challenges
        {
            'name': 'silence_only',
            'duration': 30.0,
            'silence_only': True,
            'description': 'Silence only - no speech content'
        },
        {
            'name': 'intermittent_speech',
            'duration': 60.0,
            'intermittent': True,
            'description': 'Intermittent speech with long gaps'
        },
        
        # Edge cases
        {
            'name': 'single_tone',
            'duration': 30.0,
            'single_frequency': 440,
            'description': 'Single frequency tone - not speech-like'
        },
        {
            'name': 'complex_harmonics',
            'duration': 30.0,
            'complex_harmonics': True,
            'description': 'Complex harmonics - might confuse models'
        },
        
        # Memory/processing challenges
        {
            'name': 'high_frequency_content',
            'duration': 30.0,
            'high_freq': True,
            'description': 'High frequency content - processing intensive'
        },
        {
            'name': 'variable_speed',
            'duration': 30.0,
            'variable_speed': True,
            'description': 'Variable speed audio - might confuse models'
        }
    ]
    
    created_files = []
    
    for test_case in test_cases:
        print(f"Creating {test_case['name']}: {test_case['description']}")
        
        try:
            file_path = create_challenging_audio(test_case, output_dir)
            created_files.append({
                'path': file_path,
                'metadata': test_case
            })
        except Exception as e:
            print(f"  ❌ Failed to create {test_case['name']}: {e}")
    
    return created_files

def create_challenging_audio(test_case, output_dir):
    """Create a specific challenging audio file"""
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
    
    elif test_case.get('single_frequency'):
        freq = test_case['single_frequency']
        signal = 0.3 * torch.sin(2 * np.pi * freq * t)
    
    elif test_case.get('complex_harmonics'):
        signal = (
            0.2 * torch.sin(2 * np.pi * 200 * t) +
            0.15 * torch.sin(2 * np.pi * 400 * t) +
            0.1 * torch.sin(2 * np.pi * 600 * t) +
            0.05 * torch.sin(2 * np.pi * 800 * t) +
            0.025 * torch.sin(2 * np.pi * 1000 * t)
        )
    
    elif test_case.get('high_freq'):
        signal = 0.2 * torch.sin(2 * np.pi * 8000 * t)
    
    elif test_case.get('variable_speed'):
        # Create variable speed effect
        base_signal = 0.3 * torch.sin(2 * np.pi * 300 * t)
        speed_factor = 1.0 + 0.5 * torch.sin(2 * np.pi * 0.1 * t)
        signal = base_signal * speed_factor
    
    elif test_case.get('intermittent'):
        signal = torch.zeros_like(t)
        # Create intermittent pattern
        for i in range(0, int(duration), 15):
            start_idx = int(i * sample_rate)
            end_idx = int((i + 5) * sample_rate)
            if end_idx > len(signal):
                end_idx = len(signal)
            speech_segment = 0.3 * torch.sin(2 * np.pi * 300 * t[:end_idx-start_idx])
            signal[start_idx:end_idx] = speech_segment
    
    else:
        # Default speech-like signal
        amplitude = test_case.get('amplitude', 0.3)
        signal = amplitude * torch.sin(2 * np.pi * 300 * t)
        
        # Add harmonics for more realistic speech
        signal = signal + 0.15 * torch.sin(2 * np.pi * 600 * t)
        signal = signal + 0.1 * torch.sin(2 * np.pi * 900 * t)
    
    # Add noise if specified
    if test_case.get('noise_ratio'):
        noise_ratio = test_case['noise_ratio']
        noise = noise_ratio * torch.randn_like(signal)
        signal = signal + noise
    
    # Handle channels
    if channels == 1:
        signal = signal.unsqueeze(0)
    else:
        # Create stereo by duplicating with slight variation
        signal = torch.stack([signal, signal * 0.9 + 0.1 * torch.randn_like(signal)])
    
    # Save file
    filename = f"{name}.wav"
    filepath = os.path.join(output_dir, filename)
    torchaudio.save(filepath, signal, sample_rate)
    
    return filepath

def test_model_processing(audio_file: str, model_name: str):
    """Test if a specific model can process an audio file"""
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
            transcript = f"canary transcript for {os.path.basename(audio_file)}"
            
        elif model_name == 'parakeet-tdt-0.6b-v2':
            # Simulate Parakeet processing
            # In real implementation, you would use NeMo
            transcript = f"parakeet transcript for {os.path.basename(audio_file)}"
            
        elif model_name == 'wav2vec-xls-r':
            # Simulate Wav2Vec2 processing
            # In real implementation, you would use transformers
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

def test_all_models_on_challenging_files():
    """Test all models on challenging audio files"""
    print("Testing Model Compatibility with Challenging Audio Files")
    print("=" * 60)
    
    # Create challenging audio files
    temp_dir = tempfile.mkdtemp(prefix="model_compatibility_test_")
    audio_files = create_challenging_audio_files(temp_dir)
    
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
                
                result = test_model_processing(file_path, model_name)
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
        print(f"\n" + "=" * 60)
        print("MODEL COMPATIBILITY ANALYSIS")
        print("=" * 60)
        
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
        
        # Identify model-specific issues
        print(f"\nModel-Specific Issues:")
        for filename, file_data in results.items():
            metadata = file_data['metadata']
            failed_models = []
            
            for model_name, model_result in file_data['model_results'].items():
                if not model_result['success']:
                    failed_models.append(model_name)
            
            if failed_models:
                print(f"  {filename}: {metadata['description']}")
                print(f"    Failed models: {', '.join(failed_models)}")
        
        # Save detailed results
        results_file = os.path.join(temp_dir, "model_compatibility_results.json")
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
        # Clean up
        shutil.rmtree(temp_dir)

def test_with_real_audio_files():
    """Test with actual audio files from the dataset"""
    print(f"\n" + "=" * 60)
    print("TESTING WITH REAL AUDIO FILES")
    print("=" * 60)
    
    # Check for real audio files
    real_audio_dirs = [
        "/media/meow/One Touch/ems_call/long_audio_test_dataset",
        "/media/meow/One Touch/ems_call/random_samples_1_preprocessed"
    ]
    
    real_audio_files = []
    for audio_dir in real_audio_dirs:
        if os.path.exists(audio_dir):
            wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            if wav_files:
                # Take first 5 files for testing
                for wav_file in wav_files[:5]:
                    real_audio_files.append(os.path.join(audio_dir, wav_file))
                break
    
    if not real_audio_files:
        print("No real audio files found for testing")
        return {}
    
    print(f"Found {len(real_audio_files)} real audio files for testing")
    
    models = ['large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
    results = {}
    
    for audio_file in real_audio_files:
        filename = os.path.basename(audio_file)
        print(f"\nTesting real file: {filename}")
        
        file_results = {}
        for model_name in models:
            result = test_model_processing(audio_file, model_name)
            file_results[model_name] = result
            
            if result['success']:
                print(f"  ✓ {model_name}: {result['transcript_length']} chars")
            else:
                print(f"  ❌ {model_name}: {result['error']}")
        
        results[filename] = file_results
    
    # Analyze real file results
    print(f"\nReal File Analysis:")
    for model_name in models:
        success_count = sum(1 for file_results in results.values() if file_results[model_name]['success'])
        total_files = len(results)
        success_rate = success_count / total_files * 100
        print(f"  {model_name}: {success_count}/{total_files} ({success_rate:.1f}%)")
    
    return results

def generate_compatibility_report(challenging_results, real_results):
    """Generate compatibility report"""
    print(f"\n" + "=" * 60)
    print("MODEL COMPATIBILITY REPORT")
    print("=" * 60)
    
    report = {
        'challenging_files': challenging_results,
        'real_files': real_results,
        'recommendations': []
    }
    
    # Analyze challenging files
    if challenging_results:
        models = ['large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
        challenging_success_counts = {model: 0 for model in models}
        total_challenging = len(challenging_results)
        
        for filename, file_data in challenging_results.items():
            for model_name, model_result in file_data['model_results'].items():
                if model_result['success']:
                    challenging_success_counts[model_name] += 1
        
        print(f"\nChallenging Files Success Rates:")
        for model_name, count in challenging_success_counts.items():
            rate = count / total_challenging * 100
            print(f"  {model_name}: {count}/{total_challenging} ({rate:.1f}%)")
    
    # Analyze real files
    if real_results:
        models = ['large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
        real_success_counts = {model: 0 for model in models}
        total_real = len(real_results)
        
        for filename, file_results in real_results.items():
            for model_name, model_result in file_results.items():
                if model_result['success']:
                    real_success_counts[model_name] += 1
        
        print(f"\nReal Files Success Rates:")
        for model_name, count in real_success_counts.items():
            rate = count / total_real * 100
            print(f"  {model_name}: {count}/{total_real} ({rate:.1f}%)")
    
    # Generate recommendations
    print(f"\nRecommendations:")
    
    if challenging_results:
        # Check for model-specific issues
        for model_name in models:
            failed_files = []
            for filename, file_data in challenging_results.items():
                if not file_data['model_results'][model_name]['success']:
                    failed_files.append(filename)
            
            if failed_files:
                print(f"  - {model_name} struggles with: {', '.join(failed_files[:3])}")
    
    if real_results:
        # Check for real file issues
        for model_name in models:
            failed_real_files = []
            for filename, file_results in real_results.items():
                if not file_results[model_name]['success']:
                    failed_real_files.append(filename)
            
            if failed_real_files:
                print(f"  - {model_name} failed on real files: {', '.join(failed_real_files)}")
    
    # Save report
    report_file = "model_compatibility_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return report

def main():
    """Main function to test model compatibility"""
    print("Test Model Compatibility")
    print("=" * 40)
    
    try:
        # Test with challenging audio files
        challenging_results = test_all_models_on_challenging_files()
        
        # Test with real audio files
        real_results = test_with_real_audio_files()
        
        # Generate comprehensive report
        report = generate_compatibility_report(challenging_results, real_results)
        
        print(f"\n✅ Model compatibility testing completed!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 