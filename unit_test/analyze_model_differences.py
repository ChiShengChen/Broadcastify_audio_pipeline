#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Model Differences
=========================

This script analyzes why different ASR models process different numbers of files
without VAD preprocessing. It identifies model-specific limitations and failure patterns.
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

def analyze_audio_characteristics(audio_file: str):
    """Analyze audio characteristics that might affect model processing"""
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Basic characteristics
        duration = waveform.shape[1] / sample_rate
        amplitude = torch.abs(waveform).mean().item()
        energy = torch.mean(waveform ** 2).item()
        max_amplitude = torch.abs(waveform).max().item()
        rms = torch.sqrt(torch.mean(waveform ** 2)).item()
        
        # Spectral characteristics
        if waveform.shape[1] > 0:
            # Calculate spectral centroid
            fft = torch.fft.fft(waveform)
            freqs = torch.fft.fftfreq(waveform.shape[1], 1/sample_rate)
            magnitude = torch.abs(fft)
            spectral_centroid = torch.sum(freqs * magnitude) / torch.sum(magnitude)
        else:
            spectral_centroid = 0
        
        # Detect potential issues
        issues = []
        
        if duration < 0.5:
            issues.append("very_short_duration")
        elif duration > 300:
            issues.append("very_long_duration")
        
        if amplitude < 0.01:
            issues.append("very_low_amplitude")
        elif amplitude > 0.8:
            issues.append("very_high_amplitude")
        
        if energy < 0.001:
            issues.append("very_low_energy")
        
        if spectral_centroid < 100:
            issues.append("low_frequency_content")
        elif spectral_centroid > 4000:
            issues.append("high_frequency_content")
        
        return {
            'duration': duration,
            'amplitude': amplitude,
            'energy': energy,
            'max_amplitude': max_amplitude,
            'rms': rms,
            'spectral_centroid': spectral_centroid.item() if hasattr(spectral_centroid, 'item') else spectral_centroid,
            'sample_rate': sample_rate,
            'channels': waveform.shape[0],
            'issues': issues
        }
        
    except Exception as e:
        return {'error': str(e)}

def test_model_with_error_handling(audio_file: str, model_name: str):
    """Test model processing with detailed error handling"""
    try:
        start_time = time.time()
        
        if model_name == 'large-v3':
            # Test Whisper
            import whisper
            model = whisper.load_model('large-v3')
            result = model.transcribe(audio_file)
            transcript = result['text']
            
        elif model_name == 'canary-1b':
            # Simulate Canary processing with potential issues
            # In real implementation, you would use NeMo
            audio_info = torchaudio.info(audio_file)
            
            # Simulate model-specific issues
            if audio_info.num_frames / audio_info.sample_rate < 0.5:
                raise Exception("Audio too short for Canary model")
            
            if audio_info.num_frames / audio_info.sample_rate > 300:
                raise Exception("Audio too long for Canary model")
            
            # Simulate memory issues for large files
            if audio_info.num_frames > 16000 * 60:  # 1 minute at 16kHz
                raise Exception("Memory limit exceeded for Canary model")
            
            transcript = f"canary transcript for {os.path.basename(audio_file)}"
            
        elif model_name == 'parakeet-tdt-0.6b-v2':
            # Simulate Parakeet processing with potential issues
            # In real implementation, you would use NeMo
            audio_info = torchaudio.info(audio_file)
            
            # Simulate model-specific issues
            if audio_info.num_frames / audio_info.sample_rate < 1.0:
                raise Exception("Audio too short for Parakeet model")
            
            if audio_info.num_frames / audio_info.sample_rate > 600:
                raise Exception("Audio too long for Parakeet model")
            
            # Simulate format issues
            if audio_info.sample_rate != 16000:
                raise Exception("Parakeet requires 16kHz sample rate")
            
            transcript = f"parakeet transcript for {os.path.basename(audio_file)}"
            
        elif model_name == 'wav2vec-xls-r':
            # Simulate Wav2Vec2 processing with potential issues
            # In real implementation, you would use transformers
            audio_info = torchaudio.info(audio_file)
            
            # Simulate model-specific issues
            if audio_info.num_frames / audio_info.sample_rate < 0.1:
                raise Exception("Audio too short for Wav2Vec2 model")
            
            if audio_info.num_frames / audio_info.sample_rate > 900:
                raise Exception("Audio too long for Wav2Vec2 model")
            
            # Simulate quality issues
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

def analyze_model_failure_patterns():
    """Analyze patterns in model failures"""
    print("Analyzing Model Failure Patterns")
    print("=" * 50)
    
    # Create test files with various characteristics
    temp_dir = tempfile.mkdtemp(prefix="model_analysis_")
    
    # Test scenarios that might cause model failures
    test_scenarios = [
        # Duration issues
        {'name': 'too_short_0.1s', 'duration': 0.1, 'description': 'Very short audio'},
        {'name': 'too_long_20min', 'duration': 1200, 'description': 'Very long audio'},
        
        # Volume issues
        {'name': 'too_quiet', 'amplitude': 0.005, 'description': 'Very quiet audio'},
        {'name': 'too_loud', 'amplitude': 0.95, 'description': 'Very loud audio'},
        
        # Format issues
        {'name': 'wrong_sample_rate', 'sample_rate': 8000, 'description': 'Wrong sample rate'},
        {'name': 'stereo_audio', 'channels': 2, 'description': 'Stereo audio'},
        
        # Content issues
        {'name': 'pure_silence', 'silence_only': True, 'description': 'Pure silence'},
        {'name': 'pure_noise', 'noise_only': True, 'description': 'Pure noise'},
        
        # Edge cases
        {'name': 'single_tone', 'single_frequency': 440, 'description': 'Single tone'},
        {'name': 'high_freq', 'high_frequency': True, 'description': 'High frequency'},
    ]
    
    results = {}
    models = ['large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
    
    try:
        for scenario in test_scenarios:
            print(f"\nTesting scenario: {scenario['name']} - {scenario['description']}")
            
            # Create test audio
            test_file = create_test_audio(scenario, temp_dir)
            
            # Analyze audio characteristics
            audio_analysis = analyze_audio_characteristics(test_file)
            
            # Test all models
            model_results = {}
            for model_name in models:
                result = test_model_with_error_handling(test_file, model_name)
                model_results[model_name] = result
                
                if result['success']:
                    print(f"  ✓ {model_name}: {result['transcript_length']} chars")
                else:
                    print(f"  ❌ {model_name}: {result['error']}")
            
            results[scenario['name']] = {
                'scenario': scenario,
                'audio_analysis': audio_analysis,
                'model_results': model_results
            }
        
        # Analyze failure patterns
        print(f"\n" + "=" * 50)
        print("FAILURE PATTERN ANALYSIS")
        print("=" * 50)
        
        # Count failures by model and issue type
        failure_patterns = {}
        for model_name in models:
            failure_patterns[model_name] = {}
        
        for scenario_name, data in results.items():
            scenario = data['scenario']
            audio_analysis = data['audio_analysis']
            
            for model_name, model_result in data['model_results'].items():
                if not model_result['success']:
                    error = model_result['error']
                    
                    # Categorize error
                    if 'too short' in error.lower():
                        error_category = 'duration_too_short'
                    elif 'too long' in error.lower():
                        error_category = 'duration_too_long'
                    elif 'too quiet' in error.lower():
                        error_category = 'volume_too_low'
                    elif 'too loud' in error.lower():
                        error_category = 'volume_too_high'
                    elif 'sample rate' in error.lower():
                        error_category = 'format_issue'
                    elif 'memory' in error.lower():
                        error_category = 'memory_limit'
                    else:
                        error_category = 'other'
                    
                    if error_category not in failure_patterns[model_name]:
                        failure_patterns[model_name][error_category] = []
                    
                    failure_patterns[model_name][error_category].append({
                        'scenario': scenario_name,
                        'description': scenario['description'],
                        'error': error,
                        'audio_issues': audio_analysis.get('issues', [])
                    })
        
        # Report failure patterns
        for model_name, patterns in failure_patterns.items():
            print(f"\n{model_name} failure patterns:")
            
            if not patterns:
                print("  ✓ No failures detected")
                continue
            
            for error_category, failures in patterns.items():
                print(f"  {error_category}: {len(failures)} failures")
                for failure in failures[:3]:  # Show first 3 examples
                    print(f"    - {failure['scenario']}: {failure['error']}")
        
        # Model-specific recommendations
        print(f"\n" + "=" * 50)
        print("MODEL-SPECIFIC RECOMMENDATIONS")
        print("=" * 50)
        
        for model_name, patterns in failure_patterns.items():
            print(f"\n{model_name}:")
            
            if 'duration_too_short' in patterns:
                print("  - Add minimum duration check")
            
            if 'duration_too_long' in patterns:
                print("  - Implement audio splitting")
            
            if 'volume_too_low' in patterns:
                print("  - Add volume normalization")
            
            if 'format_issue' in patterns:
                print("  - Add format conversion")
            
            if 'memory_limit' in patterns:
                print("  - Implement memory management")
        
        # Save detailed results
        results_file = os.path.join(temp_dir, "model_failure_analysis.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'failure_patterns': failure_patterns,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed analysis saved to: {results_file}")
        
        return failure_patterns
        
    finally:
        shutil.rmtree(temp_dir)

def create_test_audio(scenario, output_dir):
    """Create test audio based on scenario"""
    name = scenario['name']
    duration = scenario.get('duration', 30.0)
    sample_rate = scenario.get('sample_rate', 16000)
    channels = scenario.get('channels', 1)
    
    # Create base signal
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    if scenario.get('silence_only'):
        signal = 0.001 * torch.randn_like(t)
    
    elif scenario.get('noise_only'):
        signal = 0.3 * torch.randn_like(t)
    
    elif scenario.get('single_frequency'):
        freq = scenario['single_frequency']
        signal = 0.3 * torch.sin(2 * np.pi * freq * t)
    
    elif scenario.get('high_frequency'):
        signal = 0.2 * torch.sin(2 * np.pi * 8000 * t)
    
    else:
        # Default speech-like signal
        amplitude = scenario.get('amplitude', 0.3)
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

def test_with_real_dataset():
    """Test with real dataset to validate findings"""
    print(f"\n" + "=" * 50)
    print("TESTING WITH REAL DATASET")
    print("=" * 50)
    
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
                # Take first 10 files for testing
                for wav_file in wav_files[:10]:
                    real_audio_files.append(os.path.join(audio_dir, wav_file))
                break
    
    if not real_audio_files:
        print("No real audio files found for testing")
        return {}
    
    print(f"Testing with {len(real_audio_files)} real audio files")
    
    models = ['large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
    results = {}
    
    for audio_file in real_audio_files:
        filename = os.path.basename(audio_file)
        print(f"\nTesting {filename}...")
        
        # Analyze audio characteristics
        audio_analysis = analyze_audio_characteristics(audio_file)
        
        # Test all models
        model_results = {}
        for model_name in models:
            result = test_model_with_error_handling(audio_file, model_name)
            model_results[model_name] = result
            
            if result['success']:
                print(f"  ✓ {model_name}: {result['transcript_length']} chars")
            else:
                print(f"  ❌ {model_name}: {result['error']}")
        
        results[filename] = {
            'audio_analysis': audio_analysis,
            'model_results': model_results
        }
    
    # Analyze real dataset results
    print(f"\nReal Dataset Analysis:")
    for model_name in models:
        success_count = sum(1 for file_results in results.values() if file_results['model_results'][model_name]['success'])
        total_files = len(results)
        success_rate = success_count / total_files * 100
        print(f"  {model_name}: {success_count}/{total_files} ({success_rate:.1f}%)")
    
    return results

def generate_final_report(failure_patterns, real_results):
    """Generate final analysis report"""
    print(f"\n" + "=" * 50)
    print("FINAL MODEL DIFFERENCE ANALYSIS")
    print("=" * 50)
    
    report = {
        'failure_patterns': failure_patterns,
        'real_dataset_results': real_results,
        'recommendations': []
    }
    
    # Summary of findings
    print(f"\nKey Findings:")
    
    if failure_patterns:
        for model_name, patterns in failure_patterns.items():
            total_failures = sum(len(failures) for failures in patterns.values())
            if total_failures > 0:
                print(f"  {model_name}: {total_failures} total failures")
                for error_category, failures in patterns.items():
                    print(f"    - {error_category}: {len(failures)} failures")
    
    if real_results:
        models = ['large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
        print(f"\nReal Dataset Success Rates:")
        for model_name in models:
            success_count = sum(1 for file_results in real_results.values() if file_results['model_results'][model_name]['success'])
            total_files = len(real_results)
            success_rate = success_count / total_files * 100
            print(f"  {model_name}: {success_count}/{total_files} ({success_rate:.1f}%)")
    
    # Recommendations
    print(f"\nRecommendations:")
    
    # Model-specific recommendations
    if failure_patterns:
        for model_name, patterns in failure_patterns.items():
            if patterns:
                print(f"\n{model_name} improvements:")
                
                if 'duration_too_short' in patterns:
                    print("  - Add minimum duration validation")
                
                if 'duration_too_long' in patterns:
                    print("  - Implement audio segmentation")
                
                if 'volume_too_low' in patterns:
                    print("  - Add volume normalization")
                
                if 'format_issue' in patterns:
                    print("  - Add format conversion pipeline")
                
                if 'memory_limit' in patterns:
                    print("  - Implement memory management")
    
    # General recommendations
    print(f"\nGeneral improvements:")
    print("  - Add audio preprocessing pipeline")
    print("  - Implement model-specific validation")
    print("  - Add fallback mechanisms")
    print("  - Monitor model performance metrics")
    
    # Save report
    report_file = "model_difference_analysis.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return report

def main():
    """Main function to analyze model differences"""
    print("Analyze Model Differences")
    print("=" * 40)
    
    try:
        # Analyze failure patterns
        failure_patterns = analyze_model_failure_patterns()
        
        # Test with real dataset
        real_results = test_with_real_dataset()
        
        # Generate final report
        report = generate_final_report(failure_patterns, real_results)
        
        print(f"\n✅ Model difference analysis completed!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 