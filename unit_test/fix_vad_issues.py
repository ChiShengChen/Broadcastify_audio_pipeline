#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix VAD Issues
==============

This script provides solutions for the VAD detection issues identified:
1. Adjust VAD parameters for better detection
2. Test with real audio files
3. Provide recommendations for production use
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

def create_realistic_speech_files(output_dir: str):
    """Create more realistic speech-like audio files"""
    os.makedirs(output_dir, exist_ok=True)
    
    sample_rate = 16000
    
    # Create more realistic speech patterns
    test_cases = [
        ('continuous_speech', 8.0, lambda t: 0.3 * torch.sin(2 * np.pi * 200 * t) + 0.2 * torch.sin(2 * np.pi * 400 * t) + 0.1 * torch.sin(2 * np.pi * 600 * t)),
        ('speech_with_pauses', 10.0, lambda t: torch.where((t > 2) & (t < 6), 0.4 * torch.sin(2 * np.pi * 250 * t), 0.01 * torch.randn_like(t))),
        ('loud_speech', 6.0, lambda t: 0.6 * torch.sin(2 * np.pi * 300 * t) + 0.3 * torch.sin(2 * np.pi * 500 * t)),
        ('quiet_speech', 7.0, lambda t: 0.15 * torch.sin(2 * np.pi * 200 * t) + 0.1 * torch.sin(2 * np.pi * 400 * t)),
        ('speech_with_noise', 9.0, lambda t: 0.3 * torch.sin(2 * np.pi * 250 * t) + 0.15 * torch.randn_like(t)),
    ]
    
    files_created = []
    
    for i, (name, duration, signal_func) in enumerate(test_cases):
        t = torch.linspace(0, duration, int(sample_rate * duration))
        signal = signal_func(t)
        
        # Add small noise
        signal = signal + 0.02 * torch.randn_like(signal)
        
        filename = f"{name}_{i:02d}.wav"
        filepath = os.path.join(output_dir, filename)
        torchaudio.save(filepath, signal.unsqueeze(0), sample_rate)
        files_created.append(filepath)
        
        print(f"Created: {filename} (duration: {duration}s)")
    
    return files_created

def test_optimized_vad_parameters(audio_file: str):
    """Test VAD with optimized parameters"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from vad_pipeline import VADPipeline
        
        # Optimized parameter sets based on diagnosis
        optimized_sets = [
            {'name': 'very_sensitive', 'speech_threshold': 0.1, 'min_speech_duration': 0.1, 'min_silence_duration': 0.1},
            {'name': 'sensitive', 'speech_threshold': 0.2, 'min_speech_duration': 0.2, 'min_silence_duration': 0.15},
            {'name': 'balanced', 'speech_threshold': 0.3, 'min_speech_duration': 0.3, 'min_silence_duration': 0.2},
            {'name': 'conservative', 'speech_threshold': 0.4, 'min_speech_duration': 0.4, 'min_silence_duration': 0.25},
            {'name': 'very_conservative', 'speech_threshold': 0.5, 'min_speech_duration': 0.5, 'min_silence_duration': 0.3},
        ]
        
        results = {}
        
        for param_set in optimized_sets:
            temp_dir = tempfile.mkdtemp(prefix=f"vad_opt_{param_set['name']}_")
            
            try:
                vad = VADPipeline(
                    speech_threshold=param_set['speech_threshold'],
                    min_speech_duration=param_set['min_speech_duration'],
                    min_silence_duration=param_set['min_silence_duration'],
                    target_sample_rate=16000
                )
                
                result = vad.process_audio_file(audio_file, temp_dir)
                results[param_set['name']] = {
                    'params': param_set,
                    'result': result
                }
                
            except Exception as e:
                results[param_set['name']] = {
                    'params': param_set,
                    'error': str(e)
                }
            
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        
        return results
    
    except Exception as e:
        return {'error': f"Optimized VAD testing failed: {str(e)}"}

def test_with_real_audio_files():
    """Test with actual audio files from the dataset"""
    print("\n" + "=" * 50)
    print("TESTING WITH REAL AUDIO FILES")
    print("=" * 50)
    
    # Check if we have access to real audio files
    real_audio_dirs = [
        "/media/meow/One Touch/ems_call/long_audio_test_dataset",
        "/media/meow/One Touch/ems_call/random_samples_1_preprocessed"
    ]
    
    real_audio_files = []
    for audio_dir in real_audio_dirs:
        if os.path.exists(audio_dir):
            wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            if wav_files:
                # Take first 3 files for testing
                for wav_file in wav_files[:3]:
                    real_audio_files.append(os.path.join(audio_dir, wav_file))
                break
    
    if not real_audio_files:
        print("No real audio files found for testing")
        return {}
    
    print(f"Found {len(real_audio_files)} real audio files for testing")
    
    results = {}
    for audio_file in real_audio_files:
        filename = os.path.basename(audio_file)
        print(f"\nTesting real file: {filename}")
        
        result = test_optimized_vad_parameters(audio_file)
        results[filename] = result
        
        # Show results for this file
        for param_name, param_result in result.items():
            if 'error' not in param_result:
                vad_result = param_result['result']
                if 'total_speech_duration' in vad_result:
                    speech_ratio = vad_result.get('speech_ratio', 0)
                    print(f"  {param_name}: {vad_result['total_speech_duration']:.2f}s ({speech_ratio:.1%})")
                else:
                    print(f"  {param_name}: No speech detected")
            else:
                print(f"  {param_name}: Error - {param_result['error']}")
    
    return results

def generate_recommendations():
    """Generate recommendations for fixing VAD issues"""
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS FOR FIXING VAD ISSUES")
    print("=" * 50)
    
    recommendations = {
        'immediate_fixes': [
            "1. Lower VAD speech threshold from 0.5 to 0.2-0.3",
            "2. Reduce min_speech_duration from 0.5s to 0.2-0.3s", 
            "3. Reduce min_silence_duration from 0.3s to 0.15-0.2s",
            "4. Test with real audio files to validate parameters"
        ],
        'pipeline_changes': [
            "1. Update run_pipeline.sh VAD parameters",
            "2. Add VAD parameter validation",
            "3. Implement fallback to original files if VAD fails",
            "4. Add VAD quality metrics to pipeline output"
        ],
        'code_changes': [
            "1. Modify vad_pipeline.py default parameters",
            "2. Add VAD parameter tuning based on audio characteristics",
            "3. Implement adaptive VAD thresholds",
            "4. Add VAD failure detection and reporting"
        ]
    }
    
    for category, items in recommendations.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for item in items:
            print(f"  {item}")
    
    return recommendations

def create_fixed_vad_config():
    """Create a fixed VAD configuration file"""
    fixed_config = {
        'vad_parameters': {
            'speech_threshold': 0.2,  # Reduced from 0.5
            'min_speech_duration': 0.2,  # Reduced from 0.5
            'min_silence_duration': 0.15,  # Reduced from 0.3
            'target_sample_rate': 16000,
            'chunk_size': 512
        },
        'fallback_options': {
            'use_original_if_vad_fails': True,
            'min_speech_ratio_threshold': 0.01,  # 1% minimum speech
            'max_vad_failure_rate': 0.5  # 50% maximum failure rate
        },
        'quality_metrics': {
            'track_speech_ratio': True,
            'track_segment_count': True,
            'track_processing_time': True
        }
    }
    
    config_file = "fixed_vad_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_config, f, indent=2, ensure_ascii=False)
    
    print(f"\nFixed VAD configuration saved to: {config_file}")
    return config_file

def main():
    """Main function to fix VAD issues"""
    print("Fix VAD Issues")
    print("=" * 40)
    
    # Create test directory
    test_dir = tempfile.mkdtemp(prefix="fix_vad_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Create realistic test files
        audio_dir = os.path.join(test_dir, "realistic_audio")
        audio_files = create_realistic_speech_files(audio_dir)
        
        print(f"\nCreated {len(audio_files)} realistic speech files")
        
        # Test with optimized parameters
        print("\n" + "=" * 40)
        print("TESTING OPTIMIZED VAD PARAMETERS")
        print("=" * 40)
        
        test_file = audio_files[0]
        filename = os.path.basename(test_file)
        print(f"\nTesting optimized parameters on {filename}...")
        
        optimized_results = test_optimized_vad_parameters(test_file)
        
        if 'error' in optimized_results:
            print(f"❌ Optimized testing failed: {optimized_results['error']}")
        else:
            for param_name, result in optimized_results.items():
                params = result['params']
                print(f"\n  {param_name.upper()}:")
                print(f"    Threshold: {params['speech_threshold']}")
                print(f"    Min speech: {params['min_speech_duration']}s")
                print(f"    Min silence: {params['min_silence_duration']}s")
                
                if 'error' in result:
                    print(f"    ❌ Error: {result['error']}")
                else:
                    vad_result = result['result']
                    if 'total_speech_duration' in vad_result:
                        speech_ratio = vad_result.get('speech_ratio', 0)
                        print(f"    ✓ Speech: {vad_result['total_speech_duration']:.2f}s ({speech_ratio:.1%})")
                    else:
                        print(f"    ❌ No speech detected")
        
        # Test with real audio files
        real_results = test_with_real_audio_files()
        
        # Generate recommendations
        recommendations = generate_recommendations()
        
        # Create fixed configuration
        config_file = create_fixed_vad_config()
        
        # Summary
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        
        print(f"\n✅ VAD issues identified and solutions provided")
        print(f"✅ Fixed configuration created: {config_file}")
        print(f"✅ Recommendations generated for implementation")
        
        if real_results:
            print(f"✅ Tested with {len(real_results)} real audio files")
        
        print(f"\nNext steps:")
        print(f"1. Update VAD parameters in run_pipeline.sh")
        print(f"2. Test with real dataset")
        print(f"3. Monitor VAD performance")
        print(f"4. Implement fallback mechanisms")
        
    except Exception as e:
        print(f"Fix failed: {e}")
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