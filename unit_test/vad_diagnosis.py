#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAD Diagnosis Tool
==================

This script specifically diagnoses why VAD is not detecting speech segments
in the test audio files, which is causing all ASR models to fail when VAD is enabled.
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
import matplotlib.pyplot as plt

def create_diagnostic_audio_files(output_dir: str):
    """Create audio files with different characteristics for VAD diagnosis"""
    os.makedirs(output_dir, exist_ok=True)
    
    sample_rate = 16000
    
    # Create different types of audio for testing
    test_cases = [
        ('pure_sine', 5.0, lambda t: 0.5 * torch.sin(2 * np.pi * 300 * t)),
        ('speech_like', 5.0, lambda t: 0.3 * torch.sin(2 * np.pi * 200 * t) + 0.2 * torch.sin(2 * np.pi * 400 * t)),
        ('noisy_speech', 5.0, lambda t: 0.3 * torch.sin(2 * np.pi * 250 * t) + 0.2 * torch.randn_like(t)),
        ('low_volume', 5.0, lambda t: 0.1 * torch.sin(2 * np.pi * 300 * t)),
        ('high_volume', 5.0, lambda t: 0.8 * torch.sin(2 * np.pi * 300 * t)),
        ('intermittent', 5.0, lambda t: torch.where(t < 2.5, 0.4 * torch.sin(2 * np.pi * 300 * t), 0.01 * torch.randn_like(t))),
        ('complex_speech', 5.0, lambda t: 0.3 * torch.sin(2 * np.pi * 200 * t) + 0.15 * torch.sin(2 * np.pi * 400 * t) + 0.1 * torch.sin(2 * np.pi * 600 * t)),
    ]
    
    files_created = []
    
    for i, (name, duration, signal_func) in enumerate(test_cases):
        t = torch.linspace(0, duration, int(sample_rate * duration))
        signal = signal_func(t)
        
        # Add small noise to all signals
        signal = signal + 0.01 * torch.randn_like(signal)
        
        filename = f"{name}_{i:02d}.wav"
        filepath = os.path.join(output_dir, filename)
        torchaudio.save(filepath, signal.unsqueeze(0), sample_rate)
        files_created.append(filepath)
        
        print(f"Created: {filename} (duration: {duration}s)")
    
    return files_created

def analyze_audio_characteristics(audio_file: str):
    """Analyze audio file characteristics that might affect VAD"""
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
    
    return {
        'duration': duration,
        'amplitude': amplitude,
        'energy': energy,
        'max_amplitude': max_amplitude,
        'rms': rms,
        'spectral_centroid': spectral_centroid.item() if hasattr(spectral_centroid, 'item') else spectral_centroid,
        'file_size': os.path.getsize(audio_file)
    }

def test_vad_detection(audio_file: str):
    """Test VAD detection on a single file"""
    try:
        # Import VAD pipeline
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from vad_pipeline import VADPipeline
        
        # Create VAD pipeline
        vad = VADPipeline(
            speech_threshold=0.5,
            min_speech_duration=0.5,
            min_silence_duration=0.3,
            target_sample_rate=16000
        )
        
        # Create temporary output directory
        temp_dir = tempfile.mkdtemp(prefix="vad_test_")
        
        try:
            # Process single file
            result = vad.process_audio_file(audio_file, temp_dir)
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            return result
            
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return {'error': str(e)}
    
    except Exception as e:
        return {'error': f"VAD not available: {str(e)}"}

def test_vad_parameters(audio_file: str):
    """Test VAD with different parameters"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from vad_pipeline import VADPipeline
        
        # Test different VAD parameters
        parameter_sets = [
            {'speech_threshold': 0.1, 'min_speech_duration': 0.1, 'min_silence_duration': 0.1},
            {'speech_threshold': 0.3, 'min_speech_duration': 0.3, 'min_silence_duration': 0.2},
            {'speech_threshold': 0.5, 'min_speech_duration': 0.5, 'min_silence_duration': 0.3},
            {'speech_threshold': 0.7, 'min_speech_duration': 0.7, 'min_silence_duration': 0.4},
            {'speech_threshold': 0.9, 'min_speech_duration': 0.9, 'min_silence_duration': 0.5},
        ]
        
        results = {}
        
        for i, params in enumerate(parameter_sets):
            temp_dir = tempfile.mkdtemp(prefix=f"vad_test_{i}_")
            
            try:
                vad = VADPipeline(
                    speech_threshold=params['speech_threshold'],
                    min_speech_duration=params['min_speech_duration'],
                    min_silence_duration=params['min_silence_duration'],
                    target_sample_rate=16000
                )
                
                result = vad.process_audio_file(audio_file, temp_dir)
                results[f"set_{i}"] = {
                    'params': params,
                    'result': result
                }
                
            except Exception as e:
                results[f"set_{i}"] = {
                    'params': params,
                    'error': str(e)
                }
            
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        
        return results
    
    except Exception as e:
        return {'error': f"VAD parameter testing failed: {str(e)}"}

def main():
    """Main diagnosis function"""
    print("VAD Diagnosis Tool")
    print("=" * 40)
    
    # Create test directory
    test_dir = tempfile.mkdtemp(prefix="vad_diagnosis_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Create diagnostic audio files
        audio_dir = os.path.join(test_dir, "diagnostic_audio")
        audio_files = create_diagnostic_audio_files(audio_dir)
        
        print(f"\nCreated {len(audio_files)} diagnostic audio files")
        
        # Analyze each file
        print("\n" + "=" * 40)
        print("AUDIO CHARACTERISTICS ANALYSIS")
        print("=" * 40)
        
        characteristics = {}
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            print(f"\nAnalyzing {filename}...")
            
            char = analyze_audio_characteristics(audio_file)
            characteristics[filename] = char
            
            print(f"  Duration: {char['duration']:.2f}s")
            print(f"  Amplitude: {char['amplitude']:.4f}")
            print(f"  Energy: {char['energy']:.6f}")
            print(f"  RMS: {char['rms']:.4f}")
            print(f"  Max amplitude: {char['max_amplitude']:.4f}")
            print(f"  Spectral centroid: {char['spectral_centroid']:.1f} Hz")
        
        # Test VAD detection on each file
        print("\n" + "=" * 40)
        print("VAD DETECTION TESTING")
        print("=" * 40)
        
        vad_results = {}
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            print(f"\nTesting VAD on {filename}...")
            
            result = test_vad_detection(audio_file)
            vad_results[filename] = result
            
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ✓ VAD processed successfully")
                if 'total_speech_duration' in result:
                    print(f"  Speech duration: {result['total_speech_duration']:.2f}s")
                    print(f"  Speech ratio: {result.get('speech_ratio', 0):.1%}")
                    print(f"  Segments found: {len(result.get('segments', []))}")
        
        # Test VAD with different parameters
        print("\n" + "=" * 40)
        print("VAD PARAMETER TESTING")
        print("=" * 40)
        
        # Test on one representative file
        test_file = audio_files[0]
        filename = os.path.basename(test_file)
        print(f"\nTesting VAD parameters on {filename}...")
        
        param_results = test_vad_parameters(test_file)
        
        if 'error' in param_results:
            print(f"  ❌ Parameter testing failed: {param_results['error']}")
        else:
            for param_set, result in param_results.items():
                params = result['params']
                print(f"\n  Parameters: threshold={params['speech_threshold']}, "
                      f"min_speech={params['min_speech_duration']}, "
                      f"min_silence={params['min_silence_duration']}")
                
                if 'error' in result:
                    print(f"    ❌ Error: {result['error']}")
                else:
                    vad_result = result['result']
                    if 'total_speech_duration' in vad_result:
                        print(f"    ✓ Speech: {vad_result['total_speech_duration']:.2f}s "
                              f"({vad_result.get('speech_ratio', 0):.1%})")
                    else:
                        print(f"    ❌ No speech detected")
        
        # Generate diagnosis report
        print("\n" + "=" * 40)
        print("DIAGNOSIS REPORT")
        print("=" * 40)
        
        # Count successful VAD detections
        successful_vad = sum(1 for result in vad_results.values() if 'error' not in result and result.get('total_speech_duration', 0) > 0)
        total_files = len(audio_files)
        
        print(f"\nVAD Detection Summary:")
        print(f"  Total files: {total_files}")
        print(f"  Files with speech detected: {successful_vad}")
        print(f"  Detection rate: {successful_vad/total_files:.1%}")
        
        # Identify problematic files
        print(f"\nProblematic Files:")
        for filename, result in vad_results.items():
            if 'error' in result:
                print(f"  ❌ {filename}: {result['error']}")
            elif result.get('total_speech_duration', 0) == 0:
                char = characteristics[filename]
                print(f"  ⚠️  {filename}: No speech detected")
                print(f"     - Amplitude: {char['amplitude']:.4f}")
                print(f"     - Energy: {char['energy']:.6f}")
                print(f"     - RMS: {char['rms']:.4f}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if successful_vad == 0:
            print(f"  🚨 All files failed VAD detection!")
            print(f"  - Check VAD threshold settings")
            print(f"  - Verify audio file format and quality")
            print(f"  - Consider using different VAD parameters")
        elif successful_vad < total_files:
            print(f"  ⚠️  Some files failed VAD detection")
            print(f"  - Review audio characteristics of failed files")
            print(f"  - Adjust VAD parameters for better detection")
        else:
            print(f"  ✅ All files passed VAD detection")
        
        # Save detailed results
        results_file = os.path.join(test_dir, "vad_diagnosis_results.json")
        all_results = {
            'characteristics': characteristics,
            'vad_results': vad_results,
            'parameter_testing': param_results,
            'summary': {
                'total_files': total_files,
                'successful_vad': successful_vad,
                'detection_rate': successful_vad/total_files
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ask if user wants to keep files
        response = input("\nKeep test files for inspection? (y/n): ").lower().strip()
        if response != 'y':
            shutil.rmtree(test_dir)
            print(f"Cleaned up: {test_dir}")

if __name__ == "__main__":
    main() 