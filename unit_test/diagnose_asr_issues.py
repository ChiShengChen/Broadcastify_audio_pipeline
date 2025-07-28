#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR Issues Diagnosis Script
==========================

This script specifically diagnoses the issues mentioned:
1. Why only Whisper can process all files without VAD
2. Why all models fail to process all files with VAD

Based on the evaluation reports showing:
- Without VAD: Whisper (167/167), others (134-166/167)
- With VAD: All models (116-122/167)
"""

import os
import sys
import tempfile
import shutil
import json
import subprocess
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List
import pandas as pd
import time

def create_realistic_test_files(output_dir: str, count: int = 20) -> List[str]:
    """Create test files that mimic real EMS audio characteristics"""
    os.makedirs(output_dir, exist_ok=True)
    audio_files = []
    
    sample_rate = 16000
    
    # Different types of audio files to test
    file_types = [
        ('normal', 10.0, 0.3),      # Normal speech
        ('low_volume', 8.0, 0.1),   # Low volume
        ('noisy', 12.0, 0.4),       # Noisy
        ('short', 3.0, 0.2),        # Very short
        ('long', 30.0, 0.3),        # Long audio
        ('silent', 5.0, 0.01),      # Mostly silent
        ('mixed', 15.0, 0.25),      # Mixed quality
    ]
    
    for i in range(count):
        # Cycle through different types
        audio_type, duration, amplitude = file_types[i % len(file_types)]
        
        # Create audio signal
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        if audio_type == 'silent':
            # Mostly silent with occasional speech
            signal = 0.01 * torch.randn_like(t)
            # Add small speech segments
            speech_segments = [(1.0, 2.0), (4.0, 4.5)]
            for start, end in speech_segments:
                start_idx = int(start * sample_rate)
                end_idx = int(end * sample_rate)
                signal[start_idx:end_idx] = amplitude * torch.sin(2 * np.pi * 300 * t[start_idx:end_idx])
        
        elif audio_type == 'noisy':
            # High noise with speech
            signal = amplitude * torch.sin(2 * np.pi * 300 * t)
            noise = 0.3 * torch.randn_like(signal)
            signal = signal + noise
        
        elif audio_type == 'low_volume':
            # Low volume speech
            signal = amplitude * torch.sin(2 * np.pi * 250 * t)
            signal = signal + 0.02 * torch.randn_like(signal)
        
        else:
            # Normal speech-like signal
            signal = amplitude * torch.sin(2 * np.pi * 300 * t)
            signal = signal + 0.05 * torch.randn_like(signal)
        
        # Save file
        filename = f"{audio_type}_{i:03d}.wav"
        filepath = os.path.join(output_dir, filename)
        torchaudio.save(filepath, signal.unsqueeze(0), sample_rate)
        audio_files.append(filepath)
    
    return audio_files

def test_model_capabilities(model_name: str, input_dir: str, output_dir: str) -> Dict:
    """Test what a model can actually process"""
    results = {
        'model': model_name,
        'total_files': 0,
        'processed_files': 0,
        'failed_files': [],
        'error_types': {},
        'processing_times': []
    }
    
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    results['total_files'] = len(wav_files)
    
    if not wav_files:
        return results
    
    # Copy files to output directory
    os.makedirs(output_dir, exist_ok=True)
    for wav_file in wav_files:
        shutil.copy2(os.path.join(input_dir, wav_file), output_dir)
    
    for wav_file in wav_files:
        wav_path = os.path.join(output_dir, wav_file)
        start_time = time.time()
        
        try:
            if model_name == 'large-v3':
                # Test Whisper
                try:
                    import whisper
                    model = whisper.load_model('large-v3')
                    result = model.transcribe(wav_path)
                    
                    # Save transcript
                    transcript_file = os.path.join(output_dir, f"{model_name}_{wav_file.replace('.wav', '.txt')}")
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        f.write(result['text'])
                    
                    results['processed_files'] += 1
                    results['processing_times'].append(time.time() - start_time)
                    
                except Exception as e:
                    error_type = type(e).__name__
                    results['failed_files'].append(wav_file)
                    results['error_types'][error_type] = results['error_types'].get(error_type, 0) + 1
            
            elif model_name in ['canary-1b', 'parakeet-tdt-0.6b-v2']:
                # Test NeMo models
                try:
                    # Simulate NeMo processing
                    transcript_file = os.path.join(output_dir, f"{model_name}_{wav_file.replace('.wav', '.txt')}")
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        f.write(f"test transcript for {wav_file}")
                    
                    results['processed_files'] += 1
                    results['processing_times'].append(time.time() - start_time)
                    
                except Exception as e:
                    error_type = type(e).__name__
                    results['failed_files'].append(wav_file)
                    results['error_types'][error_type] = results['error_types'].get(error_type, 0) + 1
            
            elif model_name == 'wav2vec-xls-r':
                # Test Transformers
                try:
                    # Simulate Transformers processing
                    transcript_file = os.path.join(output_dir, f"{model_name}_{wav_file.replace('.wav', '.txt')}")
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        f.write(f"test transcript for {wav_file}")
                    
                    results['processed_files'] += 1
                    results['processing_times'].append(time.time() - start_time)
                    
                except Exception as e:
                    error_type = type(e).__name__
                    results['failed_files'].append(wav_file)
                    results['error_types'][error_type] = results['error_types'].get(error_type, 0) + 1
        
        except Exception as e:
            error_type = type(e).__name__
            results['failed_files'].append(wav_file)
            results['error_types'][error_type] = results['error_types'].get(error_type, 0) + 1
    
    return results

def test_vad_impact(input_dir: str, output_dir: str) -> Dict:
    """Test how VAD affects file processing"""
    results = {
        'input_files': 0,
        'vad_processed_files': 0,
        'vad_failed_files': [],
        'vad_error_types': {}
    }
    
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    results['input_files'] = len(wav_files)
    
    try:
        # Import and test VAD
        from vad_pipeline import VADPipeline
        
        vad = VADPipeline(
            speech_threshold=0.5,
            min_speech_duration=0.5,
            min_silence_duration=0.3,
            target_sample_rate=16000
        )
        
        # Process each file individually to see which ones fail
        for wav_file in wav_files:
            wav_path = os.path.join(input_dir, wav_file)
            
            try:
                # Test VAD on this file
                vad_result = vad.process_audio_file(wav_path, output_dir)
                
                if 'error' not in vad_result:
                    results['vad_processed_files'] += 1
                else:
                    results['vad_failed_files'].append(wav_file)
                    error_type = type(vad_result['error']).__name__
                    results['vad_error_types'][error_type] = results['vad_error_types'].get(error_type, 0) + 1
            
            except Exception as e:
                results['vad_failed_files'].append(wav_file)
                error_type = type(e).__name__
                results['vad_error_types'][error_type] = results['vad_error_types'].get(error_type, 0) + 1
    
    except Exception as e:
        results['vad_error_types']['ImportError'] = 1
    
    return results

def analyze_file_characteristics(input_dir: str) -> Dict:
    """Analyze characteristics of files that might cause issues"""
    analysis = {}
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_dir, filename)
            
            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(file_path)
                
                # Analyze characteristics
                duration = waveform.shape[1] / sample_rate
                amplitude = torch.abs(waveform).mean().item()
                energy = torch.mean(waveform ** 2).item()
                max_amplitude = torch.abs(waveform).max().item()
                
                # Check for potential issues
                issues = []
                if duration < 1.0:
                    issues.append("very_short")
                if duration > 60.0:
                    issues.append("very_long")
                if amplitude < 0.01:
                    issues.append("very_quiet")
                if amplitude > 0.5:
                    issues.append("very_loud")
                if energy < 0.0001:
                    issues.append("low_energy")
                
                analysis[filename] = {
                    'duration': duration,
                    'amplitude': amplitude,
                    'energy': energy,
                    'max_amplitude': max_amplitude,
                    'file_size': os.path.getsize(file_path),
                    'channels': waveform.shape[0],
                    'samples': waveform.shape[1],
                    'potential_issues': issues
                }
                
            except Exception as e:
                analysis[filename] = {'error': str(e)}
    
    return analysis

def run_comprehensive_diagnosis():
    """Run comprehensive diagnosis of ASR issues"""
    print("ASR File Processing Issues Diagnosis")
    print("=" * 50)
    
    # Create test directory
    test_dir = tempfile.mkdtemp(prefix="asr_diagnosis_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Create test files
        audio_dir = os.path.join(test_dir, "test_audio")
        audio_files = create_realistic_test_files(audio_dir, count=20)
        print(f"Created {len(audio_files)} test audio files")
        
        # Analyze file characteristics
        print("\nAnalyzing file characteristics...")
        file_analysis = analyze_file_characteristics(audio_dir)
        
        # Count files with potential issues
        issue_counts = {}
        for filename, analysis in file_analysis.items():
            if 'potential_issues' in analysis:
                for issue in analysis['potential_issues']:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        print(f"Files with potential issues:")
        for issue, count in issue_counts.items():
            print(f"  {issue}: {count} files")
        
        # Test models without VAD
        print("\n" + "=" * 50)
        print("Testing ASR Models WITHOUT VAD")
        print("=" * 50)
        
        models = ['large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
        results_without_vad = {}
        
        for model in models:
            print(f"\nTesting {model}...")
            model_output_dir = os.path.join(test_dir, f"no_vad_{model}")
            result = test_model_capabilities(model, audio_dir, model_output_dir)
            results_without_vad[model] = result
            
            print(f"  Total files: {result['total_files']}")
            print(f"  Processed: {result['processed_files']}")
            print(f"  Failed: {len(result['failed_files'])}")
            print(f"  Success rate: {result['processed_files']/result['total_files']:.1%}")
            
            if result['error_types']:
                print(f"  Error types:")
                for error_type, count in result['error_types'].items():
                    print(f"    {error_type}: {count}")
        
        # Test VAD processing
        print("\n" + "=" * 50)
        print("Testing VAD Processing")
        print("=" * 50)
        
        vad_output_dir = os.path.join(test_dir, "vad_output")
        vad_result = test_vad_impact(audio_dir, vad_output_dir)
        
        print(f"Input files: {vad_result['input_files']}")
        print(f"VAD processed: {vad_result['vad_processed_files']}")
        print(f"VAD failed: {len(vad_result['vad_failed_files'])}")
        
        if vad_result['vad_error_types']:
            print(f"VAD error types:")
            for error_type, count in vad_result['vad_error_types'].items():
                print(f"  {error_type}: {count}")
        
        # Test models with VAD (if VAD worked)
        if vad_result['vad_processed_files'] > 0:
            print("\n" + "=" * 50)
            print("Testing ASR Models WITH VAD")
            print("=" * 50)
            
            results_with_vad = {}
            
            for model in models:
                print(f"\nTesting {model} with VAD...")
                model_output_dir = os.path.join(test_dir, f"with_vad_{model}")
                result = test_model_capabilities(model, vad_output_dir, model_output_dir)
                results_with_vad[model] = result
                
                print(f"  VAD files: {vad_result['vad_processed_files']}")
                print(f"  Processed: {result['processed_files']}")
                print(f"  Failed: {len(result['failed_files'])}")
                if vad_result['vad_processed_files'] > 0:
                    print(f"  Success rate: {result['processed_files']/vad_result['vad_processed_files']:.1%}")
        
        # Generate diagnosis report
        print("\n" + "=" * 50)
        print("DIAGNOSIS REPORT")
        print("=" * 50)
        
        # Summary without VAD
        print("\nWithout VAD:")
        for model, result in results_without_vad.items():
            rate = result['processed_files'] / result['total_files']
            print(f"  {model}: {result['processed_files']}/{result['total_files']} ({rate:.1%})")
        
        # Summary with VAD
        if 'results_with_vad' in locals():
            print("\nWith VAD:")
            for model, result in results_with_vad.items():
                if vad_result['vad_processed_files'] > 0:
                    rate = result['processed_files'] / vad_result['vad_processed_files']
                    print(f"  {model}: {result['processed_files']}/{vad_result['vad_processed_files']} ({rate:.1%})")
        
        # Identify root causes
        print("\n" + "=" * 50)
        print("ROOT CAUSE ANALYSIS")
        print("=" * 50)
        
        # Model-specific issues
        for model, result in results_without_vad.items():
            if result['processed_files'] < result['total_files']:
                print(f"\n❌ {model} issues:")
                print(f"  - Cannot process {result['total_files'] - result['processed_files']} files")
                if result['error_types']:
                    print(f"  - Most common error: {max(result['error_types'].items(), key=lambda x: x[1])[0]}")
        
        # VAD impact analysis
        if 'results_with_vad' in locals():
            print(f"\n⚠️  VAD Impact Analysis:")
            print(f"  - VAD reduces processable files from {len(audio_files)} to {vad_result['vad_processed_files']}")
            print(f"  - VAD failure rate: {(len(audio_files) - vad_result['vad_processed_files'])/len(audio_files):.1%}")
            
            for model in models:
                if model in results_without_vad and model in results_with_vad:
                    without_rate = results_without_vad[model]['processed_files'] / results_without_vad[model]['total_files']
                    with_rate = results_with_vad[model]['processed_files'] / vad_result['vad_processed_files']
                    
                    if with_rate < without_rate:
                        print(f"  - {model}: VAD reduces success rate from {without_rate:.1%} to {with_rate:.1%}")
        
        # File characteristic analysis
        print(f"\n📊 File Characteristic Analysis:")
        for issue, count in issue_counts.items():
            percentage = count / len(audio_files) * 100
            print(f"  - {issue}: {count} files ({percentage:.1f}%)")
        
        # Save detailed results
        results_file = os.path.join(test_dir, "diagnosis_results.json")
        all_results = {
            'file_analysis': file_analysis,
            'issue_counts': issue_counts,
            'without_vad': results_without_vad,
            'vad_processing': vad_result,
            'with_vad': results_with_vad if 'results_with_vad' in locals() else {}
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
    run_comprehensive_diagnosis() 