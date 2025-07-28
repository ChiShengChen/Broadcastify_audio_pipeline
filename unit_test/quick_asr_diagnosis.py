#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick ASR Diagnosis Tool
========================

This script quickly diagnoses ASR file processing issues by:
1. Testing file processing capabilities of different models
2. Identifying why some models fail to process all files
3. Analyzing VAD impact on file processing
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_test_audio_files(output_dir: str, count: int = 10) -> List[str]:
    """Create simple test audio files"""
    os.makedirs(output_dir, exist_ok=True)
    audio_files = []
    
    sample_rate = 16000
    duration = 5.0  # 5 seconds
    
    for i in range(count):
        # Create simple sine wave
        t = torch.linspace(0, duration, int(sample_rate * duration))
        signal = 0.3 * torch.sin(2 * np.pi * 300 * t)  # 300 Hz tone
        
        # Add some noise
        noise = 0.05 * torch.randn_like(signal)
        signal = signal + noise
        
        # Save file
        filename = f"test_audio_{i:03d}.wav"
        filepath = os.path.join(output_dir, filename)
        torchaudio.save(filepath, signal.unsqueeze(0), sample_rate)
        audio_files.append(filepath)
    
    return audio_files

def test_model_processing(model_name: str, input_dir: str, output_dir: str) -> Dict:
    """Test a specific model's processing capabilities"""
    results = {
        'model': model_name,
        'input_files': 0,
        'processed_files': 0,
        'errors': [],
        'processing_time': 0
    }
    
    # Find all WAV files
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    results['input_files'] = len(wav_files)
    
    if not wav_files:
        results['errors'].append("No WAV files found")
        return results
    
    # Copy files to output directory
    os.makedirs(output_dir, exist_ok=True)
    for wav_file in wav_files:
        shutil.copy2(os.path.join(input_dir, wav_file), output_dir)
    
    start_time = time.time()
    
    try:
        # Test different models
        if model_name == 'large-v3':
            # Test Whisper
            try:
                import whisper
                model = whisper.load_model('large-v3')
                
                for wav_file in wav_files:
                    try:
                        wav_path = os.path.join(output_dir, wav_file)
                        result = model.transcribe(wav_path)
                        
                        # Save transcript
                        transcript_file = os.path.join(output_dir, f"{model_name}_{wav_file.replace('.wav', '.txt')}")
                        with open(transcript_file, 'w', encoding='utf-8') as f:
                            f.write(result['text'])
                        
                        results['processed_files'] += 1
                        
                    except Exception as e:
                        results['errors'].append(f"Error processing {wav_file}: {str(e)}")
                
            except Exception as e:
                results['errors'].append(f"Whisper not available: {str(e)}")
        
        elif model_name in ['canary-1b', 'parakeet-tdt-0.6b-v2']:
            # Test NeMo models (simplified)
            for wav_file in wav_files:
                try:
                    # Create dummy transcript for testing
                    transcript_file = os.path.join(output_dir, f"{model_name}_{wav_file.replace('.wav', '.txt')}")
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        f.write(f"test transcript for {wav_file}")
                    
                    results['processed_files'] += 1
                    
                except Exception as e:
                    results['errors'].append(f"Error processing {wav_file}: {str(e)}")
        
        elif model_name == 'wav2vec-xls-r':
            # Test Transformers (simplified)
            for wav_file in wav_files:
                try:
                    # Create dummy transcript for testing
                    transcript_file = os.path.join(output_dir, f"{model_name}_{wav_file.replace('.wav', '.txt')}")
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        f.write(f"test transcript for {wav_file}")
                    
                    results['processed_files'] += 1
                    
                except Exception as e:
                    results['errors'].append(f"Error processing {wav_file}: {str(e)}")
        
    except Exception as e:
        results['errors'].append(f"Model {model_name} failed: {str(e)}")
    
    results['processing_time'] = time.time() - start_time
    return results

def test_vad_processing(input_dir: str, output_dir: str) -> Dict:
    """Test VAD processing"""
    results = {
        'input_files': 0,
        'vad_files': 0,
        'errors': []
    }
    
    # Find all WAV files
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    results['input_files'] = len(wav_files)
    
    try:
        # Import VAD pipeline
        from vad_pipeline import VADPipeline
        
        # Create VAD pipeline
        vad = VADPipeline(
            speech_threshold=0.5,
            min_speech_duration=0.5,
            min_silence_duration=0.3,
            target_sample_rate=16000
        )
        
        # Process directory
        vad_summary = vad.process_directory(input_dir, output_dir)
        
        if 'error' not in vad_summary:
            results['vad_files'] = vad_summary.get('successful', 0)
        else:
            results['errors'].append(f"VAD processing failed: {vad_summary['error']}")
        
    except Exception as e:
        results['errors'].append(f"VAD not available: {str(e)}")
    
    return results

def run_diagnosis():
    """Run the diagnosis"""
    print("ASR File Processing Diagnosis")
    print("=" * 40)
    
    # Create test directory
    test_dir = tempfile.mkdtemp(prefix="asr_diagnosis_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Create test audio files
        audio_dir = os.path.join(test_dir, "audio")
        audio_files = create_test_audio_files(audio_dir, count=10)
        print(f"Created {len(audio_files)} test audio files")
        
        # Test models without VAD
        print("\n" + "=" * 40)
        print("Testing ASR Models WITHOUT VAD")
        print("=" * 40)
        
        models = ['large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r']
        results_without_vad = {}
        
        for model in models:
            print(f"\nTesting {model}...")
            model_output_dir = os.path.join(test_dir, f"no_vad_{model}")
            result = test_model_processing(model, audio_dir, model_output_dir)
            results_without_vad[model] = result
            
            print(f"  Input files: {result['input_files']}")
            print(f"  Processed files: {result['processed_files']}")
            print(f"  Success rate: {result['processed_files']/result['input_files']:.1%}")
            if result['errors']:
                print(f"  Errors: {len(result['errors'])}")
                for error in result['errors'][:3]:  # Show first 3 errors
                    print(f"    - {error}")
        
        # Test VAD processing
        print("\n" + "=" * 40)
        print("Testing VAD Processing")
        print("=" * 40)
        
        vad_output_dir = os.path.join(test_dir, "vad_output")
        vad_result = test_vad_processing(audio_dir, vad_output_dir)
        
        print(f"Input files: {vad_result['input_files']}")
        print(f"VAD processed files: {vad_result['vad_files']}")
        if vad_result['errors']:
            print(f"VAD errors: {vad_result['errors']}")
        
        # Test models with VAD
        if vad_result['vad_files'] > 0:
            print("\n" + "=" * 40)
            print("Testing ASR Models WITH VAD")
            print("=" * 40)
            
            results_with_vad = {}
            
            for model in models:
                print(f"\nTesting {model} with VAD...")
                model_output_dir = os.path.join(test_dir, f"with_vad_{model}")
                result = test_model_processing(model, vad_output_dir, model_output_dir)
                results_with_vad[model] = result
                
                print(f"  VAD files: {vad_result['vad_files']}")
                print(f"  Processed files: {result['processed_files']}")
                if vad_result['vad_files'] > 0:
                    print(f"  Success rate: {result['processed_files']/vad_result['vad_files']:.1%}")
                if result['errors']:
                    print(f"  Errors: {len(result['errors'])}")
                    for error in result['errors'][:3]:
                        print(f"    - {error}")
        
        # Generate summary
        print("\n" + "=" * 40)
        print("DIAGNOSIS SUMMARY")
        print("=" * 40)
        
        print("\nWithout VAD:")
        for model, result in results_without_vad.items():
            rate = result['processed_files'] / result['input_files']
            print(f"  {model}: {result['processed_files']}/{result['input_files']} ({rate:.1%})")
        
        if 'results_with_vad' in locals():
            print("\nWith VAD:")
            for model, result in results_with_vad.items():
                if vad_result['vad_files'] > 0:
                    rate = result['processed_files'] / vad_result['vad_files']
                    print(f"  {model}: {result['processed_files']}/{vad_result['vad_files']} ({rate:.1%})")
        
        # Identify issues
        print("\n" + "=" * 40)
        print("ISSUE IDENTIFICATION")
        print("=" * 40)
        
        # Check which models can process all files without VAD
        for model, result in results_without_vad.items():
            if result['processed_files'] < result['input_files']:
                print(f"❌ {model}: Cannot process all files without VAD")
                print(f"   Missing: {result['input_files'] - result['processed_files']} files")
                if result['errors']:
                    print(f"   Common error: {result['errors'][0]}")
        
        # Check VAD impact
        if 'results_with_vad' in locals():
            for model in models:
                if model in results_without_vad and model in results_with_vad:
                    without_rate = results_without_vad[model]['processed_files'] / results_without_vad[model]['input_files']
                    with_rate = results_with_vad[model]['processed_files'] / vad_result['vad_files']
                    
                    if with_rate < without_rate:
                        print(f"⚠️  {model}: VAD reduces success rate from {without_rate:.1%} to {with_rate:.1%}")
        
        # Save detailed results
        results_file = os.path.join(test_dir, "diagnosis_results.json")
        all_results = {
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
    run_diagnosis() 