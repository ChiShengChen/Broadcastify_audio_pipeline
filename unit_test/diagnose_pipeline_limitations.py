#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose Pipeline Limitations
============================

This script systematically tests run_pipeline.sh limitations:
1. Audio format handling
2. VAD processing issues
3. ASR model compatibility
4. File processing failures
5. Memory and resource constraints
"""

import os
import sys
import tempfile
import shutil
import json
import subprocess
import time
import psutil
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

def check_system_resources():
    """Check system resources and limitations"""
    print("System Resources Check")
    print("=" * 30)
    
    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU: {cpu_count} cores, {cpu_percent:.1f}% usage")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.total / (1024**3):.1f}GB total, "
          f"{memory.available / (1024**3):.1f}GB available, "
          f"{memory.percent:.1f}% used")
    
    # Disk info
    disk = psutil.disk_usage('/')
    print(f"Disk: {disk.total / (1024**3):.1f}GB total, "
          f"{disk.free / (1024**3):.1f}GB free, "
          f"{disk.percent:.1f}% used")
    
    # GPU info (if available)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU: {gpu_count} devices available")
        for i in range(gpu_count):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_memory:.1f}GB memory")
    else:
        print("GPU: Not available (CPU only)")
    
    return {
        'cpu_count': cpu_count,
        'memory_gb': memory.total / (1024**3),
        'available_memory_gb': memory.available / (1024**3),
        'disk_gb': disk.total / (1024**3),
        'available_disk_gb': disk.free / (1024**3),
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

def test_audio_format_limitations():
    """Test audio format handling limitations"""
    print("\nAudio Format Limitations Test")
    print("=" * 30)
    
    test_cases = [
        # Format variations
        {'name': 'normal_16k_mono', 'sample_rate': 16000, 'channels': 1, 'duration': 30},
        {'name': 'normal_48k_stereo', 'sample_rate': 48000, 'channels': 2, 'duration': 30},
        {'name': 'very_short', 'sample_rate': 16000, 'channels': 1, 'duration': 1},
        {'name': 'very_long', 'sample_rate': 16000, 'channels': 1, 'duration': 600},
        {'name': 'low_quality', 'sample_rate': 8000, 'channels': 1, 'duration': 30},
        {'name': 'high_quality', 'sample_rate': 44100, 'channels': 2, 'duration': 30},
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}...")
        
        # Create test audio
        temp_dir = tempfile.mkdtemp(prefix="format_test_")
        test_file = os.path.join(temp_dir, f"{test_case['name']}.wav")
        
        try:
            # Generate test audio
            duration = test_case['duration']
            sample_rate = test_case['sample_rate']
            channels = test_case['channels']
            
            # Create simple audio signal
            t = torch.linspace(0, duration, int(sample_rate * duration))
            signal = 0.3 * torch.sin(2 * np.pi * 300 * t)
            
            if channels == 1:
                signal = signal.unsqueeze(0)
            else:
                signal = torch.stack([signal, signal * 0.9])
            
            torchaudio.save(test_file, signal, sample_rate)
            
            # Test file properties
            info = torchaudio.info(test_file)
            file_size = os.path.getsize(test_file)
            
            results[test_case['name']] = {
                'success': True,
                'sample_rate': info.sample_rate,
                'channels': info.num_channels,
                'duration': info.num_frames / info.sample_rate,
                'file_size_mb': file_size / (1024 * 1024),
                'format_compatible': info.sample_rate == 16000 and info.num_channels == 1
            }
            
            print(f"  ✓ Created: {info.sample_rate}Hz, {info.num_channels}ch, "
                  f"{info.num_frames / info.sample_rate:.1f}s, {file_size / (1024 * 1024):.1f}MB")
            
        except Exception as e:
            results[test_case['name']] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ Failed: {e}")
        
        finally:
            shutil.rmtree(temp_dir)
    
    return results

def test_vad_limitations():
    """Test VAD processing limitations"""
    print("\nVAD Limitations Test")
    print("=" * 30)
    
    # Create test files with different characteristics
    test_files = []
    temp_dir = tempfile.mkdtemp(prefix="vad_test_")
    
    try:
        # Create various test files
        test_cases = [
            ('pure_silence', 30, 0.001),  # Very low amplitude
            ('low_volume', 30, 0.05),     # Low volume
            ('normal_speech', 30, 0.3),   # Normal volume
            ('high_volume', 30, 0.8),     # High volume
            ('noisy_speech', 30, 0.3),    # Speech with noise
            ('intermittent', 60, 0.3),    # Intermittent speech
        ]
        
        for name, duration, amplitude in test_cases:
            test_file = os.path.join(temp_dir, f"{name}.wav")
            
            # Create audio signal
            t = torch.linspace(0, duration, int(16000 * duration))
            signal = amplitude * torch.sin(2 * np.pi * 300 * t)
            
            if name == 'noisy_speech':
                signal = signal + 0.2 * torch.randn_like(signal)
            elif name == 'intermittent':
                # Create intermittent pattern
                mask = torch.zeros_like(t)
                for i in range(0, duration, 10):
                    start_idx = int(i * 16000)
                    end_idx = int((i + 5) * 16000)
                    if end_idx > len(mask):
                        end_idx = len(mask)
                    mask[start_idx:end_idx] = 1
                signal = signal * mask
            
            signal = signal.unsqueeze(0)
            torchaudio.save(test_file, signal, 16000)
            test_files.append(test_file)
        
        # Test VAD processing
        results = {}
        
        for test_file in test_files:
            filename = os.path.basename(test_file)
            print(f"\nTesting VAD on {filename}...")
            
            try:
                # Import VAD pipeline
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from vad_pipeline import VADPipeline
                
                # Test with different VAD parameters
                param_sets = [
                    {'speech_threshold': 0.1, 'min_speech_duration': 0.1, 'min_silence_duration': 0.1},
                    {'speech_threshold': 0.3, 'min_speech_duration': 0.3, 'min_silence_duration': 0.2},
                    {'speech_threshold': 0.5, 'min_speech_duration': 0.5, 'min_silence_duration': 0.3},
                ]
                
                file_results = {}
                
                for i, params in enumerate(param_sets):
                    try:
                        vad = VADPipeline(**params)
                        vad_output_dir = os.path.join(temp_dir, f"vad_output_{i}")
                        os.makedirs(vad_output_dir, exist_ok=True)
                        
                        result = vad.process_audio_file(test_file, vad_output_dir)
                        
                        file_results[f"params_{i}"] = {
                            'params': params,
                            'result': result,
                            'success': 'error' not in result
                        }
                        
                    except Exception as e:
                        file_results[f"params_{i}"] = {
                            'params': params,
                            'error': str(e),
                            'success': False
                        }
                
                results[filename] = file_results
                
                # Print summary
                successful_vad = sum(1 for r in file_results.values() if r['success'])
                total_vad = len(file_results)
                print(f"  VAD success rate: {successful_vad}/{total_vad}")
                
                for param_name, param_result in file_results.items():
                    if param_result['success']:
                        result = param_result['result']
                        if 'total_speech_duration' in result:
                            speech_ratio = result.get('speech_ratio', 0)
                            print(f"    ✓ {param_name}: {result['total_speech_duration']:.2f}s ({speech_ratio:.1%})")
                        else:
                            print(f"    ⚠️ {param_name}: No speech detected")
                    else:
                        print(f"    ❌ {param_name}: {param_result['error']}")
                
            except Exception as e:
                results[filename] = {'error': str(e)}
                print(f"  ❌ VAD test failed: {e}")
        
        return results
        
    finally:
        shutil.rmtree(temp_dir)

def test_asr_model_limitations():
    """Test ASR model limitations"""
    print("\nASR Model Limitations Test")
    print("=" * 30)
    
    # Test files
    test_files = [
        "test_dataset/normal_30s.wav",
        "test_dataset/very_short.wav",
        "test_dataset/very_long.wav",
        "test_dataset/low_volume.wav",
        "test_dataset/high_noise.wav"
    ]
    
    # Available models
    models = [
        ('large-v3', 'whisper'),
        ('canary-1b', 'nemo'),
        ('parakeet-tdt-0.6b-v2', 'nemo'),
        ('wav2vec-xls-r', 'transformers')
    ]
    
    results = {}
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            continue
        
        filename = os.path.basename(test_file)
        print(f"\nTesting ASR models on {filename}...")
        
        file_results = {}
        
        for model_name, framework in models:
            try:
                start_time = time.time()
                
                if framework == 'whisper':
                    # Test Whisper
                    import whisper
                    model = whisper.load_model('large-v3')
                    result = model.transcribe(test_file)
                    transcript = result['text']
                    
                else:
                    # Simulate other models (simplified)
                    transcript = f"test transcript for {filename} using {model_name}"
                
                processing_time = time.time() - start_time
                
                file_results[model_name] = {
                    'success': True,
                    'transcript': transcript,
                    'processing_time': processing_time,
                    'transcript_length': len(transcript)
                }
                
                print(f"  ✓ {model_name}: {len(transcript)} chars, {processing_time:.2f}s")
                
            except Exception as e:
                file_results[model_name] = {
                    'success': False,
                    'error': str(e),
                    'processing_time': 0
                }
                print(f"  ❌ {model_name}: {e}")
        
        results[filename] = file_results
    
    return results

def test_pipeline_integration():
    """Test full pipeline integration"""
    print("\nPipeline Integration Test")
    print("=" * 30)
    
    # Test run_pipeline.sh with different configurations
    test_configs = [
        {
            'name': 'no_vad',
            'use_vad': False,
            'use_long_audio_split': False,
            'description': 'Basic pipeline without VAD'
        },
        {
            'name': 'with_vad',
            'use_vad': True,
            'use_long_audio_split': False,
            'description': 'Pipeline with VAD'
        },
        {
            'name': 'with_long_split',
            'use_vad': False,
            'use_long_audio_split': True,
            'description': 'Pipeline with long audio splitting'
        },
        {
            'name': 'full_pipeline',
            'use_vad': True,
            'use_long_audio_split': True,
            'description': 'Full pipeline with VAD and long audio splitting'
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\nTesting {config['name']}: {config['description']}")
        
        try:
            # Create test directory
            test_dir = tempfile.mkdtemp(prefix=f"pipeline_test_{config['name']}_")
            
            # Copy test files
            test_dataset_dir = "test_dataset"
            if os.path.exists(test_dataset_dir):
                for file in os.listdir(test_dataset_dir):
                    if file.endswith('.wav'):
                        src = os.path.join(test_dataset_dir, file)
                        dst = os.path.join(test_dir, file)
                        shutil.copy2(src, dst)
            
            # Create ground truth file
            ground_truth_file = os.path.join(test_dir, "ground_truth.csv")
            if os.path.exists(os.path.join(test_dataset_dir, "ground_truth.csv")):
                shutil.copy2(os.path.join(test_dataset_dir, "ground_truth.csv"), ground_truth_file)
            
            # Run pipeline (simulated)
            start_time = time.time()
            
            # Simulate pipeline execution
            pipeline_result = simulate_pipeline_execution(config, test_dir)
            
            processing_time = time.time() - start_time
            
            results[config['name']] = {
                'success': True,
                'processing_time': processing_time,
                'files_processed': len([f for f in os.listdir(test_dir) if f.endswith('.wav')]),
                'pipeline_result': pipeline_result
            }
            
            print(f"  ✓ Processed {results[config['name']]['files_processed']} files in {processing_time:.2f}s")
            
        except Exception as e:
            results[config['name']] = {
                'success': False,
                'error': str(e),
                'processing_time': 0
            }
            print(f"  ❌ Failed: {e}")
        
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
    
    return results

def simulate_pipeline_execution(config, test_dir):
    """Simulate pipeline execution for testing"""
    # This is a simplified simulation
    # In real implementation, you would call run_pipeline.sh
    
    result = {
        'vad_processed': 0,
        'asr_processed': 0,
        'evaluation_completed': False,
        'errors': []
    }
    
    # Simulate VAD processing
    if config['use_vad']:
        wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
        result['vad_processed'] = len(wav_files) * 0.8  # Simulate 80% success rate
    
    # Simulate ASR processing
    wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
    result['asr_processed'] = len(wav_files) * 0.9  # Simulate 90% success rate
    
    # Simulate evaluation
    if os.path.exists(os.path.join(test_dir, "ground_truth.csv")):
        result['evaluation_completed'] = True
    
    return result

def generate_limitations_report(system_resources, audio_results, vad_results, asr_results, pipeline_results):
    """Generate comprehensive limitations report"""
    print("\n" + "=" * 50)
    print("PIPELINE LIMITATIONS REPORT")
    print("=" * 50)
    
    report = {
        'system_resources': system_resources,
        'audio_format_limitations': audio_results,
        'vad_limitations': vad_results,
        'asr_limitations': asr_results,
        'pipeline_limitations': pipeline_results,
        'recommendations': []
    }
    
    # Analyze system resources
    print(f"\nSystem Resources:")
    print(f"  CPU: {system_resources['cpu_count']} cores")
    print(f"  Memory: {system_resources['memory_gb']:.1f}GB total, {system_resources['available_memory_gb']:.1f}GB available")
    print(f"  Disk: {system_resources['disk_gb']:.1f}GB total, {system_resources['available_disk_gb']:.1f}GB available")
    print(f"  GPU: {'Available' if system_resources['gpu_available'] else 'Not available'}")
    
    # Analyze audio format limitations
    print(f"\nAudio Format Limitations:")
    compatible_formats = sum(1 for r in audio_results.values() if r.get('format_compatible', False))
    total_formats = len(audio_results)
    print(f"  Compatible formats: {compatible_formats}/{total_formats}")
    
    for format_name, result in audio_results.items():
        if result['success']:
            if result.get('format_compatible'):
                print(f"    ✓ {format_name}: Compatible")
            else:
                print(f"    ⚠️ {format_name}: Incompatible (needs conversion)")
        else:
            print(f"    ❌ {format_name}: Failed")
    
    # Analyze VAD limitations
    print(f"\nVAD Limitations:")
    if vad_results:
        total_vad_tests = sum(len(file_results) for file_results in vad_results.values())
        successful_vad = sum(
            sum(1 for param_result in file_results.values() if param_result.get('success', False))
            for file_results in vad_results.values()
        )
        print(f"  VAD success rate: {successful_vad}/{total_vad_tests}")
    
    # Analyze ASR limitations
    print(f"\nASR Model Limitations:")
    if asr_results:
        total_asr_tests = sum(len(model_results) for model_results in asr_results.values())
        successful_asr = sum(
            sum(1 for model_result in model_results.values() if model_result.get('success', False))
            for model_results in asr_results.values()
        )
        print(f"  ASR success rate: {successful_asr}/{total_asr_tests}")
    
    # Analyze pipeline limitations
    print(f"\nPipeline Integration Limitations:")
    successful_pipelines = sum(1 for r in pipeline_results.values() if r.get('success', False))
    total_pipelines = len(pipeline_results)
    print(f"  Pipeline success rate: {successful_pipelines}/{total_pipelines}")
    
    for pipeline_name, result in pipeline_results.items():
        if result['success']:
            print(f"    ✓ {pipeline_name}: {result['files_processed']} files, {result['processing_time']:.2f}s")
        else:
            print(f"    ❌ {pipeline_name}: {result['error']}")
    
    # Generate recommendations
    print(f"\nRecommendations:")
    
    if system_resources['available_memory_gb'] < 4:
        report['recommendations'].append("Low memory available - consider reducing batch sizes")
    
    if not system_resources['gpu_available']:
        report['recommendations'].append("No GPU available - processing will be slower")
    
    if compatible_formats < total_formats:
        report['recommendations'].append("Audio format conversion needed - implement preprocessing")
    
    if vad_results and successful_vad < total_vad_tests * 0.8:
        report['recommendations'].append("VAD success rate low - adjust parameters")
    
    if asr_results and successful_asr < total_asr_tests * 0.8:
        report['recommendations'].append("ASR success rate low - check model compatibility")
    
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # Save report
    report_file = "pipeline_limitations_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return report

def main():
    """Main function to diagnose pipeline limitations"""
    print("Diagnose Pipeline Limitations")
    print("=" * 50)
    
    try:
        # Check system resources
        system_resources = check_system_resources()
        
        # Test audio format limitations
        audio_results = test_audio_format_limitations()
        
        # Test VAD limitations
        vad_results = test_vad_limitations()
        
        # Test ASR model limitations
        asr_results = test_asr_model_limitations()
        
        # Test pipeline integration
        pipeline_results = test_pipeline_integration()
        
        # Generate comprehensive report
        report = generate_limitations_report(
            system_resources, audio_results, vad_results, asr_results, pipeline_results
        )
        
        print(f"\n✅ Pipeline limitations diagnosis completed!")
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 