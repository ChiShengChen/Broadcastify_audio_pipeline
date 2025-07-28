#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Test Dataset
==================

Generate minimal test dataset to reproduce ASR pipeline issues:
1. Normal speech files
2. Long audio files with different formats
3. Silence and noise files
4. Edge cases for VAD and ASR testing
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
import librosa
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

def create_test_audio_files(output_dir: str):
    """Create comprehensive test audio files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Test scenarios
    test_cases = [
        # Normal speech files
        {
            'name': 'normal_30s',
            'duration': 30.0,
            'sample_rate': 16000,
            'channels': 1,
            'type': 'speech_like',
            'description': 'Normal speech-like audio, 30s, 16kHz mono'
        },
        {
            'name': 'normal_180s',
            'duration': 180.0,
            'sample_rate': 48000,
            'channels': 2,
            'type': 'speech_like',
            'description': 'Long audio, 3min, 48kHz stereo - tests format handling'
        },
        
        # Silence and noise files
        {
            'name': 'silence_30s',
            'duration': 30.0,
            'sample_rate': 16000,
            'channels': 1,
            'type': 'silence',
            'description': 'Pure silence, 30s, 16kHz mono'
        },
        {
            'name': 'silence_noise',
            'duration': 30.0,
            'sample_rate': 16000,
            'channels': 1,
            'type': 'low_noise',
            'description': 'Low noise (-30dB), tests VAD sensitivity'
        },
        
        # Complex patterns
        {
            'name': 'speech_gap',
            'duration': 60.0,
            'sample_rate': 16000,
            'channels': 1,
            'type': 'speech_gap',
            'description': '15s speech + 30s silence + 15s speech'
        },
        {
            'name': 'intermittent_speech',
            'duration': 45.0,
            'sample_rate': 16000,
            'channels': 1,
            'type': 'intermittent',
            'description': 'Intermittent speech with gaps'
        },
        
        # Edge cases
        {
            'name': 'very_short',
            'duration': 2.0,
            'sample_rate': 16000,
            'channels': 1,
            'type': 'speech_like',
            'description': 'Very short audio, 2s'
        },
        {
            'name': 'very_long',
            'duration': 300.0,
            'sample_rate': 16000,
            'channels': 1,
            'type': 'speech_like',
            'description': 'Very long audio, 5min'
        },
        {
            'name': 'low_volume',
            'duration': 30.0,
            'sample_rate': 16000,
            'channels': 1,
            'type': 'low_volume',
            'description': 'Low volume speech'
        },
        {
            'name': 'high_noise',
            'duration': 30.0,
            'sample_rate': 16000,
            'channels': 1,
            'type': 'high_noise',
            'description': 'High noise with speech'
        }
    ]
    
    created_files = []
    
    for test_case in test_cases:
        print(f"Creating {test_case['name']}: {test_case['description']}")
        
        file_path = create_audio_file(test_case, output_dir)
        created_files.append({
            'path': file_path,
            'metadata': test_case
        })
    
    # Create metadata file
    metadata_file = os.path.join(output_dir, "test_dataset_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(created_files, f, indent=2, ensure_ascii=False)
    
    print(f"\nCreated {len(created_files)} test files")
    print(f"Metadata saved to: {metadata_file}")
    
    return created_files

def create_audio_file(test_case, output_dir):
    """Create a specific audio file based on test case"""
    name = test_case['name']
    duration = test_case['duration']
    sample_rate = test_case['sample_rate']
    channels = test_case['channels']
    audio_type = test_case['type']
    
    # Create base signal
    if audio_type == 'speech_like':
        signal = create_speech_like_signal(duration, sample_rate)
    elif audio_type == 'silence':
        signal = create_silence_signal(duration, sample_rate)
    elif audio_type == 'low_noise':
        signal = create_low_noise_signal(duration, sample_rate)
    elif audio_type == 'speech_gap':
        signal = create_speech_gap_signal(duration, sample_rate)
    elif audio_type == 'intermittent':
        signal = create_intermittent_signal(duration, sample_rate)
    elif audio_type == 'low_volume':
        signal = create_low_volume_signal(duration, sample_rate)
    elif audio_type == 'high_noise':
        signal = create_high_noise_signal(duration, sample_rate)
    else:
        raise ValueError(f"Unknown audio type: {audio_type}")
    
    # Handle channels
    if channels == 1:
        signal = signal.unsqueeze(0)  # Add channel dimension
    elif channels == 2:
        # Create stereo by duplicating and adding slight variation
        signal = torch.stack([signal, signal * 0.9 + 0.1 * torch.randn_like(signal)])
    
    # Save file
    filename = f"{name}.wav"
    filepath = os.path.join(output_dir, filename)
    torchaudio.save(filepath, signal, sample_rate)
    
    return filepath

def create_speech_like_signal(duration, sample_rate):
    """Create speech-like signal with multiple frequencies"""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Create speech-like signal with harmonics
    signal = (
        0.3 * torch.sin(2 * np.pi * 200 * t) +   # Fundamental frequency
        0.2 * torch.sin(2 * np.pi * 400 * t) +   # First harmonic
        0.1 * torch.sin(2 * np.pi * 600 * t) +   # Second harmonic
        0.05 * torch.sin(2 * np.pi * 800 * t)    # Third harmonic
    )
    
    # Add amplitude modulation to simulate speech
    am = 0.5 + 0.5 * torch.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
    signal = signal * am
    
    # Add small noise
    signal = signal + 0.02 * torch.randn_like(signal)
    
    return signal

def create_silence_signal(duration, sample_rate):
    """Create pure silence signal"""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    return 0.001 * torch.randn_like(t)  # Very low noise

def create_low_noise_signal(duration, sample_rate):
    """Create low noise signal (-30dB)"""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    return 0.01 * torch.randn_like(t)  # Low noise

def create_speech_gap_signal(duration, sample_rate):
    """Create speech-gap-speech pattern"""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    signal = torch.zeros_like(t)
    
    # First speech segment (0-15s)
    speech1_mask = t < 15.0
    signal[speech1_mask] = create_speech_like_signal(15.0, sample_rate)[:int(15.0 * sample_rate)]
    
    # Silence gap (15-45s)
    silence_mask = (t >= 15.0) & (t < 45.0)
    signal[silence_mask] = 0.001 * torch.randn_like(signal[silence_mask])
    
    # Second speech segment (45-60s)
    speech2_mask = t >= 45.0
    speech2_signal = create_speech_like_signal(15.0, sample_rate)
    signal[speech2_mask] = speech2_signal[:len(signal[speech2_mask])]
    
    return signal

def create_intermittent_signal(duration, sample_rate):
    """Create intermittent speech with gaps"""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    signal = torch.zeros_like(t)
    
    # Create intermittent pattern
    for i in range(0, int(duration), 10):
        start_idx = int(i * sample_rate)
        end_idx = int((i + 5) * sample_rate)
        if end_idx > len(signal):
            end_idx = len(signal)
        
        if start_idx < len(signal):
            speech_segment = create_speech_like_signal(5.0, sample_rate)
            signal[start_idx:end_idx] = speech_segment[:end_idx-start_idx]
    
    return signal

def create_low_volume_signal(duration, sample_rate):
    """Create low volume speech"""
    signal = create_speech_like_signal(duration, sample_rate)
    return 0.1 * signal  # Reduce volume

def create_high_noise_signal(duration, sample_rate):
    """Create high noise with speech"""
    signal = create_speech_like_signal(duration, sample_rate)
    noise = 0.3 * torch.randn_like(signal)
    return signal + noise

def create_ground_truth_file(audio_files, output_dir):
    """Create ground truth CSV file for testing"""
    ground_truth_data = []
    
    for audio_file in audio_files:
        filename = os.path.basename(audio_file['path'])
        # Create simple transcript for testing
        transcript = f"test transcript for {filename}"
        ground_truth_data.append({
            'Filename': filename,
            'transcript': transcript
        })
    
    ground_truth_file = os.path.join(output_dir, "ground_truth.csv")
    
    import pandas as pd
    df = pd.DataFrame(ground_truth_data)
    df.to_csv(ground_truth_file, index=False)
    
    print(f"Ground truth file created: {ground_truth_file}")
    return ground_truth_file

def main():
    """Main function to create test dataset"""
    print("Create Test Dataset")
    print("=" * 40)
    
    # Create test directory
    test_dir = "test_dataset"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Create test audio files
        audio_files = create_test_audio_files(test_dir)
        
        # Create ground truth file
        ground_truth_file = create_ground_truth_file(audio_files, test_dir)
        
        # Create test configuration
        config = {
            'test_files': len(audio_files),
            'ground_truth_file': ground_truth_file,
            'test_scenarios': [
                'normal_30s - Normal speech, 30s, 16kHz mono',
                'normal_180s - Long audio, 3min, 48kHz stereo',
                'silence_30s - Pure silence, 30s',
                'silence_noise - Low noise, tests VAD sensitivity',
                'speech_gap - Speech-silence-speech pattern',
                'intermittent_speech - Intermittent speech with gaps',
                'very_short - Very short audio, 2s',
                'very_long - Very long audio, 5min',
                'low_volume - Low volume speech',
                'high_noise - High noise with speech'
            ]
        }
        
        config_file = os.path.join(test_dir, "test_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest dataset created successfully!")
        print(f"Directory: {test_dir}")
        print(f"Files: {len(audio_files)}")
        print(f"Ground truth: {ground_truth_file}")
        print(f"Config: {config_file}")
        
        print(f"\nTest scenarios:")
        for scenario in config['test_scenarios']:
            print(f"  - {scenario}")
        
    except Exception as e:
        print(f"Failed to create test dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 