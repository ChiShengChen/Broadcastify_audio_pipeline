#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Data Generator for ASR Pipeline
====================================

This script generates test audio files with various characteristics
to test the integrated ASR pipeline with audio preprocessing.
"""

import os
import sys
import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
import random
import string

def generate_synthetic_speech(duration, sample_rate=16000, volume=0.1):
    """Generate synthetic speech-like audio"""
    # Generate random audio that mimics speech characteristics
    samples = int(duration * sample_rate)
    
    # Create a more speech-like signal with varying frequencies
    t = np.linspace(0, duration, samples)
    
    # Base frequency components (speech-like frequencies)
    freqs = [100, 200, 300, 500, 800, 1200, 2000, 3000]
    signal = np.zeros(samples)
    
    for freq in freqs:
        # Add varying amplitude to simulate speech
        amplitude = volume * random.uniform(0.1, 0.3)
        phase = random.uniform(0, 2*np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, volume * 0.05, samples)
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * volume
    
    return signal

def create_test_audio_files(output_dir, num_files=10):
    """Create test audio files with various characteristics"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    test_files = []
    
    # Test file configurations
    test_configs = [
        # (filename, duration, sample_rate, channels, volume, description)
        ("short_audio.wav", 0.3, 16000, 1, 0.1, "Short audio (0.3s)"),
        ("very_short_audio.wav", 0.1, 16000, 1, 0.1, "Very short audio (0.1s)"),
        ("normal_audio.wav", 5.0, 16000, 1, 0.1, "Normal audio (5s)"),
        ("long_audio.wav", 90.0, 16000, 1, 0.1, "Long audio (90s)"),
        ("very_long_audio.wav", 300.0, 16000, 1, 0.1, "Very long audio (300s)"),
        ("low_volume_audio.wav", 3.0, 16000, 1, 0.005, "Low volume audio"),
        ("stereo_audio.wav", 4.0, 16000, 2, 0.1, "Stereo audio"),
        ("diff_sr_audio.wav", 6.0, 44100, 1, 0.1, "Different sample rate (44.1kHz)"),
        ("high_freq_audio.wav", 7.0, 16000, 1, 0.1, "High frequency audio"),
        ("noisy_audio.wav", 8.0, 16000, 1, 0.1, "Noisy audio"),
    ]
    
    print(f"Generating {len(test_configs)} test audio files...")
    
    for i, (filename, duration, sample_rate, channels, volume, description) in enumerate(test_configs):
        print(f"  [{i+1}/{len(test_configs)}] Creating {filename}: {description}")
        
        # Generate audio
        if channels == 1:
            audio = generate_synthetic_speech(duration, sample_rate, volume)
        else:
            # Stereo audio
            left_channel = generate_synthetic_speech(duration, sample_rate, volume)
            right_channel = generate_synthetic_speech(duration, sample_rate, volume * 0.8)
            audio = np.column_stack([left_channel, right_channel])
        
        # Add special characteristics
        if "noisy" in filename:
            # Add more noise
            noise = np.random.normal(0, volume * 0.2, audio.shape)
            audio += noise
        
        elif "high_freq" in filename:
            # Add high frequency components
            t = np.linspace(0, duration, len(audio))
            high_freq = 0.05 * np.sin(2 * np.pi * 8000 * t)
            audio += high_freq
        
        # Save audio file
        file_path = output_path / filename
        sf.write(str(file_path), audio, sample_rate)
        
        # Get actual file info
        info = sf.info(str(file_path))
        test_files.append({
            "filename": filename,
            "description": description,
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "volume": np.max(np.abs(audio)),
            "file_size": file_path.stat().st_size
        })
        
        print(f"    ? Saved: {file_path}")
        print(f"      Duration: {info.duration:.2f}s, SR: {info.samplerate}Hz, Ch: {info.channels}")
    
    return test_files

def create_ground_truth_csv(test_files, output_dir):
    """Create ground truth CSV file for testing"""
    csv_path = Path(output_dir) / "test_ground_truth.csv"
    
    # Generate synthetic transcripts
    transcripts = [
        "This is a test audio file for ASR evaluation.",
        "The quick brown fox jumps over the lazy dog.",
        "Emergency medical services call center testing.",
        "Patient experiencing chest pain and shortness of breath.",
        "Dispatch ambulance to location immediately.",
        "Vital signs are stable at this time.",
        "Patient history includes diabetes and hypertension.",
        "Administer oxygen and monitor closely.",
        "ETA for ambulance arrival is ten minutes.",
        "Continue monitoring patient condition."
    ]
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Filename,transcript\n")
        for i, test_file in enumerate(test_files):
            transcript = transcripts[i % len(transcripts)]
            f.write(f"{test_file['filename']},{transcript}\n")
    
    print(f"? Created ground truth CSV: {csv_path}")
    return str(csv_path)

def create_test_summary(test_files, output_dir):
    """Create test summary JSON file"""
    summary_path = Path(output_dir) / "test_data_summary.json"
    
    summary = {
        "test_data_info": {
            "total_files": len(test_files),
            "total_duration": sum(f["duration"] for f in test_files),
            "file_formats": ["wav"],
            "sample_rates": list(set(f["sample_rate"] for f in test_files)),
            "channel_configs": list(set(f["channels"] for f in test_files))
        },
        "test_files": test_files,
        "model_compatibility": {
            "large-v3": {
                "compatible_files": len([f for f in test_files if f["duration"] > 0]),
                "description": "All files should be compatible"
            },
            "canary-1b": {
                "compatible_files": len([f for f in test_files if 0.5 <= f["duration"] <= 60 and f["sample_rate"] == 16000]),
                "description": "Files with 0.5-60s duration and 16kHz sample rate"
            },
            "parakeet-tdt-0.6b-v2": {
                "compatible_files": len([f for f in test_files if f["duration"] >= 1.0 and f["sample_rate"] == 16000]),
                "description": "Files with >=1.0s duration and 16kHz sample rate"
            },
            "wav2vec-xls-r": {
                "compatible_files": len([f for f in test_files if f["duration"] >= 0.1 and f["sample_rate"] == 16000]),
                "description": "Files with >=0.1s duration and 16kHz sample rate"
            }
        }
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"? Created test summary: {summary_path}")
    return str(summary_path)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate test data for ASR pipeline testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output_dir', type=str, default='./test_data',
                       help='Output directory for test data')
    parser.add_argument('--num_files', type=int, default=10,
                       help='Number of test files to generate')
    parser.add_argument('--create_ground_truth', action='store_true',
                       help='Create ground truth CSV file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("=== Test Data Generator ===")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of files: {args.num_files}")
        print(f"Create ground truth: {args.create_ground_truth}")
        print()
    
    # Create test audio files
    test_files = create_test_audio_files(args.output_dir, args.num_files)
    
    # Create ground truth CSV if requested
    ground_truth_path = None
    if args.create_ground_truth:
        ground_truth_path = create_ground_truth_csv(test_files, args.output_dir)
    
    # Create test summary
    summary_path = create_test_summary(test_files, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST DATA GENERATION SUMMARY")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Total files generated: {len(test_files)}")
    print(f"Total duration: {sum(f['duration'] for f in test_files):.2f} seconds")
    print()
    
    print("File characteristics:")
    for test_file in test_files:
        print(f"  - {test_file['filename']}: {test_file['description']}")
        print(f"    Duration: {test_file['duration']:.2f}s, SR: {test_file['sample_rate']}Hz, Ch: {test_file['channels']}")
    
    print()
    print("Model compatibility analysis:")
    summary_data = json.load(open(summary_path))
    for model, info in summary_data["model_compatibility"].items():
        print(f"  - {model}: {info['compatible_files']}/{len(test_files)} files ({info['description']})")
    
    if ground_truth_path:
        print(f"\nGround truth CSV: {ground_truth_path}")
    
    print(f"\nTest summary: {summary_path}")
    print("\n? Test data generation completed successfully!")
    print("\nNext steps:")
    print("1. Run the integrated pipeline:")
    print(f"   ./run_integrated_pipeline.sh --input_dir {args.output_dir} --output_dir ./pipeline_results")
    print("\n2. Test audio preprocessing:")
    print(f"   python3 audio_preprocessor.py --input_dir {args.output_dir} --output_dir ./preprocessed_test --verbose")
    print("\n3. Run quick test:")
    print("   ./quick_start.sh")

if __name__ == '__main__':
    main() 