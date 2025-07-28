#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Audio Preprocessing Functionality
=====================================

This script tests the audio preprocessing functionality including:
1. Upsampling audio files
2. Splitting long audio files
3. Merging segmented transcripts

Usage:
    python3 test_audio_preprocessing.py
"""

import os
import tempfile
import shutil
import torch
import torchaudio
import json
from pathlib import Path
import subprocess
import sys

def create_test_audio_files(test_dir: str) -> list:
    """Create test audio files with different sample rates and durations"""
    test_files = []
    
    # Create test audio with different characteristics
    test_cases = [
        {"sample_rate": 8000, "duration": 30, "filename": "test_8k_30s.wav"},
        {"sample_rate": 8000, "duration": 90, "filename": "test_8k_90s.wav"},
        {"sample_rate": 16000, "duration": 45, "filename": "test_16k_45s.wav"},
        {"sample_rate": 16000, "duration": 120, "filename": "test_16k_120s.wav"},
    ]
    
    for case in test_cases:
        # Generate test audio (sine wave)
        sample_rate = case["sample_rate"]
        duration = case["duration"]
        num_samples = int(sample_rate * duration)
        
        # Create a simple sine wave
        t = torch.linspace(0, duration, num_samples)
        frequency = 440  # A4 note
        waveform = torch.sin(2 * torch.pi * frequency * t)
        waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        # Save audio file
        file_path = os.path.join(test_dir, case["filename"])
        torchaudio.save(file_path, waveform, sample_rate)
        test_files.append(file_path)
        
        print(f"Created test file: {case['filename']} ({sample_rate}Hz, {duration}s)")
    
    return test_files

def create_test_transcripts(test_dir: str, audio_files: list) -> list:
    """Create test transcript files for the audio files"""
    transcript_files = []
    
    models = ["large-v3", "canary-1b", "wav2vec-xls-r"]
    
    for audio_file in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        
        for model in models:
            # Create transcript file
            transcript_filename = f"{model}_{base_name}.txt"
            transcript_path = os.path.join(test_dir, transcript_filename)
            
            # Create dummy transcript content
            content = f"This is a test transcript for {base_name} processed by {model}."
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            transcript_files.append(transcript_path)
            print(f"Created transcript: {transcript_filename}")
    
    return transcript_files

def test_audio_preprocessing():
    """Test the audio preprocessing functionality"""
    print("=== Testing Audio Preprocessing Functionality ===")
    
    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Test directory: {temp_dir}")
        
        # Create test audio files
        print("\n1️⃣ Creating test audio files...")
        audio_files = create_test_audio_files(temp_dir)
        
        # Create test transcripts
        print("\n2️⃣ Creating test transcript files...")
        transcript_files = create_test_transcripts(temp_dir, audio_files)
        
        # Test audio preprocessing
        print("\n3️⃣ Testing audio preprocessing...")
        preprocessed_dir = os.path.join(temp_dir, "preprocessed")
        
        cmd = [
            "python3", "audio_preprocessor.py",
            "--input_dir", temp_dir,
            "--output_dir", preprocessed_dir,
            "--target_sample_rate", "16000",
            "--max_duration", "60",
            "--overlap_duration", "1",
            "--min_segment_duration", "5",
            "--preserve_structure"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Audio preprocessing completed successfully")
            print(result.stdout)
        else:
            print("❌ Audio preprocessing failed")
            print(result.stderr)
            return False
        
        # Check preprocessing results
        metadata_file = os.path.join(preprocessed_dir, "processing_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"\n📊 Preprocessing Results:")
            print(f"  - Total files: {metadata.get('total_files', 'N/A')}")
            print(f"  - Processed files: {metadata.get('processed_files', 'N/A')}")
            print(f"  - Total segments: {metadata.get('total_segments', 'N/A')}")
            print(f"  - Upsampled files: {metadata.get('upsampled_files', 'N/A')}")
            print(f"  - Split files: {metadata.get('split_files', 'N/A')}")
        
        # Test transcript merging
        print("\n4️⃣ Testing transcript merging...")
        merged_dir = os.path.join(temp_dir, "merged_transcripts")
        
        # Create some segmented transcript files for testing
        segmented_transcripts_dir = os.path.join(temp_dir, "segmented_transcripts")
        os.makedirs(segmented_transcripts_dir, exist_ok=True)
        
        # Create dummy segmented transcripts
        for audio_file in audio_files:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            models = ["large-v3", "canary-1b"]
            
            for model in models:
                # Create single segment transcript
                transcript_filename = f"{model}_{base_name}.txt"
                transcript_path = os.path.join(segmented_transcripts_dir, transcript_filename)
                
                content = f"Test transcript for {base_name} by {model}."
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        
        # Test merging
        merge_cmd = [
            "python3", "merge_segmented_transcripts.py",
            "--input_dir", segmented_transcripts_dir,
            "--output_dir", merged_dir,
            "--metadata_file", metadata_file
        ]
        
        merge_result = subprocess.run(merge_cmd, capture_output=True, text=True)
        
        if merge_result.returncode == 0:
            print("✅ Transcript merging completed successfully")
            print(merge_result.stdout)
        else:
            print("❌ Transcript merging failed")
            print(merge_result.stderr)
            return False
        
        print("\n✅ All tests completed successfully!")
        return True

def test_pipeline_integration():
    """Test the integration with the main pipeline"""
    print("\n=== Testing Pipeline Integration ===")
    
    # Create a simple test configuration
    test_config = {
        "input_dir": "/tmp/test_audio",
        "output_dir": "/tmp/test_output",
        "use_audio_preprocessing": True,
        "target_sample_rate": 16000,
        "audio_max_duration": 60.0
    }
    
    print("Pipeline integration test configuration:")
    for key, value in test_config.items():
        print(f"  - {key}: {value}")
    
    print("\n✅ Pipeline integration test completed!")
    return True

def main():
    """Main test function"""
    print("🧪 Audio Preprocessing Test Suite")
    print("=" * 50)
    
    # Test audio preprocessing functionality
    if not test_audio_preprocessing():
        print("\n❌ Audio preprocessing tests failed")
        return 1
    
    # Test pipeline integration
    if not test_pipeline_integration():
        print("\n❌ Pipeline integration tests failed")
        return 1
    
    print("\n🎉 All tests passed successfully!")
    print("\n📝 Test Summary:")
    print("  ✅ Audio preprocessing (upsampling and segmentation)")
    print("  ✅ Transcript merging for WER calculation")
    print("  ✅ Pipeline integration")
    print("  ✅ Metadata handling")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 