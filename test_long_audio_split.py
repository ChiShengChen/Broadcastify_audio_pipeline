#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for long audio splitting functionality
"""

import os
import torch
import torchaudio
import tempfile
import shutil
from pathlib import Path

def create_test_audio(duration_seconds=150, sample_rate=16000, output_path="test_long_audio.wav"):
    """
    Create a test audio file with specified duration
    
    Args:
        duration_seconds: Duration in seconds (default: 150s = 2.5 minutes)
        sample_rate: Sample rate (default: 16000Hz)
        output_path: Output file path
    """
    print(f"Creating test audio file: {output_path}")
    print(f"  - Duration: {duration_seconds}s")
    print(f"  - Sample rate: {sample_rate}Hz")
    
    # Create a simple sine wave
    num_samples = int(duration_seconds * sample_rate)
    frequency = 440  # A4 note
    t = torch.linspace(0, duration_seconds, num_samples)
    waveform = torch.sin(2 * torch.pi * frequency * t)
    
    # Add some silence at the beginning and end
    silence_samples = int(5 * sample_rate)  # 5 seconds of silence
    waveform = torch.cat([
        torch.zeros(silence_samples),
        waveform,
        torch.zeros(silence_samples)
    ])
    
    # Reshape for torchaudio (channels, samples)
    waveform = waveform.unsqueeze(0)
    
    # Save audio file
    torchaudio.save(output_path, waveform, sample_rate)
    print(f"  - Saved to: {output_path}")
    
    return output_path

def test_long_audio_splitter():
    """Test the long audio splitter functionality"""
    print("=== Testing Long Audio Splitter ===")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test audio files
        test_files = []
        
        # File 1: Short file (should not be split)
        short_file = create_test_audio(60, 16000, os.path.join(input_dir, "short_audio.wav"))
        test_files.append(short_file)
        
        # File 2: Long file (should be split)
        long_file = create_test_audio(180, 16000, os.path.join(input_dir, "long_audio.wav"))
        test_files.append(long_file)
        
        print(f"\nCreated {len(test_files)} test files in {input_dir}")
        
        # Test the long audio splitter
        try:
            from long_audio_splitter import LongAudioSplitter
            
            splitter = LongAudioSplitter(
                max_duration=120.0,  # 2 minutes
                speech_threshold=0.5,
                min_speech_duration=0.5,
                min_silence_duration=0.3
            )
            
            print(f"\nRunning long audio splitter...")
            results = splitter.process_directory(input_dir, output_dir)
            
            print(f"\nResults:")
            print(f"  - Total files: {results['total_files']}")
            print(f"  - Files split: {results['split_files']}")
            print(f"  - Files unchanged: {results['unsplit_files']}")
            print(f"  - Errors: {results['errors']}")
            
            # Check output structure
            if os.path.exists(output_dir):
                print(f"\nOutput directory structure:")
                for root, dirs, files in os.walk(output_dir):
                    level = root.replace(output_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        print(f"{subindent}{file}")
            
            print("\n✅ Long audio splitter test completed successfully!")
            
        except ImportError as e:
            print(f"❌ Error importing long_audio_splitter: {e}")
        except Exception as e:
            print(f"❌ Error during testing: {e}")

def test_merge_split_transcripts():
    """Test the merge split transcripts functionality"""
    print("\n=== Testing Merge Split Transcripts ===")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "transcripts")
        output_dir = os.path.join(temp_dir, "merged")
        metadata_dir = os.path.join(temp_dir, "metadata")
        
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Create mock transcript files
        transcript_files = [
            "large-v3_testfile_segment_000.txt",
            "large-v3_testfile_segment_001.txt",
            "large-v3_testfile_segment_002.txt",
            "large-v3_shortfile.txt"  # Regular transcript
        ]
        
        for filename in transcript_files:
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"Mock transcript content for {filename}")
        
        # Create mock metadata
        metadata_content = {
            "original_file": "testfile.wav",
            "original_duration": 180.0,
            "max_segment_duration": 120.0,
            "total_segments": 3,
            "segments": [
                {"segment_id": 0, "filename": "testfile_segment_000.wav"},
                {"segment_id": 1, "filename": "testfile_segment_001.wav"},
                {"segment_id": 2, "filename": "testfile_segment_002.wav"}
            ]
        }
        
        metadata_file = os.path.join(metadata_dir, "testfile_metadata.json")
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata_content, f, indent=2)
        
        print(f"Created mock data:")
        print(f"  - Transcript files: {len(transcript_files)}")
        print(f"  - Metadata file: {metadata_file}")
        
        # Test the merge functionality
        try:
            from merge_split_transcripts import process_split_transcripts, copy_unsplit_transcripts
            
            print(f"\nRunning merge split transcripts...")
            
            model_prefixes = ['large-v3']
            process_split_transcripts(input_dir, output_dir, metadata_dir, model_prefixes)
            copy_unsplit_transcripts(input_dir, output_dir, model_prefixes)
            
            # Check output structure
            if os.path.exists(output_dir):
                print(f"\nOutput directory structure:")
                for root, dirs, files in os.walk(output_dir):
                    level = root.replace(output_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        print(f"{subindent}{file}")
            
            print("\n✅ Merge split transcripts test completed successfully!")
            
        except ImportError as e:
            print(f"❌ Error importing merge_split_transcripts: {e}")
        except Exception as e:
            print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    print("Running tests for long audio splitting functionality...")
    
    # Test long audio splitter
    test_long_audio_splitter()
    
    # Test merge functionality
    test_merge_split_transcripts()
    
    print("\n🎉 All tests completed!") 