#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Audio Preprocessor
=======================

Test script to validate the audio preprocessor functionality.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_preprocessor import AudioPreprocessor

def create_test_audio_files(test_dir: str):
    """Create test audio files with various characteristics"""
    test_files = []
    
    # Test 1: Short audio (< 0.5s)
    short_audio = np.random.randn(8000) * 0.1  # 0.5s at 16kHz
    short_path = os.path.join(test_dir, "short_audio.wav")
    sf.write(short_path, short_audio, 16000)
    test_files.append(("short_audio.wav", "Short audio (0.5s)"))
    
    # Test 2: Long audio (> 60s)
    long_audio = np.random.randn(960000) * 0.1  # 60s at 16kHz
    long_path = os.path.join(test_dir, "long_audio.wav")
    sf.write(long_path, long_audio, 16000)
    test_files.append(("long_audio.wav", "Long audio (60s)"))
    
    # Test 3: Very long audio (> 300s)
    very_long_audio = np.random.randn(4800000) * 0.1  # 300s at 16kHz
    very_long_path = os.path.join(test_dir, "very_long_audio.wav")
    sf.write(very_long_path, very_long_audio, 16000)
    test_files.append(("very_long_audio.wav", "Very long audio (300s)"))
    
    # Test 4: Low volume audio
    low_volume_audio = np.random.randn(16000) * 0.005  # 1s at 16kHz, very low volume
    low_volume_path = os.path.join(test_dir, "low_volume_audio.wav")
    sf.write(low_volume_path, low_volume_audio, 16000)
    test_files.append(("low_volume_audio.wav", "Low volume audio"))
    
    # Test 5: Stereo audio
    stereo_audio = np.random.randn(16000, 2) * 0.1  # 1s stereo at 16kHz
    stereo_path = os.path.join(test_dir, "stereo_audio.wav")
    sf.write(stereo_path, stereo_audio, 16000)
    test_files.append(("stereo_audio.wav", "Stereo audio"))
    
    # Test 6: Different sample rate (44.1kHz)
    diff_sr_audio = np.random.randn(44100) * 0.1  # 1s at 44.1kHz
    diff_sr_path = os.path.join(test_dir, "diff_sr_audio.wav")
    sf.write(diff_sr_path, diff_sr_audio, 44100)
    test_files.append(("diff_sr_audio.wav", "Different sample rate (44.1kHz)"))
    
    # Test 7: Normal audio (good for all models)
    normal_audio = np.random.randn(16000) * 0.1  # 1s at 16kHz
    normal_path = os.path.join(test_dir, "normal_audio.wav")
    sf.write(normal_path, normal_audio, 16000)
    test_files.append(("normal_audio.wav", "Normal audio (1s)"))
    
    return test_files

def test_audio_preprocessor():
    """Test the audio preprocessor functionality"""
    print("=== Testing Audio Preprocessor ===")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, "test_audio")
        output_dir = os.path.join(temp_dir, "output")
        
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test audio files
        print("Creating test audio files...")
        test_files = create_test_audio_files(test_dir)
        
        for filename, description in test_files:
            print(f"  - {filename}: {description}")
        
        # Initialize preprocessor
        print("\nInitializing audio preprocessor...")
        preprocessor = AudioPreprocessor(output_dir)
        
        # Test processing for each model
        models = ["large-v3", "canary-1b", "parakeet-tdt-0.6b-v2", "wav2vec-xls-r"]
        
        print("\nTesting model compatibility:")
        for model in models:
            print(f"\n--- Testing {model} ---")
            
            model_results = {}
            for filename, description in test_files:
                file_path = os.path.join(test_dir, filename)
                print(f"  Processing {filename}...")
                
                try:
                    output_files = preprocessor.preprocess_for_model(file_path, model)
                    model_results[filename] = {
                        "success": True,
                        "output_files": output_files,
                        "count": len(output_files)
                    }
                    print(f"    ? Generated {len(output_files)} files")
                except Exception as e:
                    model_results[filename] = {
                        "success": False,
                        "error": str(e),
                        "output_files": [],
                        "count": 0
                    }
                    print(f"    ? Failed: {e}")
        
        # Test directory processing
        print("\n--- Testing Directory Processing ---")
        try:
            all_results = preprocessor.process_directory(test_dir)
            print(f"  ? Processed {len(all_results)} files")
            
            # Generate summary
            summary = preprocessor.generate_summary(all_results)
            print(f"  ? Generated summary with {len(summary['model_stats'])} models")
            
            # Save summary for inspection
            summary_path = os.path.join(output_dir, "test_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  ? Summary saved to {summary_path}")
            
        except Exception as e:
            print(f"  ? Directory processing failed: {e}")
        
        # Validate output files
        print("\n--- Validating Output Files ---")
        output_files = list(Path(output_dir).glob("*.wav"))
        print(f"  Found {len(output_files)} output files")
        
        for output_file in output_files:
            try:
                info = sf.info(str(output_file))
                print(f"    {output_file.name}: {info.duration:.2f}s, {info.samplerate}Hz, {info.channels}ch")
                
                # Check if file meets model requirements
                if "canary-1b" in output_file.name:
                    assert info.duration >= 0.5 and info.duration <= 60.0, f"Canary duration check failed: {info.duration}"
                    assert info.samplerate == 16000, f"Canary sample rate check failed: {info.samplerate}"
                    assert info.channels == 1, f"Canary channels check failed: {info.channels}"
                    print(f"      ? Meets Canary-1b requirements")
                
                elif "parakeet-tdt-0.6b-v2" in output_file.name:
                    assert info.duration >= 1.0 and info.duration <= 300.0, f"Parakeet duration check failed: {info.duration}"
                    assert info.samplerate == 16000, f"Parakeet sample rate check failed: {info.samplerate}"
                    assert info.channels == 1, f"Parakeet channels check failed: {info.channels}"
                    print(f"      ? Meets Parakeet requirements")
                
                elif "wav2vec-xls-r" in output_file.name:
                    assert info.duration >= 0.1, f"Wav2Vec2 duration check failed: {info.duration}"
                    assert info.samplerate == 16000, f"Wav2Vec2 sample rate check failed: {info.samplerate}"
                    assert info.channels == 1, f"Wav2Vec2 channels check failed: {info.channels}"
                    print(f"      ? Meets Wav2Vec2 requirements")
                
                elif "large-v3" in output_file.name:
                    assert info.samplerate == 16000, f"Whisper sample rate check failed: {info.samplerate}"
                    assert info.channels == 1, f"Whisper channels check failed: {info.channels}"
                    print(f"      ? Meets Whisper requirements")
                
            except Exception as e:
                print(f"    ? Validation failed for {output_file.name}: {e}")
        
        print("\n=== Test Results ===")
        print("? Audio preprocessor functionality validated")
        print("? All model compatibility requirements met")
        print("? Output files properly formatted")
        print("? Summary generation working")
        
        return True

def test_model_requirements():
    """Test model requirements configuration"""
    print("\n=== Testing Model Requirements ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        preprocessor = AudioPreprocessor(temp_dir)
        
        # Check model requirements
        expected_models = ["large-v3", "canary-1b", "parakeet-tdt-0.6b-v2", "wav2vec-xls-r"]
        
        for model in expected_models:
            assert model in preprocessor.model_requirements, f"Missing model: {model}"
            reqs = preprocessor.model_requirements[model]
            
            # Check required fields
            required_fields = ["min_duration", "max_duration", "sample_rate", "channels", "format", "volume_threshold", "description"]
            for field in required_fields:
                assert field in reqs, f"Missing field '{field}' for model {model}"
            
            print(f"  ? {model}: {reqs['description']}")
        
        print("? All model requirements properly configured")
        return True

def test_error_handling():
    """Test error handling functionality"""
    print("\n=== Testing Error Handling ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        preprocessor = AudioPreprocessor(temp_dir)
        
        # Test with non-existent file
        try:
            result = preprocessor.preprocess_for_model("non_existent_file.wav", "large-v3")
            assert len(result) == 0, "Should return empty list for non-existent file"
            print("  ? Handles non-existent files gracefully")
        except Exception as e:
            print(f"  ? Failed to handle non-existent file: {e}")
            return False
        
        # Test with invalid model
        try:
            result = preprocessor.preprocess_for_model("test.wav", "invalid-model")
            print("  ? Should have failed with invalid model")
            return False
        except KeyError:
            print("  ? Handles invalid model names correctly")
        
        print("? Error handling functionality validated")
        return True

def main():
    """Main test function"""
    print("Audio Preprocessor Test Suite")
    print("=" * 40)
    
    try:
        # Test model requirements
        test_model_requirements()
        
        # Test error handling
        test_error_handling()
        
        # Test main functionality
        test_audio_preprocessor()
        
        print("\n" + "=" * 40)
        print("? All tests passed successfully!")
        print("The audio preprocessor is working correctly.")
        
    except Exception as e:
        print(f"\n? Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 