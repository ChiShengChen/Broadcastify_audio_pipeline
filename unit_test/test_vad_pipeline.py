#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAD Pipeline Test Script
========================

This script performs basic tests to verify that the VAD pipeline is properly installed
and can be used with your audio data.
"""

import os
import sys
import tempfile
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def create_test_audio(output_path: str, duration: float = 5.0, sample_rate: int = 16000):
    """Create a test audio file with speech-like segments"""
    print(f"Creating test audio file: {output_path}")
    
    # Generate a simple test signal with speech-like patterns
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create segments with different characteristics
    signal = np.zeros_like(t)
    
    # Add some "speech-like" segments (higher amplitude, modulated frequency)
    speech_segments = [
        (0.5, 1.5),   # First speech segment
        (2.0, 3.0),   # Second speech segment  
        (3.5, 4.5)    # Third speech segment
    ]
    
    for start, end in speech_segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        segment_t = t[start_idx:end_idx]
        
        # Generate speech-like signal (amplitude modulated sine wave)
        speech_signal = np.sin(2 * np.pi * 440 * segment_t) * np.sin(2 * np.pi * 5 * segment_t) * 0.3
        speech_signal += np.random.normal(0, 0.05, len(speech_signal))  # Add some noise
        
        signal[start_idx:end_idx] = speech_signal
    
    # Add very low-level background noise in silent segments
    noise_mask = signal == 0
    signal[noise_mask] = np.random.normal(0, 0.01, np.sum(noise_mask))
    
    # Convert to tensor and save
    audio_tensor = torch.from_numpy(signal).float().unsqueeze(0)  # Add channel dimension
    torchaudio.save(output_path, audio_tensor, sample_rate)
    print(f"Test audio saved: {output_path}")
    return output_path


def test_vad_import():
    """Test if VAD pipeline can be imported"""
    print("=== Testing VAD Pipeline Import ===")
    try:
        from vad_pipeline import VADPipeline
        print("✓ VAD Pipeline import successful")
        return True
    except Exception as e:
        print(f"✗ VAD Pipeline import failed: {e}")
        return False


def test_vad_model_loading():
    """Test if Silero VAD model can be loaded"""
    print("=== Testing VAD Model Loading ===")
    try:
        from vad_pipeline import VADPipeline
        vad = VADPipeline()
        vad.load_vad_model()
        print("✓ VAD model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ VAD model loading failed: {e}")
        print("This might be due to network issues or missing dependencies.")
        return False


def test_vad_processing():
    """Test VAD processing on a sample audio file"""
    print("=== Testing VAD Processing ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test audio
        test_audio_path = os.path.join(temp_dir, "test_audio.wav")
        create_test_audio(test_audio_path)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "vad_output")
        
        try:
            from vad_pipeline import VADPipeline
            
            # Initialize VAD pipeline
            vad = VADPipeline(
                speech_threshold=0.3,  # Lower threshold for test audio
                min_speech_duration=0.2,
                min_silence_duration=0.2
            )
            
            # Process the test file
            result = vad.process_audio_file(
                input_path=test_audio_path,
                output_dir=output_dir
            )
            
            if 'error' not in result:
                print(f"✓ VAD processing successful")
                print(f"  - Detected {result['num_segments']} speech segments")
                print(f"  - Speech ratio: {result['speech_ratio']:.1%}")
                print(f"  - Total speech: {result['total_speech_duration']:.1f}s / {result['original_duration']:.1f}s")
                
                # Check if segments were created
                output_path = Path(output_dir) / "test_audio"
                if output_path.exists():
                    segments = list(output_path.glob("segment_*.wav"))
                    print(f"  - Created {len(segments)} segment files")
                
                return True
            else:
                print(f"✗ VAD processing failed: {result['error']}")
                return False
                
        except Exception as e:
            print(f"✗ VAD processing failed: {e}")
            return False


def test_asr_integration():
    """Test if ASR models can be imported"""
    print("=== Testing ASR Integration ===")
    
    # Test each framework
    results = {}
    
    # Test Transformers
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torchaudio
        results['transformers'] = True
        print("✓ Transformers available")
    except ImportError as e:
        results['transformers'] = False
        print(f"✗ Transformers not available: {e}")
    
    # Test NeMo
    try:
        import nemo.collections.asr as nemo_asr
        results['nemo'] = True
        print("✓ NeMo available")
    except ImportError as e:
        results['nemo'] = False
        print(f"✗ NeMo not available: {e}")
    
    # Test Whisper
    try:
        import whisper
        results['whisper'] = True
        print("✓ Whisper available")
    except ImportError as e:
        results['whisper'] = False
        print(f"✗ Whisper not available: {e}")
    
    return results


def test_directory_structure():
    """Test if required files exist"""
    print("=== Testing Directory Structure ===")
    
    required_files = [
        'vad_pipeline.py',
        'run_vad_asr_pipeline.py',
        'run_vad_pipeline.sh',
        'vad_config.json',
        'example_vad_usage.py'
    ]
    
    all_exist = True
    for file in required_files:
        file_path = current_dir / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (missing)")
            all_exist = False
    
    return all_exist


def print_system_info():
    """Print system information"""
    print("=== System Information ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: Available (GPU: {torch.cuda.get_device_name()})")
    else:
        print("CUDA: Not available (will use CPU)")
    
    try:
        import torchaudio
        print(f"TorchAudio: {torchaudio.__version__}")
    except ImportError:
        print("TorchAudio: Not available")


def main():
    """Run all tests"""
    print("VAD Pipeline Test Suite")
    print("=======================")
    print()
    
    # Print system info
    print_system_info()
    print()
    
    # Run tests
    tests = [
        ("Directory Structure", test_directory_structure),
        ("VAD Import", test_vad_import),
        ("VAD Model Loading", test_vad_model_loading),
        ("VAD Processing", test_vad_processing),
        ("ASR Integration", test_asr_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("=== Test Summary ===")
    passed = sum(1 for result in results.values() if result is True)
    failed = len(results) - passed
    
    print(f"Tests passed: {passed}/{len(results)}")
    print(f"Tests failed: {failed}/{len(results)}")
    
    if all(results.values()):
        print("\n🎉 All tests passed! VAD pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Try the example: python3 ems_call/example_vad_usage.py")
        print("2. Run VAD on your data: python3 ems_call/vad_pipeline.py --input_dir YOUR_DIR --output_dir OUTPUT_DIR")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Common fixes:")
        print("- Install missing dependencies: pip install -r ems_call/requirements.txt")
        print("- Check network connection for model downloads")
        print("- Ensure CUDA drivers are properly installed (optional)")


if __name__ == '__main__':
    main() 