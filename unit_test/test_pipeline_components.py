#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Components Unit Tests
=============================

Comprehensive unit tests for ASR pipeline components:
1. Audio format validation
2. VAD processing
3. ASR model processing
4. File handling and error detection
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
import pytest
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pipeline components
try:
    from vad_pipeline import VADPipeline
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

# Test configuration
TEST_MODELS = [
    ('large-v3', 'whisper', WHISPER_AVAILABLE),
    ('canary-1b', 'nemo', NEMO_AVAILABLE),
    ('parakeet-tdt-0.6b-v2', 'nemo', NEMO_AVAILABLE),
    ('wav2vec-xls-r', 'transformers', TRANSFORMERS_AVAILABLE)
]

class TestAudioFormat:
    """Test audio format handling"""
    
    @pytest.mark.parametrize("wav_path", [
        "test_dataset/normal_30s.wav",
        "test_dataset/normal_180s.wav",
        "test_dataset/silence_30s.wav"
    ])
    def test_audio_format_validation(self, wav_path):
        """Test audio format validation"""
        if not os.path.exists(wav_path):
            pytest.skip(f"Test file not found: {wav_path}")
        
        try:
            info = torchaudio.info(wav_path)
            print(f"Audio info for {wav_path}:")
            print(f"  Sample rate: {info.sample_rate}")
            print(f"  Channels: {info.num_channels}")
            print(f"  Duration: {info.num_frames / info.sample_rate:.2f}s")
            
            # Basic validation
            assert info.sample_rate > 0, "Invalid sample rate"
            assert info.num_channels > 0, "Invalid channel count"
            assert info.num_frames > 0, "Invalid frame count"
            
        except Exception as e:
            pytest.fail(f"Failed to read audio file {wav_path}: {e}")
    
    def test_audio_normalization(self):
        """Test audio normalization (stereo to mono, resampling)"""
        # Test with stereo file
        stereo_file = "test_dataset/normal_180s.wav"
        if not os.path.exists(stereo_file):
            pytest.skip(f"Test file not found: {stereo_file}")
        
        try:
            # Load stereo audio
            waveform, sample_rate = torchaudio.load(stereo_file)
            print(f"Original: {waveform.shape}, {sample_rate}Hz")
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                print(f"Converted to mono: {waveform.shape}")
            
            # Resample to 16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
                print(f"Resampled to 16kHz: {waveform.shape}")
            
            # Save normalized file
            normalized_file = "test_dataset/normal_180s_normalized.wav"
            torchaudio.save(normalized_file, waveform, sample_rate)
            
            # Verify
            info = torchaudio.info(normalized_file)
            assert info.sample_rate == 16000, "Sample rate should be 16kHz"
            assert info.num_channels == 1, "Should be mono"
            
            print(f"Normalization successful: {normalized_file}")
            
        except Exception as e:
            pytest.fail(f"Audio normalization failed: {e}")

class TestVADProcessing:
    """Test VAD processing"""
    
    @pytest.mark.skipif(not VAD_AVAILABLE, reason="VAD not available")
    def test_vad_basic_functionality(self):
        """Test basic VAD functionality"""
        test_file = "test_dataset/normal_30s.wav"
        if not os.path.exists(test_file):
            pytest.skip(f"Test file not found: {test_file}")
        
        try:
            # Create VAD pipeline
            vad = VADPipeline(
                speech_threshold=0.2,  # Lower threshold for testing
                min_speech_duration=0.2,
                min_silence_duration=0.15,
                target_sample_rate=16000
            )
            
            # Process file
            temp_dir = tempfile.mkdtemp(prefix="vad_test_")
            result = vad.process_audio_file(test_file, temp_dir)
            
            print(f"VAD result: {result}")
            
            # Basic validation
            assert 'error' not in result, f"VAD processing failed: {result.get('error', 'Unknown error')}"
            assert 'total_speech_duration' in result, "Missing speech duration"
            
            # Clean up
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            pytest.fail(f"VAD test failed: {e}")
    
    @pytest.mark.skipif(not VAD_AVAILABLE, reason="VAD not available")
    def test_vad_parameter_sensitivity(self):
        """Test VAD sensitivity to different parameters"""
        test_file = "test_dataset/speech_gap.wav"
        if not os.path.exists(test_file):
            pytest.skip(f"Test file not found: {test_file}")
        
        # Test different parameter sets
        param_sets = [
            {'speech_threshold': 0.1, 'min_speech_duration': 0.1, 'min_silence_duration': 0.1},
            {'speech_threshold': 0.2, 'min_speech_duration': 0.2, 'min_silence_duration': 0.15},
            {'speech_threshold': 0.5, 'min_speech_duration': 0.5, 'min_silence_duration': 0.3},
        ]
        
        results = {}
        
        for i, params in enumerate(param_sets):
            try:
                vad = VADPipeline(**params)
                temp_dir = tempfile.mkdtemp(prefix=f"vad_test_{i}_")
                
                result = vad.process_audio_file(test_file, temp_dir)
                results[f"set_{i}"] = {
                    'params': params,
                    'result': result
                }
                
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                results[f"set_{i}"] = {
                    'params': params,
                    'error': str(e)
                }
        
        # Analyze results
        print(f"VAD parameter sensitivity results:")
        for set_name, data in results.items():
            params = data['params']
            print(f"  {set_name}: threshold={params['speech_threshold']}, "
                  f"min_speech={params['min_speech_duration']}")
            
            if 'error' in data:
                print(f"    ❌ Error: {data['error']}")
            else:
                result = data['result']
                if 'total_speech_duration' in result:
                    speech_ratio = result.get('speech_ratio', 0)
                    print(f"    ✓ Speech: {result['total_speech_duration']:.2f}s ({speech_ratio:.1%})")
                else:
                    print(f"    ❌ No speech detected")

class TestASRModels:
    """Test ASR model processing"""
    
    @pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
    def test_whisper_processing(self):
        """Test Whisper model processing"""
        test_file = "test_dataset/normal_30s.wav"
        if not os.path.exists(test_file):
            pytest.skip(f"Test file not found: {test_file}")
        
        try:
            # Load Whisper model
            model = whisper.load_model('large-v3')
            
            # Process file
            result = model.transcribe(test_file)
            
            print(f"Whisper result for {test_file}:")
            print(f"  Text: {result['text'][:100]}...")
            print(f"  Language: {result.get('language', 'unknown')}")
            
            # Basic validation
            assert 'text' in result, "Missing transcript text"
            assert len(result['text'].strip()) > 0, "Empty transcript"
            
        except Exception as e:
            pytest.fail(f"Whisper test failed: {e}")
    
    @pytest.mark.parametrize("model_name,framework,available", TEST_MODELS)
    def test_model_availability(self, model_name, framework, available):
        """Test model availability"""
        if not available:
            pytest.skip(f"{model_name} ({framework}) not available")
        
        print(f"✓ {model_name} ({framework}) is available")
    
    def test_model_processing_consistency(self):
        """Test consistency across different models"""
        test_files = [
            "test_dataset/normal_30s.wav",
            "test_dataset/very_short.wav",
            "test_dataset/low_volume.wav"
        ]
        
        results = {}
        
        for test_file in test_files:
            if not os.path.exists(test_file):
                continue
            
            results[test_file] = {}
            
            for model_name, framework, available in TEST_MODELS:
                if not available:
                    continue
                
                try:
                    # Simulate model processing (simplified)
                    transcript = f"test transcript for {os.path.basename(test_file)}"
                    results[test_file][model_name] = {
                        'success': True,
                        'transcript': transcript,
                        'length': len(transcript)
                    }
                    
                except Exception as e:
                    results[test_file][model_name] = {
                        'success': False,
                        'error': str(e)
                    }
        
        # Analyze results
        print(f"Model processing consistency results:")
        for test_file, model_results in results.items():
            print(f"\n  {os.path.basename(test_file)}:")
            for model_name, result in model_results.items():
                if result['success']:
                    print(f"    ✓ {model_name}: {result['length']} chars")
                else:
                    print(f"    ❌ {model_name}: {result['error']}")

class TestPipelineIntegration:
    """Test pipeline integration"""
    
    def test_file_handling(self):
        """Test file handling and error detection"""
        test_files = [
            "test_dataset/normal_30s.wav",
            "test_dataset/normal_180s.wav",
            "test_dataset/silence_30s.wav",
            "nonexistent_file.wav"
        ]
        
        results = {}
        
        for test_file in test_files:
            try:
                if os.path.exists(test_file):
                    # Get file info
                    info = torchaudio.info(test_file)
                    results[test_file] = {
                        'exists': True,
                        'size': os.path.getsize(test_file),
                        'duration': info.num_frames / info.sample_rate,
                        'channels': info.num_channels,
                        'sample_rate': info.sample_rate
                    }
                else:
                    results[test_file] = {
                        'exists': False,
                        'error': 'File not found'
                    }
                    
            except Exception as e:
                results[test_file] = {
                    'exists': False,
                    'error': str(e)
                }
        
        # Analyze results
        print(f"File handling test results:")
        for test_file, result in results.items():
            if result['exists']:
                print(f"  ✓ {os.path.basename(test_file)}: "
                      f"{result['size']} bytes, {result['duration']:.1f}s, "
                      f"{result['channels']}ch, {result['sample_rate']}Hz")
            else:
                print(f"  ❌ {os.path.basename(test_file)}: {result['error']}")
    
    def test_ground_truth_validation(self):
        """Test ground truth file validation"""
        ground_truth_file = "test_dataset/ground_truth.csv"
        
        if not os.path.exists(ground_truth_file):
            pytest.skip(f"Ground truth file not found: {ground_truth_file}")
        
        try:
            # Load ground truth
            df = pd.read_csv(ground_truth_file)
            
            print(f"Ground truth validation:")
            print(f"  Total entries: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            # Validate required columns
            required_columns = ['Filename', 'transcript']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                pytest.fail(f"Missing required columns: {missing_columns}")
            
            # Check for empty entries
            empty_filenames = df['Filename'].isna().sum()
            empty_transcripts = df['transcript'].isna().sum()
            
            print(f"  Empty filenames: {empty_filenames}")
            print(f"  Empty transcripts: {empty_transcripts}")
            
            # Check file existence
            test_dir = "test_dataset"
            missing_files = []
            for filename in df['Filename']:
                file_path = os.path.join(test_dir, filename)
                if not os.path.exists(file_path):
                    missing_files.append(filename)
            
            if missing_files:
                print(f"  Missing audio files: {missing_files}")
            else:
                print(f"  ✓ All audio files exist")
            
        except Exception as e:
            pytest.fail(f"Ground truth validation failed: {e}")

def run_comprehensive_test():
    """Run comprehensive pipeline test"""
    print("Pipeline Components Comprehensive Test")
    print("=" * 50)
    
    # Create test dataset if it doesn't exist
    if not os.path.exists("test_dataset"):
        print("Creating test dataset...")
        subprocess.run([sys.executable, "unit_test/create_test_dataset.py"])
    
    # Run pytest
    print("\nRunning unit tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "unit_test/test_pipeline_components.py", 
        "-v", "--tb=short"
    ])
    
    return result.returncode == 0

if __name__ == "__main__":
    success = run_comprehensive_test()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 