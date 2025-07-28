#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Preprocessor for ASR Model Compatibility
=============================================

This script preprocesses audio files to ensure compatibility with different ASR models
based on their specific limitations and requirements.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
import librosa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Audio preprocessor for ASR model compatibility"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model-specific requirements
        self.model_requirements = {
            "large-v3": {
                "min_duration": 0.0,      # No minimum
                "max_duration": float('inf'),  # No maximum
                "sample_rate": 16000,      # Preferred
                "channels": 1,             # Mono
                "format": "wav",
                "volume_threshold": 0.0,   # No minimum
                "description": "Whisper large-v3 - Most flexible"
            },
            "canary-1b": {
                "min_duration": 0.5,      # Minimum 0.5s
                "max_duration": 60.0,     # Maximum 60s
                "sample_rate": 16000,     # Required
                "channels": 1,            # Mono
                "format": "wav",
                "volume_threshold": 0.01, # Minimum volume
                "description": "NeMo Canary-1b - Strict duration limits"
            },
            "parakeet-tdt-0.6b-v2": {
                "min_duration": 1.0,      # Minimum 1.0s
                "max_duration": 300.0,    # Maximum 300s
                "sample_rate": 16000,     # Required
                "channels": 1,            # Mono
                "format": "wav",
                "volume_threshold": 0.0,  # No minimum
                "description": "NeMo Parakeet - Medium flexibility"
            },
            "wav2vec-xls-r": {
                "min_duration": 0.1,      # Minimum 0.1s
                "max_duration": float('inf'),  # No maximum
                "sample_rate": 16000,     # Required
                "channels": 1,            # Mono
                "format": "wav",
                "volume_threshold": 0.01, # Minimum volume
                "description": "Wav2Vec2 - Good flexibility"
            }
        }
    
    def get_audio_info(self, audio_path: str) -> Dict:
        """Get audio file information"""
        try:
            # Try using soundfile first
            info = sf.info(audio_path)
            duration = info.duration
            sample_rate = info.samplerate
            channels = info.channels
            
            # Calculate volume
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            volume = np.abs(audio).max()
            
            return {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "volume": volume,
                "format": Path(audio_path).suffix.lower()
            }
        except Exception as e:
            logger.warning(f"Error reading {audio_path}: {e}")
            return None
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            audio, sample_rate = sf.read(audio_path)
            return audio, sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return None, None
    
    def resample_audio(self, audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if src_sr == target_sr:
            return audio
        
        # Use librosa for resampling
        audio_resampled = librosa.resample(
            audio, 
            orig_sr=src_sr, 
            target_sr=target_sr
        )
        return audio_resampled
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mono"""
        if len(audio.shape) == 1:
            return audio
        return audio.mean(axis=1)
    
    def normalize_volume(self, audio: np.ndarray, target_volume: float = 0.5) -> np.ndarray:
        """Normalize audio volume"""
        current_max = np.abs(audio).max()
        if current_max > 0:
            scale_factor = target_volume / current_max
            audio = audio * scale_factor
        return audio
    
    def trim_silence(self, audio: np.ndarray, sample_rate: int, 
                    threshold: float = 0.01, min_duration: float = 0.1) -> np.ndarray:
        """Trim silence from audio"""
        # Calculate frame length for silence detection
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        
        # Find non-silent regions
        non_silent_ranges = librosa.effects.split(
            audio, 
            top_db=-20,  # -20dB threshold
            frame_length=frame_length,
            hop_length=frame_length // 4
        )
        
        if len(non_silent_ranges) == 0:
            return audio
        
        # Combine all non-silent regions
        trimmed_audio = np.concatenate([
            audio[start:end] for start, end in non_silent_ranges
        ])
        
        # Ensure minimum duration
        min_samples = int(sample_rate * min_duration)
        if len(trimmed_audio) < min_samples:
            # Pad with silence if too short
            padding = np.zeros(min_samples - len(trimmed_audio))
            trimmed_audio = np.concatenate([trimmed_audio, padding])
        
        return trimmed_audio
    
    def split_long_audio(self, audio: np.ndarray, sample_rate: int, 
                        max_duration: float, overlap: float = 0.5) -> List[np.ndarray]:
        """Split long audio into segments"""
        max_samples = int(sample_rate * max_duration)
        overlap_samples = int(sample_rate * overlap)
        
        segments = []
        start = 0
        
        while start < len(audio):
            end = min(start + max_samples, len(audio))
            segment = audio[start:end]
            segments.append(segment)
            
            if end >= len(audio):
                break
            
            start = end - overlap_samples
        
        return segments
    
    def pad_short_audio(self, audio: np.ndarray, sample_rate: int, 
                       min_duration: float) -> np.ndarray:
        """Pad short audio to minimum duration"""
        min_samples = int(sample_rate * min_duration)
        
        if len(audio) >= min_samples:
            return audio
        
        # Pad with silence
        padding_samples = min_samples - len(audio)
        padding = np.zeros(padding_samples)
        padded_audio = np.concatenate([audio, padding])
        
        return padded_audio
    
    def save_audio(self, audio: np.ndarray, sample_rate: int, 
                  output_path: str, format: str = "wav") -> bool:
        """Save audio to file"""
        try:
            sf.write(output_path, audio, sample_rate, format=format.upper())
            return True
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {e}")
            return False
    
    def preprocess_for_model(self, audio_path: str, model_name: str) -> List[str]:
        """Preprocess audio for specific model"""
        requirements = self.model_requirements[model_name]
        
        # Load audio
        audio, src_sr = self.load_audio(audio_path)
        if audio is None:
            return []
        
        # Get original info
        audio_info = self.get_audio_info(audio_path)
        if audio_info is None:
            return []
        
        logger.info(f"Processing {audio_path} for {model_name}")
        logger.info(f"Original: {audio_info['duration']:.2f}s, {audio_info['sample_rate']}Hz, {audio_info['channels']}ch")
        
        # Convert to mono if needed
        if audio_info['channels'] > 1:
            audio = self.convert_to_mono(audio)
            logger.info("Converted to mono")
        
        # Resample if needed
        target_sr = requirements['sample_rate']
        if src_sr != target_sr:
            audio = self.resample_audio(audio, src_sr, target_sr)
            logger.info(f"Resampled to {target_sr}Hz")
        
        # Normalize volume if needed
        if requirements['volume_threshold'] > 0:
            current_volume = np.abs(audio).max()
            if current_volume < requirements['volume_threshold']:
                audio = self.normalize_volume(audio, requirements['volume_threshold'] * 2)
                logger.info(f"Normalized volume from {current_volume:.4f} to {np.abs(audio).max():.4f}")
        
        # Trim silence for better processing
        audio = self.trim_silence(audio, target_sr)
        
        # Handle duration constraints
        duration = len(audio) / target_sr
        min_duration = requirements['min_duration']
        max_duration = requirements['max_duration']
        
        output_files = []
        base_name = Path(audio_path).stem
        
        if duration < min_duration:
            # Pad short audio
            audio = self.pad_short_audio(audio, target_sr, min_duration)
            logger.info(f"Padded to {len(audio) / target_sr:.2f}s (min: {min_duration}s)")
            
            # Save single file
            output_path = self.output_dir / f"{base_name}_{model_name}.wav"
            if self.save_audio(audio, target_sr, str(output_path)):
                output_files.append(str(output_path))
        
        elif duration > max_duration:
            # Split long audio
            segments = self.split_long_audio(audio, target_sr, max_duration)
            logger.info(f"Split into {len(segments)} segments (max: {max_duration}s)")
            
            for i, segment in enumerate(segments):
                output_path = self.output_dir / f"{base_name}_{model_name}_part{i+1:03d}.wav"
                if self.save_audio(segment, target_sr, str(output_path)):
                    output_files.append(str(output_path))
        
        else:
            # Save as single file
            output_path = self.output_dir / f"{base_name}_{model_name}.wav"
            if self.save_audio(audio, target_sr, str(output_path)):
                output_files.append(str(output_path))
        
        logger.info(f"Generated {len(output_files)} files for {model_name}")
        return output_files
    
    def preprocess_all_models(self, audio_path: str) -> Dict[str, List[str]]:
        """Preprocess audio for all models"""
        results = {}
        
        for model_name in self.model_requirements.keys():
            try:
                output_files = self.preprocess_for_model(audio_path, model_name)
                results[model_name] = output_files
            except Exception as e:
                logger.error(f"Failed to process {audio_path} for {model_name}: {e}")
                results[model_name] = []
        
        return results
    
    def process_directory(self, input_dir: str) -> Dict[str, Dict[str, List[str]]]:
        """Process all audio files in directory"""
        input_path = Path(input_dir)
        audio_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.mp3")) + \
                     list(input_path.glob("*.m4a")) + list(input_path.glob("*.flac"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return {}
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        all_results = {}
        for audio_file in audio_files:
            logger.info(f"Processing {audio_file.name}")
            results = self.preprocess_all_models(str(audio_file))
            all_results[audio_file.name] = results
        
        return all_results
    
    def generate_summary(self, results: Dict[str, Dict[str, List[str]]]) -> Dict:
        """Generate processing summary"""
        summary = {
            "total_files": len(results),
            "models": list(self.model_requirements.keys()),
            "model_stats": {},
            "file_stats": {}
        }
        
        # Model statistics
        for model_name in self.model_requirements.keys():
            total_outputs = sum(len(results[file][model_name]) for file in results)
            successful_files = sum(1 for file in results if results[file][model_name])
            summary["model_stats"][model_name] = {
                "total_output_files": total_outputs,
                "successful_input_files": successful_files,
                "success_rate": f"{successful_files}/{len(results)} ({successful_files/len(results)*100:.1f}%)"
            }
        
        # File statistics
        for file_name, file_results in results.items():
            summary["file_stats"][file_name] = {
                "total_output_files": sum(len(outputs) for outputs in file_results.values()),
                "models_processed": sum(1 for outputs in file_results.values() if outputs),
                "output_files_per_model": {model: len(outputs) for model, outputs in file_results.items()}
            }
        
        return summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Audio Preprocessor for ASR Model Compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed audio files')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'large-v3', 'canary-1b', 'parakeet-tdt-0.6b-v2', 'wav2vec-xls-r'],
                       help='Target model(s) for preprocessing')
    parser.add_argument('--summary_file', type=str, default='preprocessing_summary.json',
                       help='Output summary file path')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(args.output_dir)
    
    # Process files
    logger.info(f"Starting audio preprocessing...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    results = preprocessor.process_directory(args.input_dir)
    
    # Generate summary
    summary = preprocessor.generate_summary(results)
    
    # Save summary
    summary_path = Path(args.output_dir) / args.summary_file
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("AUDIO PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total input files: {summary['total_files']}")
    print(f"Output directory: {args.output_dir}")
    print(f"Summary file: {summary_path}")
    print()
    
    print("MODEL PROCESSING STATISTICS:")
    print("-" * 40)
    for model_name, stats in summary['model_stats'].items():
        print(f"{model_name:20} | {stats['success_rate']:15} | {stats['total_output_files']:3d} files")
    
    print()
    print("MODEL REQUIREMENTS:")
    print("-" * 40)
    for model_name, reqs in preprocessor.model_requirements.items():
        print(f"{model_name:20} | {reqs['description']}")
    
    print()
    print(f"Processing completed successfully!")
    print(f"Check {summary_path} for detailed results")


if __name__ == '__main__':
    main() 