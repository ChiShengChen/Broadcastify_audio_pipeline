#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Preprocessor for ASR Pipeline
==================================

This script provides audio preprocessing functionality including:
1. Upsampling audio files to target sample rate (e.g., 8000Hz -> 16000Hz)
2. Splitting long audio files into segments (e.g., >60s -> 60s segments)
3. Maintaining metadata for proper WER calculation

Usage:
    python3 audio_preprocessor.py --input_dir /path/to/audio --output_dir /path/to/output --target_sample_rate 16000 --max_duration 60
"""

import os
import torch
import torchaudio
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
from tqdm import tqdm
import shutil
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AudioPreprocessor:
    """Audio preprocessing with upsampling and segmentation"""
    
    def __init__(self, 
                 target_sample_rate: int = 16000,
                 max_duration: float = 60.0,  # Maximum duration per segment
                 overlap_duration: float = 1.0,  # Overlap between segments
                 min_segment_duration: float = 5.0,  # Minimum segment duration
                 preserve_original_structure: bool = True):
        """
        Initialize Audio Preprocessor
        
        Args:
            target_sample_rate: Target sample rate for upsampling
            max_duration: Maximum duration for each segment (seconds)
            overlap_duration: Overlap between segments (seconds)
            min_segment_duration: Minimum duration for valid segments (seconds)
            preserve_original_structure: Whether to preserve original directory structure
        """
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.overlap_duration = overlap_duration
        self.min_segment_duration = min_segment_duration
        self.preserve_original_structure = preserve_original_structure
        
        print("Audio Preprocessor initialized with parameters:")
        print(f"  - Target sample rate: {target_sample_rate}Hz")
        print(f"  - Max segment duration: {max_duration}s")
        print(f"  - Overlap duration: {overlap_duration}s")
        print(f"  - Min segment duration: {min_segment_duration}s")
        print(f"  - Preserve structure: {preserve_original_structure}")
    
    def get_audio_info(self, audio_path: str) -> Dict:
        """Get audio file information"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate
            
            return {
                'path': audio_path,
                'sample_rate': sample_rate,
                'duration': duration,
                'channels': waveform.shape[0],
                'file_size': os.path.getsize(audio_path),
                'needs_upsampling': sample_rate != self.target_sample_rate,
                'needs_splitting': duration > self.max_duration
            }
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None
    
    def upsample_audio(self, waveform: torch.Tensor, original_sample_rate: int) -> torch.Tensor:
        """
        Upsample audio to target sample rate
        
        Args:
            waveform: Input audio tensor
            original_sample_rate: Original sample rate
            
        Returns:
            Upsampled waveform
        """
        if original_sample_rate == self.target_sample_rate:
            return waveform
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample
        resampler = torchaudio.transforms.Resample(original_sample_rate, self.target_sample_rate)
        upsampled_waveform = resampler(waveform)
        
        return upsampled_waveform
    
    def split_audio(self, waveform: torch.Tensor, sample_rate: int) -> List[Tuple[torch.Tensor, float, float]]:
        """
        Split audio into segments
        
        Args:
            waveform: Input audio tensor
            sample_rate: Sample rate
            
        Returns:
            List of (segment_waveform, start_time, end_time) tuples
        """
        duration = waveform.shape[1] / sample_rate
        
        if duration <= self.max_duration:
            return [(waveform, 0.0, duration)]
        
        segments = []
        segment_samples = int(self.max_duration * sample_rate)
        overlap_samples = int(self.overlap_duration * sample_rate)
        step_samples = segment_samples - overlap_samples
        
        start_sample = 0
        while start_sample < waveform.shape[1]:
            end_sample = min(start_sample + segment_samples, waveform.shape[1])
            segment_waveform = waveform[:, start_sample:end_sample]
            
            segment_duration = segment_waveform.shape[1] / sample_rate
            
            # Only include segments that meet minimum duration
            if segment_duration >= self.min_segment_duration:
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                segments.append((segment_waveform, start_time, end_time))
            
            start_sample += step_samples
        
        return segments
    
    def process_audio_file(self, input_path: str, output_dir: str) -> Dict:
        """
        Process a single audio file
        
        Args:
            input_path: Input audio file path
            output_dir: Output directory
            
        Returns:
            Processing metadata
        """
        # Get audio info
        audio_info = self.get_audio_info(input_path)
        if audio_info is None:
            return {'status': 'error', 'message': 'Failed to read audio file'}
        
        # Load audio
        waveform, original_sample_rate = torchaudio.load(input_path)
        
        # Upsample if needed
        if audio_info['needs_upsampling']:
            waveform = self.upsample_audio(waveform, original_sample_rate)
            sample_rate = self.target_sample_rate
        else:
            sample_rate = original_sample_rate
        
        # Split audio if needed
        segments = self.split_audio(waveform, sample_rate)
        
        # Create output directory structure
        if self.preserve_original_structure:
            relative_path = os.path.relpath(input_path, self.input_dir)
            file_output_dir = os.path.join(output_dir, os.path.dirname(relative_path))
        else:
            file_output_dir = output_dir
        
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Save segments
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        saved_segments = []
        
        for i, (segment_waveform, start_time, end_time) in enumerate(segments):
            if len(segments) == 1:
                # Single segment, use original filename
                output_filename = f"{base_name}.wav"
            else:
                # Multiple segments, add segment number
                output_filename = f"{base_name}_segment_{i:03d}.wav"
            
            output_path = os.path.join(file_output_dir, output_filename)
            
            # Save segment
            torchaudio.save(output_path, segment_waveform, sample_rate)
            
            saved_segments.append({
                'filename': output_filename,
                'path': output_path,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'sample_rate': sample_rate
            })
        
        return {
            'status': 'success',
            'original_file': input_path,
            'original_duration': audio_info['duration'],
            'original_sample_rate': audio_info['sample_rate'],
            'needs_upsampling': audio_info['needs_upsampling'],
            'needs_splitting': audio_info['needs_splitting'],
            'segments': saved_segments,
            'total_segments': len(saved_segments)
        }
    
    def process_directory(self, input_dir: str, output_dir: str) -> Dict:
        """
        Process all audio files in directory
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            
        Returns:
            Processing summary
        """
        self.input_dir = input_dir
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        audio_files = []
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            return {'status': 'error', 'message': 'No audio files found'}
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Process files
        results = {
            'status': 'success',
            'total_files': len(audio_files),
            'processed_files': 0,
            'error_files': 0,
            'total_segments': 0,
            'upsampled_files': 0,
            'split_files': 0,
            'file_results': []
        }
        
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                result = self.process_audio_file(audio_file, output_dir)
                results['file_results'].append(result)
                
                if result['status'] == 'success':
                    results['processed_files'] += 1
                    results['total_segments'] += result['total_segments']
                    
                    if result['needs_upsampling']:
                        results['upsampled_files'] += 1
                    
                    if result['needs_splitting']:
                        results['split_files'] += 1
                else:
                    results['error_files'] += 1
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results['error_files'] += 1
                results['file_results'].append({
                    'status': 'error',
                    'original_file': audio_file,
                    'message': str(e)
                })
        
        # Save processing metadata
        metadata_file = os.path.join(output_dir, 'processing_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Audio Preprocessor for ASR Pipeline")
    parser.add_argument("--input_dir", required=True, help="Input directory with audio files")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed files")
    parser.add_argument("--target_sample_rate", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument("--max_duration", type=float, default=60.0, help="Maximum segment duration in seconds (default: 60.0)")
    parser.add_argument("--overlap_duration", type=float, default=1.0, help="Overlap between segments in seconds (default: 1.0)")
    parser.add_argument("--min_segment_duration", type=float, default=5.0, help="Minimum segment duration in seconds (default: 5.0)")
    parser.add_argument("--preserve_structure", action="store_true", help="Preserve original directory structure")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        target_sample_rate=args.target_sample_rate,
        max_duration=args.max_duration,
        overlap_duration=args.overlap_duration,
        min_segment_duration=args.min_segment_duration,
        preserve_original_structure=args.preserve_structure
    )
    
    # Process directory
    print(f"Processing audio files from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    results = preprocessor.process_directory(args.input_dir, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files: {results['total_files']}")
    print(f"Successfully processed: {results['processed_files']}")
    print(f"Errors: {results['error_files']}")
    print(f"Total segments created: {results['total_segments']}")
    print(f"Files upsampled: {results['upsampled_files']}")
    print(f"Files split: {results['split_files']}")
    print(f"Metadata saved to: {os.path.join(args.output_dir, 'processing_metadata.json')}")
    
    if results['status'] == 'success':
        print("\n✅ Processing completed successfully!")
    else:
        print(f"\n❌ Processing failed: {results['message']}")

if __name__ == "__main__":
    main() 