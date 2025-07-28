#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Filter Module
==================

This module provides audio filtering functionality including:
1. High-pass filter (removes low-frequency noise)
2. Band-pass filter (focuses on speech frequencies)
3. Wiener filter (noise reduction)
4. Combined filtering pipeline

Usage:
    python3 audio_filter.py --input_dir /path/to/audio --output_dir /path/to/output --enable-filters
"""

import os
import torch
import torchaudio
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import warnings
from tqdm import tqdm
import json

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AudioFilter:
    """Audio filtering with various filter types"""
    
    def __init__(self,
                 enable_filters: bool = True,
                 highpass_cutoff: float = 300.0,
                 lowcut: float = 300.0,
                 highcut: float = 3000.0,
                 filter_order: int = 5,
                 enable_wiener: bool = False,
                 target_sample_rate: int = 16000):
        """
        Initialize Audio Filter
        
        Args:
            enable_filters: Enable audio preprocessing filters
            highpass_cutoff: High-pass filter cutoff frequency (Hz)
            lowcut: Band-pass filter low cutoff (Hz)
            highcut: Band-pass filter high cutoff (Hz)
            filter_order: Filter order
            enable_wiener: Enable Wiener filter for noise reduction
            target_sample_rate: Target sample rate for resampling
        """
        self.enable_filters = enable_filters
        self.highpass_cutoff = highpass_cutoff
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.enable_wiener = enable_wiener
        self.target_sample_rate = target_sample_rate
        
        if enable_filters:
            print("Audio Filter initialized with parameters:")
            print(f"  - High-pass cutoff: {highpass_cutoff}Hz")
            print(f"  - Band-pass range: {lowcut}-{highcut}Hz")
            print(f"  - Filter order: {filter_order}")
            print(f"  - Wiener filter: {'Enabled' if enable_wiener else 'Disabled'}")
            print(f"  - Target sample rate: {target_sample_rate}Hz")
    
    def butter_highpass(self, cutoff, sample_rate, order=5):
        """Create Butterworth high-pass filter"""
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return b, a
    
    def butter_bandpass(self, lowcut, highcut, sample_rate, order=5):
        """Create Butterworth band-pass filter"""
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band", analog=False)
        return b, a
    
    def apply_highpass_filter(self, waveform, sample_rate):
        """Apply high-pass filter to remove low-frequency noise"""
        if not self.enable_filters:
            return waveform
            
        b, a = self.butter_highpass(self.highpass_cutoff, sample_rate, self.filter_order)
        filtered_waveform = lfilter(b, a, waveform.squeeze().numpy())
        return torch.tensor(filtered_waveform).unsqueeze(0)
    
    def apply_bandpass_filter(self, waveform, sample_rate):
        """Apply band-pass filter for speech enhancement"""
        if not self.enable_filters:
            return waveform
            
        b, a = self.butter_bandpass(self.lowcut, self.highcut, sample_rate, self.filter_order)
        filtered_waveform = lfilter(b, a, waveform.squeeze().numpy())
        return torch.tensor(filtered_waveform).unsqueeze(0)
    
    def apply_wiener_filter(self, waveform):
        """Apply Wiener filter for noise reduction"""
        if not self.enable_wiener:
            return waveform
            
        waveform_np = waveform.squeeze().numpy()
        filtered = scipy.signal.wiener(waveform_np, mysize=5, noise=1e-6)
        return torch.tensor(filtered).unsqueeze(0)
    
    def resample_audio(self, waveform, original_sample_rate):
        """Resample audio to target sample rate"""
        if original_sample_rate == self.target_sample_rate:
            return waveform, original_sample_rate
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample
        resampler = torchaudio.transforms.Resample(original_sample_rate, self.target_sample_rate)
        resampled_waveform = resampler(waveform)
        
        return resampled_waveform, self.target_sample_rate
    
    def filter_audio(self, waveform, sample_rate):
        """Apply all enabled filters to audio"""
        # Resample if needed
        waveform, sample_rate = self.resample_audio(waveform, sample_rate)
        
        if not self.enable_filters:
            return waveform, sample_rate
        
        print("Applying audio filters...")
        
        # 1. High-pass filter to remove low-frequency noise (e.g., AC hum)
        waveform = self.apply_highpass_filter(waveform, sample_rate)
        
        # 2. Band-pass filter for speech frequency range
        waveform = self.apply_bandpass_filter(waveform, sample_rate)
        
        # 3. Optional Wiener filter for noise reduction
        if self.enable_wiener:
            waveform = self.apply_wiener_filter(waveform)
        
        # Normalize amplitude after filtering
        max_amplitude = torch.max(torch.abs(waveform))
        if max_amplitude > 0:
            waveform = waveform / max_amplitude
        
        return waveform, sample_rate
    
    def process_audio_file(self, input_path: str, output_path: str) -> Dict:
        """
        Process a single audio file with filters
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            
        Returns:
            Processing metadata
        """
        try:
            # Load audio
            waveform, original_sample_rate = torchaudio.load(input_path)
            
            # Get original info
            original_duration = waveform.shape[1] / original_sample_rate
            original_channels = waveform.shape[0]
            
            # Apply filters
            filtered_waveform, final_sample_rate = self.filter_audio(waveform, original_sample_rate)
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save filtered audio
            torchaudio.save(output_path, filtered_waveform, final_sample_rate)
            
            return {
                'status': 'success',
                'input_file': input_path,
                'output_file': output_path,
                'original_duration': original_duration,
                'original_sample_rate': original_sample_rate,
                'original_channels': original_channels,
                'final_sample_rate': final_sample_rate,
                'final_channels': filtered_waveform.shape[0],
                'filters_applied': self.enable_filters,
                'highpass_applied': self.enable_filters,
                'bandpass_applied': self.enable_filters,
                'wiener_applied': self.enable_wiener
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'input_file': input_path,
                'error': str(e)
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
            'filters_enabled': self.enable_filters,
            'wiener_enabled': self.enable_wiener,
            'file_results': []
        }
        
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            # Create output path
            relative_path = os.path.relpath(audio_file, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            
            # Process file
            result = self.process_audio_file(audio_file, output_path)
            results['file_results'].append(result)
            
            if result['status'] == 'success':
                results['processed_files'] += 1
            else:
                results['error_files'] += 1
        
        # Save processing metadata
        metadata_file = os.path.join(output_dir, 'filter_processing_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results


def main():
    """CLI for audio filter"""
    parser = argparse.ArgumentParser(
        description="Audio Filter for ASR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with audio files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for filtered files')
    
    # Filter parameters
    parser.add_argument('--enable-filters', action='store_true', help='Enable audio filters')
    parser.add_argument('--no-filters', action='store_true', help='Disable audio filters')
    parser.add_argument('--highpass_cutoff', type=float, default=300.0, help='High-pass cutoff (Hz)')
    parser.add_argument('--lowcut', type=float, default=300.0, help='Band-pass low cutoff (Hz)')
    parser.add_argument('--highcut', type=float, default=3000.0, help='Band-pass high cutoff (Hz)')
    parser.add_argument('--filter_order', type=int, default=5, help='Filter order')
    parser.add_argument('--enable-wiener', action='store_true', help='Enable Wiener filter')
    parser.add_argument('--target_sample_rate', type=int, default=16000, help='Target sample rate (Hz)')
    
    args = parser.parse_args()
    
    # Determine if filters should be enabled
    enable_filters = args.enable_filters
    if args.no_filters:
        enable_filters = False
    
    # Initialize audio filter
    audio_filter = AudioFilter(
        enable_filters=enable_filters,
        highpass_cutoff=args.highpass_cutoff,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_order=args.filter_order,
        enable_wiener=args.enable_wiener,
        target_sample_rate=args.target_sample_rate
    )
    
    # Process directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Processing audio files from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    results = audio_filter.process_directory(args.input_dir, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("FILTERING SUMMARY")
    print("="*50)
    print(f"Total files: {results['total_files']}")
    print(f"Successfully processed: {results['processed_files']}")
    print(f"Errors: {results['error_files']}")
    print(f"Filters enabled: {results['filters_enabled']}")
    print(f"Wiener filter enabled: {results['wiener_enabled']}")
    print(f"Metadata saved to: {os.path.join(args.output_dir, 'filter_processing_metadata.json')}")
    
    if results['status'] == 'success':
        print("\n✅ Audio filtering completed successfully!")
    else:
        print(f"\n❌ Processing failed: {results['message']}")


if __name__ == '__main__':
    main() 