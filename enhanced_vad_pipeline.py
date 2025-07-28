#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced VAD Pipeline with Audio Filtering
==========================================

This enhanced version includes optional audio preprocessing filters
before VAD processing for improved speech detection in noisy environments.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
import argparse
from pathlib import Path

# Import the base VAD pipeline
from vad_pipeline import VADPipeline

class EnhancedVADPipeline(VADPipeline):
    """Enhanced VAD Pipeline with audio preprocessing filters"""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 speech_threshold: float = 0.5,
                 min_speech_duration: float = 0.5,
                 min_silence_duration: float = 0.3,
                 target_sample_rate: int = 16000,
                 # Audio filter parameters
                 enable_filters: bool = True,
                 highpass_cutoff: float = 300.0,
                 lowcut: float = 300.0,
                 highcut: float = 3000.0,
                 filter_order: int = 5,
                 enable_wiener: bool = False):
        """
        Initialize Enhanced VAD Pipeline with audio filters
        
        Additional Args:
            enable_filters: Enable audio preprocessing filters
            highpass_cutoff: High-pass filter cutoff frequency (Hz)
            lowcut: Band-pass filter low cutoff (Hz)
            highcut: Band-pass filter high cutoff (Hz) 
            filter_order: Filter order
            enable_wiener: Enable Wiener filter for noise reduction
        """
        super().__init__(chunk_size, speech_threshold, min_speech_duration,
                         min_silence_duration, target_sample_rate)
        
        self.enable_filters = enable_filters
        self.highpass_cutoff = highpass_cutoff
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.enable_wiener = enable_wiener
        
        if enable_filters:
            print("Enhanced VAD with audio filters:")
            print(f"  - High-pass cutoff: {highpass_cutoff}Hz")
            print(f"  - Band-pass range: {lowcut}-{highcut}Hz")
            print(f"  - Filter order: {filter_order}")
            print(f"  - Wiener filter: {'Enabled' if enable_wiener else 'Disabled'}")
    
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
    
    def apply_speech_enhance_filter(self, waveform, sample_rate):
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
    
    def preprocess_audio(self, waveform, sample_rate):
        """Enhanced audio preprocessing with filters"""
        # Standard preprocessing (mono conversion + resampling)
        waveform, sample_rate = super().preprocess_audio(waveform, sample_rate)
        
        if not self.enable_filters:
            return waveform, sample_rate
        
        # Apply audio enhancement filters
        print("Applying audio enhancement filters...")
        
        # 1. High-pass filter to remove low-frequency noise (e.g., AC hum)
        waveform = self.apply_highpass_filter(waveform, sample_rate)
        
        # 2. Band-pass filter for speech frequency range
        waveform = self.apply_speech_enhance_filter(waveform, sample_rate)
        
        # 3. Optional Wiener filter for noise reduction
        if self.enable_wiener:
            waveform = self.apply_wiener_filter(waveform)
        
        # Normalize amplitude after filtering
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform, sample_rate
    
    def skip_audio_filtering(self):
        """Skip audio filtering if already done by separate filter module"""
        self.enable_filters = False
        print("Audio filtering disabled - assuming already filtered by separate module")


def main():
    """CLI for enhanced VAD pipeline"""
    parser = argparse.ArgumentParser(
        description="Enhanced VAD Pipeline with Audio Filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Base arguments
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    # VAD parameters
    parser.add_argument('--speech_threshold', type=float, default=0.5, help='Speech threshold')
    parser.add_argument('--min_speech_duration', type=float, default=0.5, help='Min speech duration')
    parser.add_argument('--min_silence_duration', type=float, default=0.3, help='Min silence duration')
    
    # Filter parameters
    parser.add_argument('--no-filters', action='store_true', help='Disable audio filters')
    parser.add_argument('--highpass_cutoff', type=float, default=300.0, help='High-pass cutoff (Hz)')
    parser.add_argument('--lowcut', type=float, default=300.0, help='Band-pass low cutoff (Hz)')
    parser.add_argument('--highcut', type=float, default=3000.0, help='Band-pass high cutoff (Hz)')
    parser.add_argument('--filter_order', type=int, default=5, help='Filter order')
    parser.add_argument('--enable-wiener', action='store_true', help='Enable Wiener filter')
    
    args = parser.parse_args()
    
    # Initialize enhanced VAD pipeline
    vad = EnhancedVADPipeline(
        speech_threshold=args.speech_threshold,
        min_speech_duration=args.min_speech_duration,
        min_silence_duration=args.min_silence_duration,
        enable_filters=not args.no_filters,
        highpass_cutoff=args.highpass_cutoff,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_order=args.filter_order,
        enable_wiener=args.enable_wiener
    )
    
    # Process directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    vad.process_directory(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main() 