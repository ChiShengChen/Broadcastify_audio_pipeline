#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long Audio Splitter for OOM Prevention
======================================

This script splits long audio files (>2 minutes) into smaller segments using VAD
to avoid out-of-memory issues during ASR processing. It maintains timestamp
mapping and ensures segments are properly organized for later consolidation.

Usage:
    python3 long_audio_splitter.py --input_dir /path/to/audio --output_dir /path/to/output --max_duration 120
"""

import os
import torch
import torchaudio
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
from tqdm import tqdm
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class LongAudioSplitter:
    """Split long audio files into smaller segments using VAD"""
    
    def __init__(self, 
                 max_duration: float = 120.0,  # 2 minutes in seconds
                 speech_threshold: float = 0.5,
                 min_speech_duration: float = 0.5,
                 min_silence_duration: float = 0.3,
                 target_sample_rate: int = 16000):
        """
        Initialize Long Audio Splitter
        
        Args:
            max_duration: Maximum duration for each segment (seconds)
            speech_threshold: VAD speech detection threshold (0.0-1.0)
            min_speech_duration: Minimum duration for valid speech segments (seconds)
            min_silence_duration: Minimum silence to separate speech segments (seconds)
            target_sample_rate: Target sample rate for processing
        """
        self.max_duration = max_duration
        self.speech_threshold = speech_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.target_sample_rate = target_sample_rate
        self.vad_model = None
        
        print("Long Audio Splitter initialized with parameters:")
        print(f"  - Max segment duration: {max_duration}s")
        print(f"  - Speech threshold: {speech_threshold}")
        print(f"  - Min speech duration: {min_speech_duration}s")
        print(f"  - Min silence duration: {min_silence_duration}s")
        print(f"  - Target sample rate: {target_sample_rate}Hz")
    
    def load_vad_model(self):
        """Load Silero VAD model"""
        if self.vad_model is None:
            print("Loading Silero VAD model...")
            try:
                self.vad_model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                print("VAD model loaded successfully!")
            except Exception as e:
                print(f"Error loading VAD model: {e}")
                raise e
    
    def preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio for VAD
        
        Args:
            waveform: Input audio tensor
            sample_rate: Original sample rate
            
        Returns:
            Preprocessed waveform and sample rate
        """
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate
        
        return waveform, sample_rate
    
    def detect_speech_segments(self, waveform: torch.Tensor, sample_rate: int) -> List[Tuple[int, int]]:
        """
        Detect speech segments using VAD
        
        Args:
            waveform: Input audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            List of (start_sample, end_sample) tuples
        """
        self.load_vad_model()
        
        # Convert to numpy for VAD processing
        audio_numpy = waveform.squeeze().numpy()
        
        # Process audio in chunks for VAD
        chunk_size = 512 if sample_rate == 16000 else 256
        speech_probs = []
        
        for i in range(0, len(audio_numpy), chunk_size):
            chunk = audio_numpy[i:i + chunk_size]
            if len(chunk) == chunk_size:  # Only process full chunks
                chunk_tensor = torch.tensor(chunk)
                prob = self.vad_model(chunk_tensor, sample_rate)
                speech_probs.extend(prob.detach().numpy())
        
        # Convert to binary speech mask
        speech_mask = [prob > self.speech_threshold for prob in speech_probs]
        
        # Find speech segments
        segments = []
        in_speech = False
        start_sample = 0
        
        for i, is_speech in enumerate(speech_mask):
            if is_speech and not in_speech:
                # Start of speech
                start_sample = i * chunk_size
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech
                end_sample = i * chunk_size
                duration = (end_sample - start_sample) / sample_rate
                
                if duration >= self.min_speech_duration:
                    segments.append((start_sample, end_sample))
                
                in_speech = False
        
        # Handle case where speech continues to end of file
        if in_speech:
            end_sample = len(speech_mask) * chunk_size
            duration = (end_sample - start_sample) / sample_rate
            if duration >= self.min_speech_duration:
                segments.append((start_sample, end_sample))
        
        return segments
    
    def split_audio_by_duration(self, waveform: torch.Tensor, sample_rate: int) -> List[Tuple[torch.Tensor, float, float]]:
        """
        Split audio into segments based on max duration
        
        Args:
            waveform: Input audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            List of (segment_waveform, start_time, end_time) tuples
        """
        total_duration = waveform.shape[1] / sample_rate
        
        if total_duration <= self.max_duration:
            # No need to split
            return [(waveform, 0.0, total_duration)]
        
        segments = []
        max_samples = int(self.max_duration * sample_rate)
        
        for i in range(0, waveform.shape[1], max_samples):
            end_sample = min(i + max_samples, waveform.shape[1])
            segment_waveform = waveform[:, i:end_sample]
            
            start_time = i / sample_rate
            end_time = end_sample / sample_rate
            
            segments.append((segment_waveform, start_time, end_time))
        
        return segments
    
    def process_audio_file(self, input_path: str, output_dir: str) -> Dict:
        """
        Process a single audio file
        
        Args:
            input_path: Path to input audio file
            output_dir: Output directory for segments
            
        Returns:
            Processing metadata
        """
        print(f"Processing: {os.path.basename(input_path)}")
        
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(input_path)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            return {"error": str(e)}
        
        # Preprocess audio
        waveform, sample_rate = self.preprocess_audio(waveform, sample_rate)
        total_duration = waveform.shape[1] / sample_rate
        
        print(f"  - Duration: {total_duration:.2f}s")
        
        # Check if splitting is needed
        if total_duration <= self.max_duration:
            print(f"  - No splitting needed (duration <= {self.max_duration}s)")
            return {"split": False, "original_duration": total_duration}
        
        print(f"  - Splitting required (duration > {self.max_duration}s)")
        
        # Detect speech segments
        speech_segments = self.detect_speech_segments(waveform, sample_rate)
        print(f"  - Detected {len(speech_segments)} speech segments")
        
        # Split audio by duration
        duration_segments = self.split_audio_by_duration(waveform, sample_rate)
        print(f"  - Created {len(duration_segments)} duration-based segments")
        
        # Create output directory for this file
        base_name = Path(input_path).stem
        file_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Save segments and metadata
        segment_info = []
        for i, (segment_waveform, start_time, end_time) in enumerate(duration_segments):
            segment_filename = f"{base_name}_segment_{i:03d}.wav"
            segment_path = os.path.join(file_output_dir, segment_filename)
            
            # Save segment
            torchaudio.save(segment_path, segment_waveform, sample_rate)
            
            # Find speech segments that fall within this duration segment
            segment_speech_info = []
            for speech_start, speech_end in speech_segments:
                speech_start_time = speech_start / sample_rate
                speech_end_time = speech_end / sample_rate
                
                # Check if speech segment overlaps with duration segment
                if (speech_start_time < end_time and speech_end_time > start_time):
                    # Calculate overlap
                    overlap_start = max(speech_start_time, start_time)
                    overlap_end = min(speech_end_time, end_time)
                    
                    segment_speech_info.append({
                        "original_start": speech_start_time,
                        "original_end": speech_end_time,
                        "segment_start": overlap_start - start_time,
                        "segment_end": overlap_end - start_time
                    })
            
            segment_info.append({
                "segment_id": i,
                "filename": segment_filename,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "speech_segments": segment_speech_info
            })
        
        # Save metadata
        metadata = {
            "original_file": os.path.basename(input_path),
            "original_duration": total_duration,
            "max_segment_duration": self.max_duration,
            "total_segments": len(duration_segments),
            "segments": segment_info
        }
        
        metadata_path = os.path.join(file_output_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  - Saved {len(duration_segments)} segments to {file_output_dir}")
        
        return {
            "split": True,
            "original_duration": total_duration,
            "segments_created": len(duration_segments),
            "output_dir": file_output_dir,
            "metadata_file": metadata_path
        }
    
    def process_directory(self, input_dir: str, output_dir: str) -> Dict:
        """
        Process all audio files in a directory
        
        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for processed segments
            
        Returns:
            Processing summary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(input_dir).glob(f"*{ext}"))
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return {"error": "No audio files found"}
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process each file
        results = {
            "total_files": len(audio_files),
            "split_files": 0,
            "unsplit_files": 0,
            "errors": 0,
            "file_results": []
        }
        
        for audio_file in tqdm(audio_files, desc="Processing files"):
            result = self.process_audio_file(str(audio_file), output_dir)
            results["file_results"].append(result)
            
            if "error" in result:
                results["errors"] += 1
            elif result.get("split", False):
                results["split_files"] += 1
            else:
                results["unsplit_files"] += 1
        
        # Save processing summary
        summary_path = os.path.join(output_dir, "processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProcessing Summary:")
        print(f"  - Total files: {results['total_files']}")
        print(f"  - Files split: {results['split_files']}")
        print(f"  - Files unchanged: {results['unsplit_files']}")
        print(f"  - Errors: {results['errors']}")
        print(f"  - Summary saved to: {summary_path}")
        
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Split long audio files to prevent OOM issues")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for segments")
    parser.add_argument("--max_duration", type=float, default=120.0, help="Maximum segment duration in seconds")
    parser.add_argument("--speech_threshold", type=float, default=0.5, help="VAD speech threshold")
    parser.add_argument("--min_speech_duration", type=float, default=0.5, help="Minimum speech duration")
    parser.add_argument("--min_silence_duration", type=float, default=0.3, help="Minimum silence duration")
    
    args = parser.parse_args()
    
    # Initialize splitter
    splitter = LongAudioSplitter(
        max_duration=args.max_duration,
        speech_threshold=args.speech_threshold,
        min_speech_duration=args.min_speech_duration,
        min_silence_duration=args.min_silence_duration
    )
    
    # Process directory
    splitter.process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 