#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAD (Voice Activity Detection) Pipeline
=======================================

This script provides a pure VAD pipeline using Silero VAD to extract speech segments
from audio files before ASR processing. It can process both individual files and
entire directories.

The pipeline now supports concatenating speech segments back into a continuous audio
file while maintaining timestamp mapping for easier ASR processing and verification.

Usage:
    python3 ems_call/vad_pipeline.py --input_dir /path/to/audio --output_dir /path/to/output
    python3 ems_call/vad_pipeline.py --input_file /path/to/file.wav --output_dir /path/to/output
"""

import os
import torch
import torchaudio
import torchaudio.transforms as T
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class VADPipeline:
    """Voice Activity Detection Pipeline using Silero VAD"""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 speech_threshold: float = 0.5,
                 min_speech_duration: float = 0.5,
                 min_silence_duration: float = 0.3,
                 target_sample_rate: int = 16000,
                 concatenate_segments: bool = True):
        """
        Initialize VAD Pipeline
        
        Args:
            chunk_size: Size of audio chunks for VAD processing (samples)
            speech_threshold: Threshold for speech detection (0.0-1.0)
            min_speech_duration: Minimum duration for valid speech segments (seconds)
            min_silence_duration: Minimum silence to separate speech segments (seconds)
            target_sample_rate: Target sample rate for processing
            concatenate_segments: Whether to concatenate segments into a single file
        """
        self.chunk_size = chunk_size
        self.speech_threshold = speech_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.target_sample_rate = target_sample_rate
        self.concatenate_segments = concatenate_segments
        self.vad_model = None
        
        print("VAD Pipeline initialized with parameters:")
        print(f"  - Chunk size: {chunk_size}")
        print(f"  - Speech threshold: {speech_threshold}")
        print(f"  - Min speech duration: {min_speech_duration}s")
        print(f"  - Min silence duration: {min_silence_duration}s")
        print(f"  - Target sample rate: {target_sample_rate}Hz")
        print(f"  - Concatenate segments: {concatenate_segments}")
    
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
        
        # Resample to target sample rate if needed
        if sample_rate != self.target_sample_rate:
            print(f"Resampling from {sample_rate}Hz to {self.target_sample_rate}Hz...")
            resampler = T.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate
        
        return waveform, sample_rate
    
    def detect_speech_segments(self, waveform: torch.Tensor, sample_rate: int) -> List[Tuple[int, int]]:
        """
        Detect speech segments in audio waveform
        
        Args:
            waveform: Audio tensor [1, samples]
            sample_rate: Sample rate
            
        Returns:
            List of (start_sample, end_sample) tuples for speech segments
        """
        if self.vad_model is None:
            self.load_vad_model()
        
        speech_frames = []
        total_chunks = waveform.shape[1] // self.chunk_size
        
        # Process audio in chunks
        for frame_idx in range(total_chunks):
            start_sample = frame_idx * self.chunk_size
            end_sample = start_sample + self.chunk_size
            chunk = waveform[:, start_sample:end_sample]
            
            # Pad chunk if necessary
            if chunk.shape[1] < self.chunk_size:
                padding = torch.zeros((1, self.chunk_size - chunk.shape[1]))
                chunk = torch.cat((chunk, padding), dim=1)
            
            try:
                # Get VAD confidence score
                vad_score = self.vad_model(chunk, sample_rate).item()
                is_speech = vad_score > self.speech_threshold
                speech_frames.append(is_speech)
            except Exception as e:
                print(f"Error during VAD processing: {e}")
                speech_frames.append(False)
        
        # Convert frame-level decisions to segments
        segments = []
        current_start = None
        min_speech_frames = int(self.min_speech_duration * sample_rate / self.chunk_size)
        min_silence_frames = int(self.min_silence_duration * sample_rate / self.chunk_size)
        
        silence_counter = 0
        
        for frame_idx, is_speech in enumerate(speech_frames):
            if is_speech:
                if current_start is None:
                    current_start = frame_idx
                silence_counter = 0
            else:
                silence_counter += 1
                if current_start is not None and silence_counter >= min_silence_frames:
                    # End current segment
                    segment_length = frame_idx - current_start
                    if segment_length >= min_speech_frames:
                        start_sample = current_start * self.chunk_size
                        end_sample = (frame_idx - silence_counter) * self.chunk_size
                        segments.append((start_sample, end_sample))
                    current_start = None
                    silence_counter = 0
        
        # Handle segment that extends to end of audio
        if current_start is not None:
            segment_length = len(speech_frames) - current_start
            if segment_length >= min_speech_frames:
                start_sample = current_start * self.chunk_size
                end_sample = len(speech_frames) * self.chunk_size
                segments.append((start_sample, min(end_sample, waveform.shape[1])))
        
        return segments
    
    def concatenate_speech_segments(self, waveform: torch.Tensor, segments: List[Tuple[int, int]], sample_rate: int) -> Tuple[torch.Tensor, List[dict]]:
        """
        Concatenate speech segments into a continuous audio file
        
        Args:
            waveform: Original audio tensor
            segments: List of (start_sample, end_sample) tuples
            sample_rate: Sample rate
            
        Returns:
            Concatenated audio tensor and timestamp mapping
        """
        if not segments:
            # Return empty audio if no segments found
            empty_audio = torch.zeros((1, 0))
            return empty_audio, []
        
        concatenated_segments = []
        timestamp_mapping = []
        current_concatenated_time = 0.0
        
        for idx, (start_sample, end_sample) in enumerate(segments):
            # Extract segment
            segment = waveform[:, start_sample:end_sample]
            concatenated_segments.append(segment)
            
            # Calculate original timestamps
            original_start_time = start_sample / sample_rate
            original_end_time = end_sample / sample_rate
            segment_duration = (end_sample - start_sample) / sample_rate
            
            # Create timestamp mapping
            timestamp_mapping.append({
                'segment_id': idx + 1,
                'original_start_time': round(original_start_time, 3),
                'original_end_time': round(original_end_time, 3),
                'concatenated_start_time': round(current_concatenated_time, 3),
                'concatenated_end_time': round(current_concatenated_time + segment_duration, 3),
                'duration': round(segment_duration, 3)
            })
            
            current_concatenated_time += segment_duration
        
        # Concatenate all segments
        concatenated_audio = torch.cat(concatenated_segments, dim=1)
        
        return concatenated_audio, timestamp_mapping
    
    def process_audio_file(self, 
                          input_path: str, 
                          output_dir: str, 
                          save_individual_segments: bool = False,
                          save_metadata: bool = True) -> dict:
        """
        Process a single audio file with VAD
        
        Args:
            input_path: Path to input audio file
            output_dir: Output directory
            save_individual_segments: Whether to save individual speech segments
            save_metadata: Whether to save metadata file
            
        Returns:
            Processing results dictionary
        """
        print(f"Processing: {input_path}")
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(input_path)
            original_duration = waveform.shape[1] / sample_rate
            
            # Preprocess audio
            waveform, sample_rate = self.preprocess_audio(waveform, sample_rate)
            
            # Detect speech segments
            segments = self.detect_speech_segments(waveform, sample_rate)
            
            # Create output directory
            file_stem = Path(input_path).stem
            file_output_dir = Path(output_dir) / file_stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize variables
            total_speech_duration = 0
            segment_info = []
            timestamp_mapping = []
            
            if segments:
                # Calculate total speech duration
                for start_sample, end_sample in segments:
                    segment_duration = (end_sample - start_sample) / sample_rate
                    total_speech_duration += segment_duration
                
                if self.concatenate_segments:
                    # Concatenate segments into continuous audio
                    concatenated_audio, timestamp_mapping = self.concatenate_speech_segments(
                        waveform, segments, sample_rate
                    )
                    
                    if concatenated_audio.shape[1] > 0:  # Only save if not empty
                        # Save concatenated audio with original filename
                        concatenated_path = file_output_dir / f"{file_stem}_vad.wav"
                        torchaudio.save(str(concatenated_path), concatenated_audio, sample_rate)
                        print(f"  - Saved concatenated audio: {concatenated_path}")
                
                # Optionally save individual segments
                if save_individual_segments:
                    for idx, (start_sample, end_sample) in enumerate(segments):
                        segment = waveform[:, start_sample:end_sample]
                        segment_duration = (end_sample - start_sample) / sample_rate
                        
                        # Save segment
                        segment_path = file_output_dir / f"segment_{idx+1:03d}.wav"
                        torchaudio.save(str(segment_path), segment, sample_rate)
                        
                        # Record segment info
                        start_time = start_sample / sample_rate
                        end_time = end_sample / sample_rate
                        segment_info.append({
                            'segment_id': idx + 1,
                            'start_time': round(start_time, 3),
                            'end_time': round(end_time, 3),
                            'duration': round(segment_duration, 3),
                            'file_path': str(segment_path.relative_to(output_dir))
                        })
            
            # Create metadata
            metadata = {
                'input_file': str(Path(input_path).name),
                'original_duration': round(original_duration, 3),
                'total_speech_duration': round(total_speech_duration, 3),
                'speech_ratio': round(total_speech_duration / original_duration, 3) if original_duration > 0 else 0,
                'num_segments': len(segments),
                'sample_rate': sample_rate,
                'concatenated': self.concatenate_segments,
                'vad_parameters': {
                    'chunk_size': self.chunk_size,
                    'speech_threshold': self.speech_threshold,
                    'min_speech_duration': self.min_speech_duration,
                    'min_silence_duration': self.min_silence_duration
                },
                'timestamp_mapping': timestamp_mapping,
                'individual_segments': segment_info
            }
            
            # Add concatenated file info to metadata
            if self.concatenate_segments and segments:
                concatenated_path = file_output_dir / f"{file_stem}_vad.wav"
                if concatenated_path.exists():
                    metadata['concatenated_file'] = {
                        'file_path': str(concatenated_path.relative_to(output_dir)),
                        'duration': round(total_speech_duration, 3),
                        'timestamp_mapping_count': len(timestamp_mapping)
                    }
            
            # Save metadata
            if save_metadata:
                metadata_path = file_output_dir / f"{file_stem}_vad_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"  - Found {len(segments)} speech segments")
            print(f"  - Total speech: {total_speech_duration:.2f}s / {original_duration:.2f}s ({metadata['speech_ratio']:.1%})")
            if self.concatenate_segments and segments:
                print(f"  - Concatenated into single file with timestamp mapping")
            
            return metadata
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return {'error': str(e), 'input_file': input_path}
    
    def process_directory(self, input_dir: str, output_dir: str, extensions: List[str] = None) -> dict:
        """
        Process all audio files in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            extensions: List of file extensions to process
            
        Returns:
            Processing summary
        """
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.m4a']
        
        # Find all audio files
        input_path = Path(input_dir)
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_path.rglob(f'*{ext}'))
            audio_files.extend(input_path.rglob(f'*{ext.upper()}'))
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return {'error': 'No audio files found'}
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Process files
        results = []
        successful = 0
        failed = 0
        total_original_duration = 0
        total_speech_duration = 0
        
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            result = self.process_audio_file(str(audio_file), output_dir)
            results.append(result)
            
            if 'error' not in result:
                successful += 1
                total_original_duration += result['original_duration']
                total_speech_duration += result['total_speech_duration']
            else:
                failed += 1
        
        # Create summary
        summary = {
            'total_files': len(audio_files),
            'successful': successful,
            'failed': failed,
            'total_original_duration': round(total_original_duration, 3),
            'total_speech_duration': round(total_speech_duration, 3),
            'overall_speech_ratio': round(total_speech_duration / total_original_duration, 3) if total_original_duration > 0 else 0,
            'concatenated': self.concatenate_segments,
            'results': results
        }
        
        # Save summary
        summary_path = Path(output_dir) / 'vad_processing_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== VAD Processing Summary ===")
        print(f"Total files: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total original duration: {summary['total_original_duration']:.2f}s")
        print(f"Total speech duration: {summary['total_speech_duration']:.2f}s")
        print(f"Overall speech ratio: {summary['overall_speech_ratio']:.1%}")
        if self.concatenate_segments:
            print(f"Speech segments concatenated into single files per input")
        print(f"Summary saved to: {summary_path}")
        
        return summary


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="VAD Pipeline - Extract and optionally concatenate speech segments from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a directory with concatenated output (default)
    python3 ems_call/vad_pipeline.py --input_dir /path/to/audio --output_dir /path/to/output
    
    # Process with individual segments only (no concatenation)
    python3 ems_call/vad_pipeline.py --input_dir /path/to/audio --output_dir /path/to/output --no_concatenate --save_individual_segments
    
    # Process a single file
    python3 ems_call/vad_pipeline.py --input_file /path/to/file.wav --output_dir /path/to/output
    
    # Custom parameters
    python3 ems_call/vad_pipeline.py --input_dir /path/to/audio --output_dir /path/to/output --speech_threshold 0.7 --min_speech_duration 1.0
        """
    )
    
    # Input/Output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_dir', type=str, help='Input directory containing audio files')
    input_group.add_argument('--input_file', type=str, help='Single input audio file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed segments')
    
    # VAD Parameters
    parser.add_argument('--chunk_size', type=int, default=512, help='Chunk size for VAD processing (default: 512)')
    parser.add_argument('--speech_threshold', type=float, default=0.5, help='Speech detection threshold (default: 0.5)')
    parser.add_argument('--min_speech_duration', type=float, default=0.5, help='Minimum speech duration in seconds (default: 0.5)')
    parser.add_argument('--min_silence_duration', type=float, default=0.3, help='Minimum silence duration in seconds (default: 0.3)')
    parser.add_argument('--target_sample_rate', type=int, default=16000, help='Target sample rate (default: 16000)')
    
    # Output Options
    parser.add_argument('--no_concatenate', action='store_true', help='Disable concatenating segments into single file')
    parser.add_argument('--save_individual_segments', action='store_true', help='Save individual speech segments as separate files')
    parser.add_argument('--extensions', nargs='+', default=['.wav', '.mp3', '.flac', '.m4a'], 
                       help='Audio file extensions to process (default: .wav .mp3 .flac .m4a)')
    parser.add_argument('--no_save_metadata', action='store_true', help='Skip saving metadata files')
    
    args = parser.parse_args()
    
    # Create VAD pipeline
    vad = VADPipeline(
        chunk_size=args.chunk_size,
        speech_threshold=args.speech_threshold,
        min_speech_duration=args.min_speech_duration,
        min_silence_duration=args.min_silence_duration,
        target_sample_rate=args.target_sample_rate,
        concatenate_segments=not args.no_concatenate
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process input
    if args.input_dir:
        # Process directory
        vad.process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            extensions=args.extensions
        )
    else:
        # Process single file
        vad.process_audio_file(
            input_path=args.input_file,
            output_dir=args.output_dir,
            save_individual_segments=args.save_individual_segments,
            save_metadata=not args.no_save_metadata
        )


if __name__ == '__main__':
    main() 