#!/usr/bin/env python3
"""
Script to create long audio dataset for testing audio splitting functionality.
Concatenates every 3 audio files and generates corresponding ground truth annotations.
"""

import os
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import argparse
import json
from collections import defaultdict

def load_audio_file(file_path):
    """Load audio file and return audio data and sample rate."""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def concatenate_audio_files(file_paths, output_path, target_sr=16000):
    """Concatenate multiple audio files into one."""
    concatenated_audio = []
    max_sr = 0
    
    # First pass: determine the maximum sample rate
    for file_path in file_paths:
        audio, sr = load_audio_file(file_path)
        if audio is not None:
            max_sr = max(max_sr, sr)
    
    if max_sr == 0:
        print(f"Failed to load any audio files for {output_path}")
        return False
    
    # Second pass: resample and concatenate
    for file_path in file_paths:
        audio, sr = load_audio_file(file_path)
        if audio is not None:
            # Resample if necessary
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            concatenated_audio.append(audio)
    
    if not concatenated_audio:
        print(f"No valid audio files found for {output_path}")
        return False
    
    # Concatenate all audio segments
    final_audio = np.concatenate(concatenated_audio)
    
    # Save concatenated audio
    try:
        sf.write(output_path, final_audio, target_sr)
        return True
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        return False

def create_long_audio_dataset(input_dir, ground_truth_file, output_dir, group_size=3):
    """Create long audio dataset by concatenating groups of audio files."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth annotations
    print(f"Loading ground truth from: {ground_truth_file}")
    try:
        gt_df = pd.read_csv(ground_truth_file)
        print(f"Loaded {len(gt_df)} ground truth entries")
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return False
    
    # Get all wav files from input directory
    input_path = Path(input_dir)
    wav_files = sorted(list(input_path.glob("*.wav")))
    print(f"Found {len(wav_files)} WAV files in {input_dir}")
    
    if len(wav_files) == 0:
        print("No WAV files found in input directory")
        return False
    
    # Group files into chunks of group_size
    file_groups = []
    for i in range(0, len(wav_files), group_size):
        group = wav_files[i:i + group_size]
        file_groups.append(group)
    
    print(f"Created {len(file_groups)} groups of {group_size} files each")
    
    # Process each group
    new_annotations = []
    processing_summary = {
        "total_groups": len(file_groups),
        "successful_groups": 0,
        "failed_groups": 0,
        "total_duration": 0.0,
        "group_details": []
    }
    
    for group_idx, file_group in enumerate(file_groups):
        print(f"\nProcessing group {group_idx + 1}/{len(file_groups)}")
        
        # Create output filename
        base_names = [f.stem for f in file_group]
        output_filename = f"long_audio_group_{group_idx + 1:03d}.wav"
        output_filepath = output_path / output_filename
        
        # Get original filenames for ground truth lookup
        original_filenames = [f.name for f in file_group]
        
        # Find ground truth entries for these files
        group_transcripts = []
        total_duration = 0.0
        
        for filename in original_filenames:
            # Look for exact match or filename without extension
            filename_no_ext = Path(filename).stem
            matches = gt_df[gt_df['Filename'].str.contains(filename_no_ext, regex=False, na=False)]
            
            if len(matches) > 0:
                transcript = matches.iloc[0]['transcript']
                group_transcripts.append(transcript)
                print(f"  - Found transcript for {filename}: {transcript[:50]}...")
            else:
                print(f"  - No transcript found for {filename}")
                group_transcripts.append("")  # Empty transcript for missing files
        
        # Concatenate audio files
        print(f"  - Concatenating {len(file_group)} files...")
        success = concatenate_audio_files(file_group, output_filepath)
        
        if success:
            # Calculate duration
            try:
                audio, sr = librosa.load(output_filepath, sr=None)
                duration = len(audio) / sr
                total_duration += duration
            except:
                duration = 0.0
            
            # Create combined transcript
            combined_transcript = " ".join([t for t in group_transcripts if t.strip()])
            
            # Add to new annotations
            new_annotations.append({
                'Filename': output_filename,
                'transcript': combined_transcript,
                'original_files': original_filenames,
                'duration': duration,
                'group_id': group_idx + 1
            })
            
            processing_summary["successful_groups"] += 1
            processing_summary["total_duration"] += duration
            
            print(f"  - Success: {output_filename} ({duration:.2f}s)")
            print(f"  - Combined transcript: {combined_transcript[:100]}...")
        else:
            processing_summary["failed_groups"] += 1
            print(f"  - Failed to create {output_filename}")
        
        # Add group details to summary
        processing_summary["group_details"].append({
            "group_id": group_idx + 1,
            "output_filename": output_filename,
            "original_files": original_filenames,
            "success": success,
            "duration": duration if success else 0.0
        })
    
    # Save new ground truth file
    new_gt_df = pd.DataFrame(new_annotations)
    new_gt_file = output_path / "long_audio_ground_truth.csv"
    new_gt_df.to_csv(new_gt_file, index=False)
    
    # Save processing summary
    summary_file = output_path / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(processing_summary, f, indent=2)
    
    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Total groups: {processing_summary['total_groups']}")
    print(f"Successful: {processing_summary['successful_groups']}")
    print(f"Failed: {processing_summary['failed_groups']}")
    print(f"Total duration: {processing_summary['total_duration']:.2f} seconds")
    print(f"Average duration per group: {processing_summary['total_duration'] / max(processing_summary['successful_groups'], 1):.2f} seconds")
    print(f"\nOutput directory: {output_dir}")
    print(f"Long audio files: {output_path}")
    print(f"Ground truth file: {new_gt_file}")
    print(f"Processing summary: {summary_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Create long audio dataset for testing")
    parser.add_argument("--input_dir", required=True, 
                       help="Input directory containing WAV files")
    parser.add_argument("--ground_truth", required=True,
                       help="Path to ground truth CSV file")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for long audio files")
    parser.add_argument("--group_size", type=int, default=3,
                       help="Number of files to concatenate per group (default: 3)")
    
    args = parser.parse_args()
    
    # Import numpy here to avoid import issues
    global np
    import numpy as np
    
    print("=== Long Audio Dataset Creator ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Ground truth file: {args.ground_truth}")
    print(f"Output directory: {args.output_dir}")
    print(f"Group size: {args.group_size}")
    print()
    
    success = create_long_audio_dataset(
        args.input_dir,
        args.ground_truth,
        args.output_dir,
        args.group_size
    )
    
    if success:
        print("\n✅ Long audio dataset creation completed successfully!")
    else:
        print("\n❌ Long audio dataset creation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 