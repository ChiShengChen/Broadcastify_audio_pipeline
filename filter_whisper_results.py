#!/usr/bin/env python3
"""
Filter Whisper Results Script

This script filters out only Whisper (large-v3) ASR results from the pipeline output.
"""

import argparse
import os
import shutil
from pathlib import Path

def filter_whisper_files(input_dir: Path, output_dir: Path):
    """Filter only Whisper (large-v3) ASR results"""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all large-v3 files (Whisper results)
    whisper_files = []
    for file_path in input_dir.rglob("*.txt"):
        if "large-v3_" in file_path.name:
            whisper_files.append(file_path)
    
    print(f"Found {len(whisper_files)} Whisper (large-v3) files")
    
    # Copy Whisper files to output directory
    for file_path in whisper_files:
        # Create relative path structure
        relative_path = file_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(file_path, output_path)
        print(f"Copied: {relative_path}")
    
    print(f"Whisper files copied to: {output_dir}")
    return len(whisper_files)

def main():
    parser = argparse.ArgumentParser(description="Filter Whisper ASR Results")
    parser.add_argument("--input_dir", required=True, help="Input directory with ASR results")
    parser.add_argument("--output_dir", required=True, help="Output directory for Whisper results")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    # Filter Whisper files
    count = filter_whisper_files(input_dir, output_dir)
    
    if count > 0:
        print(f"\nSuccessfully filtered {count} Whisper files")
        print(f"Output directory: {output_dir}")
        return 0
    else:
        print("No Whisper files found")
        return 1

if __name__ == "__main__":
    exit(main()) 