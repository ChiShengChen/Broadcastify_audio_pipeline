#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Segmented Transcripts for WER Calculation
==============================================

This script merges segmented transcripts back into original file transcripts
for proper WER calculation. It handles transcripts from audio files that were
split during preprocessing.

Usage:
    python3 merge_segmented_transcripts.py --input_dir /path/to/transcripts --output_dir /path/to/merged --metadata_file /path/to/metadata.json
"""

import os
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import re

class SegmentedTranscriptMerger:
    """Merge segmented transcripts back to original files"""
    
    def __init__(self, metadata_file: str):
        """
        Initialize merger
        
        Args:
            metadata_file: Path to processing metadata JSON file
        """
        self.metadata_file = metadata_file
        self.metadata = self.load_metadata()
        
    def load_metadata(self) -> Dict:
        """Load processing metadata"""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata file {self.metadata_file}: {e}")
            return {}
    
    def find_original_file_mapping(self) -> Dict[str, List[Dict]]:
        """
        Create mapping from original files to their segments
        
        Returns:
            Dictionary mapping original filename to list of segment info
        """
        mapping = defaultdict(list)
        
        if not self.metadata or 'file_results' not in self.metadata:
            print("Warning: No file results found in metadata")
            return mapping
        
        for file_result in self.metadata['file_results']:
            if file_result['status'] == 'success' and 'segments' in file_result:
                original_file = os.path.basename(file_result['original_file'])
                original_name = os.path.splitext(original_file)[0]
                
                for segment in file_result['segments']:
                    mapping[original_name].append({
                        'segment_filename': segment['filename'],
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'duration': segment['duration'],
                        'path': segment['path']
                    })
                
                # Sort segments by start time
                mapping[original_name].sort(key=lambda x: x['start_time'])
        
        return mapping
    
    def find_model_transcripts(self, transcript_dir: str) -> Dict[str, List[str]]:
        """
        Find all transcript files grouped by model
        
        Args:
            transcript_dir: Directory containing transcript files
            
        Returns:
            Dictionary mapping model name to list of transcript files
        """
        model_transcripts = defaultdict(list)
        
        # Find all .txt files
        txt_files = glob.glob(os.path.join(transcript_dir, "**/*.txt"), recursive=True)
        
        for txt_file in txt_files:
            filename = os.path.basename(txt_file)
            
            # Extract model name from filename
            # Expected format: model_name_original_filename.txt or model_name_original_filename_segment_XXX.txt
            parts = filename.split('_')
            if len(parts) >= 2:
                model_name = parts[0]
                model_transcripts[model_name].append(txt_file)
        
        return model_transcripts
    
    def read_transcript_content(self, transcript_file: str) -> Optional[str]:
        """
        Read transcript content from file
        
        Args:
            transcript_file: Path to transcript file
            
        Returns:
            Transcript content or None if error
        """
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return content
        except Exception as e:
            print(f"Error reading transcript file {transcript_file}: {e}")
            return None
    
    def merge_segments_for_file(self, original_name: str, segments: List[Dict], 
                               model_transcripts: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Merge segments for a specific original file
        
        Args:
            original_name: Original file name (without extension)
            segments: List of segment information
            model_transcripts: Dictionary of model transcripts
            
        Returns:
            Dictionary mapping model name to merged transcript
        """
        merged_transcripts = {}
        
        for model_name, transcript_files in model_transcripts.items():
            # Find segments for this model and original file
            model_segments = []
            
            for transcript_file in transcript_files:
                filename = os.path.basename(transcript_file)
                
                # Check if this transcript corresponds to any segment of this original file
                for segment in segments:
                    segment_filename = segment['segment_filename']
                    
                    # Handle both single file and segmented cases
                    if len(segments) == 1:
                        # Single file case: model_name_original_name.txt
                        expected_filename = f"{model_name}_{original_name}.txt"
                    else:
                        # Segmented case: model_name_original_name_segment_XXX.txt
                        expected_filename = f"{model_name}_{original_name}_{segment_filename.replace('.wav', '.txt')}"
                    
                    if filename == expected_filename:
                        content = self.read_transcript_content(transcript_file)
                        if content:
                            model_segments.append({
                                'content': content,
                                'start_time': segment['start_time'],
                                'end_time': segment['end_time']
                            })
                        break
            
            # Sort segments by start time and merge
            model_segments.sort(key=lambda x: x['start_time'])
            
            if model_segments:
                # Merge transcript contents
                merged_content = ' '.join(segment['content'] for segment in model_segments)
                merged_transcripts[model_name] = merged_content
            else:
                print(f"Warning: No segments found for {original_name} with model {model_name}")
        
        return merged_transcripts
    
    def merge_all_transcripts(self, transcript_dir: str, output_dir: str) -> Dict:
        """
        Merge all segmented transcripts
        
        Args:
            transcript_dir: Directory containing transcript files
            output_dir: Output directory for merged transcripts
            
        Returns:
            Processing results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get original file mapping
        original_mapping = self.find_original_file_mapping()
        
        # Find model transcripts
        model_transcripts = self.find_model_transcripts(transcript_dir)
        
        print(f"Found {len(original_mapping)} original files to merge")
        print(f"Found {len(model_transcripts)} models: {list(model_transcripts.keys())}")
        
        results = {
            'total_original_files': len(original_mapping),
            'models_processed': list(model_transcripts.keys()),
            'merged_files': 0,
            'error_files': 0,
            'merged_transcripts': {}
        }
        
        # Process each original file
        for original_name, segments in original_mapping.items():
            print(f"Processing {original_name} ({len(segments)} segments)")
            
            try:
                # Merge segments for this file
                merged_transcripts = self.merge_segments_for_file(
                    original_name, segments, model_transcripts
                )
                
                # Save merged transcripts
                for model_name, merged_content in merged_transcripts.items():
                    output_filename = f"{model_name}_{original_name}.txt"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(merged_content)
                    
                    print(f"  ✓ Saved {model_name}: {output_filename}")
                
                results['merged_transcripts'][original_name] = merged_transcripts
                results['merged_files'] += 1
                
            except Exception as e:
                print(f"  ✗ Error processing {original_name}: {e}")
                results['error_files'] += 1
        
        return results
    
    def generate_summary(self, results: Dict) -> str:
        """Generate processing summary"""
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("SEGMENTED TRANSCRIPT MERGING SUMMARY")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Total original files: {results['total_original_files']}")
        summary_lines.append(f"Successfully merged: {results['merged_files']}")
        summary_lines.append(f"Errors: {results['error_files']}")
        summary_lines.append(f"Models processed: {', '.join(results['models_processed'])}")
        summary_lines.append("")
        
        # Show details for each original file
        for original_name, merged_transcripts in results['merged_transcripts'].items():
            summary_lines.append(f"📁 {original_name}:")
            for model_name, content in merged_transcripts.items():
                word_count = len(content.split())
                summary_lines.append(f"  - {model_name}: {word_count} words")
            summary_lines.append("")
        
        return "\n".join(summary_lines)

def main():
    parser = argparse.ArgumentParser(description="Merge Segmented Transcripts for WER Calculation")
    parser.add_argument("--input_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged transcripts")
    parser.add_argument("--metadata_file", required=True, help="Processing metadata JSON file")
    
    args = parser.parse_args()
    
    # Initialize merger
    merger = SegmentedTranscriptMerger(args.metadata_file)
    
    # Merge transcripts
    print(f"Merging segmented transcripts...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metadata file: {args.metadata_file}")
    print()
    
    results = merger.merge_all_transcripts(args.input_dir, args.output_dir)
    
    # Print summary
    summary = merger.generate_summary(results)
    print(summary)
    
    # Save summary to file
    summary_file = os.path.join(args.output_dir, "merging_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_file}")
    
    if results['error_files'] == 0:
        print("\n✅ Merging completed successfully!")
    else:
        print(f"\n⚠️  Merging completed with {results['error_files']} errors")

if __name__ == "__main__":
    main() 