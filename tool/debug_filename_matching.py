#!/usr/bin/env python3
"""
Debug script to check filename matching issues in ASR evaluation
"""

import os
import pandas as pd
import glob
from collections import defaultdict

def load_ground_truth(filepath):
    """Loads the ground truth CSV into a dictionary for easy lookup."""
    try:
        df = pd.read_csv(filepath)
        if 'Filename' not in df.columns or 'transcript' not in df.columns:
            print(f"Error: Ground truth file {filepath} must contain 'Filename' and 'transcript' columns.")
            return None
        # Handle potential NaN values in the transcript column
        df.dropna(subset=['Filename', 'transcript'], inplace=True)
        return pd.Series(df.transcript.values, index=df.Filename).to_dict()
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {filepath}")
        return None

def parse_filename(filepath, known_model_prefixes):
    """
    Parses a filepath like '.../large-v3_202412010133-841696-14744_call_2.txt'
    and returns the model name and the ground truth key (e.g., '..._call_2.wav').
    Returns: (model_name, ground_truth_key) or (None, None)
    """
    basename = os.path.basename(filepath)
    for prefix in known_model_prefixes:
        if basename.startswith(prefix + '_'):
            model_name = prefix
            # Extract the part after the prefix
            original_file_part = basename[len(prefix) + 1:]
            # The base of the file, without the .txt extension
            original_file_base = os.path.splitext(original_file_part)[0]
            # The ground truth key has a .wav extension
            gt_key = original_file_base + '.wav'
            return model_name, gt_key
    return None, None

def main():
    # Configuration
    transcript_dir = "pipeline_csside_results_20250726_070330/merged_transcripts"
    ground_truth_file = "long_audio_test_dataset/long_audio_ground_truth.csv"
    model_prefixes = ['large-v3', 'wav2vec-xls-r', 'parakeet-tdt-0.6b-v2', 'canary-1b']
    
    print("=== Debug Filename Matching ===")
    print(f"Transcript directory: {transcript_dir}")
    print(f"Ground truth file: {ground_truth_file}")
    print(f"Model prefixes: {model_prefixes}")
    print()
    
    # Load ground truth
    ground_truth_map = load_ground_truth(ground_truth_file)
    if ground_truth_map is None:
        return
    
    print("=== Ground Truth Files ===")
    for filename in sorted(ground_truth_map.keys()):
        print(f"  {filename}")
    print()
    
    # Find all transcript files
    all_txt_files = []
    for directory in [transcript_dir]:
        path = os.path.join(directory, '**', '*.txt')
        found_files = glob.glob(path, recursive=True)
        print(f"Found {len(found_files)} .txt files in {directory}")
        all_txt_files.extend(found_files)
    
    print()
    print("=== Transcript Files ===")
    for txt_file in sorted(all_txt_files):
        print(f"  {os.path.basename(txt_file)}")
    print()
    
    # Analyze matching
    print("=== Matching Analysis ===")
    model_data = defaultdict(lambda: {'matched': [], 'unmatched': []})
    
    for txt_file in all_txt_files:
        model_name, gt_key = parse_filename(txt_file, model_prefixes)
        
        if model_name and gt_key in ground_truth_map:
            model_data[model_name]['matched'].append((txt_file, gt_key))
        else:
            model_data[model_name]['unmatched'].append(txt_file)
    
    # Report results
    for model in model_prefixes:
        matched_count = len(model_data[model]['matched'])
        unmatched_count = len(model_data[model]['unmatched'])
        total_expected = len(ground_truth_map)
        
        print(f"\n{model}:")
        print(f"  Expected files: {total_expected}")
        print(f"  Matched files: {matched_count}")
        print(f"  Unmatched files: {unmatched_count}")
        print(f"  Missing files: {total_expected - matched_count}")
        
        if model_data[model]['matched']:
            print("  Matched files:")
            for txt_file, gt_key in model_data[model]['matched']:
                print(f"    {os.path.basename(txt_file)} -> {gt_key}")
        
        if model_data[model]['unmatched']:
            print("  Unmatched files:")
            for txt_file in model_data[model]['unmatched']:
                print(f"    {os.path.basename(txt_file)}")
    
    # Check for special characters in filenames
    print("\n=== Special Character Analysis ===")
    special_chars = ['%', '-', '&', '(', ')', '[', ']', '{', '}', '!', '@', '#', '$', '^', '*', '+', '=', '|', '\\', '/', '?', '<', '>', '~', '`']
    
    for filename in ground_truth_map.keys():
        found_chars = [char for char in special_chars if char in filename]
        if found_chars:
            print(f"  {filename}: contains {found_chars}")
    
    # Check for encoding issues
    print("\n=== Encoding Analysis ===")
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print("  Ground truth file: UTF-8 encoding OK")
    except UnicodeDecodeError as e:
        print(f"  Ground truth file: Encoding error - {e}")
    
    for txt_file in all_txt_files[:5]:  # Check first 5 files
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"  {os.path.basename(txt_file)}: UTF-8 encoding OK")
        except UnicodeDecodeError as e:
            print(f"  {os.path.basename(txt_file)}: Encoding error - {e}")

if __name__ == '__main__':
    main() 