#!/usr/bin/env python3

import os
import pandas as pd
import jiwer
import glob
from collections import defaultdict

# A standard transformation for both reference and hypothesis strings
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
])

def load_ground_truth(filepath):
    """Loads the ground truth CSV into a dictionary for easy lookup."""
    try:
        df = pd.read_csv(filepath)
        if 'Filename' not in df.columns or 'transcript' not in df.columns:
            print(f"Error: Ground truth file {filepath} must contain 'Filename' and 'transcript' columns.")
            return None
        df.dropna(subset=['Filename', 'transcript'], inplace=True)
        return pd.Series(df.transcript.values, index=df.Filename).to_dict()
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {filepath}")
        return None

def parse_filename(filepath, known_model_prefixes):
    """Parse filename to extract model name and ground truth key."""
    basename = os.path.basename(filepath)
    for prefix in known_model_prefixes:
        if basename.startswith(prefix + '_'):
            model_name = prefix
            original_file_part = basename[len(prefix) + 1:]
            original_file_base = os.path.splitext(original_file_part)[0]
            gt_key = original_file_base + '.wav'
            return model_name, gt_key
    return None, None

def test_total_files_matched_logic():
    """Test the Total_Files_Matched counting logic."""
    
    # Configuration
    transcript_dir = "pipeline_results_20250726_050348/merged_transcripts"
    ground_truth_file = "long_audio_test_dataset/long_audio_ground_truth.csv"
    model_prefixes = ['large-v3', 'wav2vec-xls-r', 'parakeet-tdt-0.6b-v2', 'canary-1b']
    
    ground_truth_map = load_ground_truth(ground_truth_file)
    if ground_truth_map is None:
        return
    
    # Find all transcript files
    path = os.path.join(transcript_dir, '**', '*.txt')
    all_txt_files = glob.glob(path, recursive=True)
    
    # Simulate the exact logic from evaluate_asr.py
    model_data = defaultdict(lambda: {'refs': [], 'hyps': []})
    matched_files_count = 0  # Global counter
    unmatched_files = []
    
    print("=== Testing Total_Files_Matched Logic ===")
    print(f"Total files found: {len(all_txt_files)}")
    print(f"Expected ground truth files: {len(ground_truth_map)}")
    print()
    
    # Process each file (same logic as evaluate_asr.py)
    for txt_file in all_txt_files:
        model_name, gt_key = parse_filename(txt_file, model_prefixes)
        
        if model_name and gt_key in ground_truth_map:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    hypothesis_text = transformation(f.read())
                
                reference_text = transformation(ground_truth_map[gt_key])
                
                if isinstance(reference_text, str) and hypothesis_text:
                    model_data[model_name]['refs'].append(reference_text)
                    model_data[model_name]['hyps'].append(hypothesis_text)
                    matched_files_count += 1  # Global counter increment
                    print(f"✓ Matched: {os.path.basename(txt_file)} -> {model_name}")
                else:
                    print(f"✗ Empty after transformation: {os.path.basename(txt_file)}")
                    unmatched_files.append(os.path.basename(txt_file))
            except Exception as e:
                print(f"✗ Error reading file: {os.path.basename(txt_file)} - {e}")
                unmatched_files.append(os.path.basename(txt_file))
        else:
            if not model_name:
                print(f"✗ Could not parse model name: {os.path.basename(txt_file)}")
            if gt_key not in ground_truth_map:
                print(f"✗ GT key not found: {gt_key} for {os.path.basename(txt_file)}")
            unmatched_files.append(os.path.basename(txt_file))
    
    print(f"\n=== Results ===")
    print(f"Global matched_files_count: {matched_files_count}")
    print(f"Unmatched files: {len(unmatched_files)}")
    
    # Check each model's count
    print(f"\nPer-model file counts:")
    total_model_files = 0
    for model_name, data in sorted(model_data.items()):
        model_count = len(data['refs'])
        total_model_files += model_count
        print(f"  {model_name}: {model_count} files")
    
    print(f"\n=== Validation ===")
    print(f"Global counter: {matched_files_count}")
    print(f"Sum of model counters: {total_model_files}")
    
    if matched_files_count == total_model_files:
        print("✓ Counters are consistent!")
    else:
        print("✗ Counters are inconsistent!")
        print("This indicates a bug in the counting logic.")
    
    # Show what Total_Files_Matched would be for each model
    print(f"\n=== Total_Files_Matched Values ===")
    for model_name, data in sorted(model_data.items()):
        total_files_matched = len(data['refs'])
        print(f"  {model_name}: Total_Files_Matched = {total_files_matched}")
    
    return model_data, matched_files_count

if __name__ == '__main__':
    test_total_files_matched_logic() 