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
    transcript_dir = "pipeline_results_20250726_050348/merged_transcripts"
    ground_truth_file = "long_audio_test_dataset/long_audio_ground_truth.csv"
    model_prefixes = ['large-v3', 'wav2vec-xls-r', 'parakeet-tdt-0.6b-v2', 'canary-1b']
    
    ground_truth_map = load_ground_truth(ground_truth_file)
    if ground_truth_map is None:
        return
    
    print("Ground truth files:")
    for key in sorted(ground_truth_map.keys()):
        print(f"  - {key}")
    print()
    
    # Find all transcript files
    path = os.path.join(transcript_dir, '**', '*.txt')
    all_txt_files = glob.glob(path, recursive=True)
    print(f"Found {len(all_txt_files)} transcript files")
    
    # Analyze each file
    model_data = defaultdict(lambda: {'refs': [], 'hyps': [], 'files': []})
    unmatched_files = []
    
    for txt_file in all_txt_files:
        model_name, gt_key = parse_filename(txt_file, model_prefixes)
        
        print(f"\nProcessing: {os.path.basename(txt_file)}")
        print(f"  Model: {model_name}")
        print(f"  GT Key: {gt_key}")
        
        if model_name and gt_key in ground_truth_map:
            print(f"  ✓ Found in ground truth")
            
            # Read hypothesis text
            with open(txt_file, 'r', encoding='utf-8') as f:
                hypothesis_text = f.read().strip()
            
            print(f"  Original hypothesis length: {len(hypothesis_text)}")
            
            # Apply transformation
            transformed_hypothesis = transformation(hypothesis_text)
            print(f"  Transformed hypothesis length: {len(transformed_hypothesis)}")
            print(f"  Transformed hypothesis: '{transformed_hypothesis[:100]}...'")
            
            # Check if transformation resulted in empty string
            if transformed_hypothesis:
                print(f"  ✓ Hypothesis has content after transformation")
                model_data[model_name]['refs'].append(transformation(ground_truth_map[gt_key]))
                model_data[model_name]['hyps'].append(transformed_hypothesis)
                model_data[model_name]['files'].append(os.path.basename(txt_file))
            else:
                print(f"  ✗ Hypothesis is empty after transformation!")
                unmatched_files.append(os.path.basename(txt_file))
        else:
            if not model_name:
                print(f"  ✗ Could not parse model name")
            if gt_key not in ground_truth_map:
                print(f"  ✗ GT key '{gt_key}' not found in ground truth")
            unmatched_files.append(os.path.basename(txt_file))
    
    print(f"\n=== Summary ===")
    for model_name, data in sorted(model_data.items()):
        print(f"{model_name}: {len(data['files'])} files")
        for file in data['files']:
            print(f"  - {file}")
    
    print(f"\nUnmatched files: {len(unmatched_files)}")
    for file in unmatched_files[:5]:
        print(f"  - {file}")

if __name__ == '__main__':
    main() 