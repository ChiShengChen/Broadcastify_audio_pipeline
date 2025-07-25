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
    
    print("=== Analysis of Evaluation Results ===")
    print(f"Ground truth files: {len(ground_truth_map)}")
    print(f"Ground truth keys: {sorted(ground_truth_map.keys())}")
    print()
    
    # Find all transcript files
    path = os.path.join(transcript_dir, '**', '*.txt')
    all_txt_files = glob.glob(path, recursive=True)
    print(f"Total transcript files found: {len(all_txt_files)}")
    
    # Group files by model
    model_files = defaultdict(list)
    for txt_file in all_txt_files:
        model_name, gt_key = parse_filename(txt_file, model_prefixes)
        if model_name:
            model_files[model_name].append((txt_file, gt_key))
    
    print(f"\nFiles per model:")
    for model in sorted(model_files.keys()):
        print(f"  {model}: {len(model_files[model])} files")
    
    # Check for missing files per model
    print(f"\n=== Missing Files Analysis ===")
    for model in sorted(model_files.keys()):
        print(f"\n{model}:")
        model_gt_keys = set(gt_key for _, gt_key in model_files[model])
        missing_keys = set(ground_truth_map.keys()) - model_gt_keys
        if missing_keys:
            print(f"  Missing: {missing_keys}")
        else:
            print(f"  ✓ All ground truth files present")
    
    # Check for empty or problematic files
    print(f"\n=== Content Analysis ===")
    for model in sorted(model_files.keys()):
        print(f"\n{model}:")
        empty_files = []
        for txt_file, gt_key in model_files[model]:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        empty_files.append(os.path.basename(txt_file))
                    else:
                        transformed = transformation(content)
                        if not transformed:
                            empty_files.append(f"{os.path.basename(txt_file)} (empty after transformation)")
            except Exception as e:
                empty_files.append(f"{os.path.basename(txt_file)} (error: {e})")
        
        if empty_files:
            print(f"  Empty/problematic files: {empty_files}")
        else:
            print(f"  ✓ All files have content")
    
    # Simulate the exact evaluation logic
    print(f"\n=== Simulated Evaluation Results ===")
    model_data = defaultdict(lambda: {'refs': [], 'hyps': [], 'files': []})
    
    for txt_file in all_txt_files:
        model_name, gt_key = parse_filename(txt_file, model_prefixes)
        
        if model_name and gt_key in ground_truth_map:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    hypothesis_text = f.read().strip()
                
                transformed_hypothesis = transformation(hypothesis_text)
                
                if isinstance(ground_truth_map[gt_key], str) and transformed_hypothesis:
                    model_data[model_name]['refs'].append(transformation(ground_truth_map[gt_key]))
                    model_data[model_name]['hyps'].append(transformed_hypothesis)
                    model_data[model_name]['files'].append(os.path.basename(txt_file))
            except Exception as e:
                print(f"Error processing {txt_file}: {e}")
    
    print(f"\nFinal matched files per model:")
    for model_name, data in sorted(model_data.items()):
        print(f"  {model_name}: {len(data['files'])} files")
        if len(data['files']) != 9:
            print(f"    Files: {sorted(data['files'])}")

if __name__ == '__main__':
    main() 