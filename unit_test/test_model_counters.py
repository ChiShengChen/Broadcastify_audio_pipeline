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

def test_model_counters():
    """Test how counters behave for each model."""
    
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
    
    # Sort files to ensure consistent order for testing
    all_txt_files.sort()
    
    # Simulate the exact logic from evaluate_asr.py
    model_data = defaultdict(lambda: {'refs': [], 'hyps': []})
    matched_files_count = 0  # Global counter
    unmatched_files = []
    
    print("=== Testing Model Counter Behavior ===")
    print(f"Total files to process: {len(all_txt_files)}")
    print()
    
    # Track counters for each model as we process files
    model_counters = defaultdict(int)
    current_model = None
    
    # Process each file and track counter changes
    for i, txt_file in enumerate(all_txt_files):
        model_name, gt_key = parse_filename(txt_file, model_prefixes)
        
        print(f"Processing file {i+1}/{len(all_txt_files)}: {os.path.basename(txt_file)}")
        print(f"  Model: {model_name}")
        
        # Check if we're switching to a new model
        if current_model != model_name and current_model is not None:
            print(f"  🔄 SWITCHING MODEL: {current_model} -> {model_name}")
            print(f"  📊 {current_model} final count: {model_counters[current_model]}")
        
        current_model = model_name
        
        if model_name and gt_key in ground_truth_map:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    hypothesis_text = transformation(f.read())
                
                reference_text = transformation(ground_truth_map[gt_key])
                
                if isinstance(reference_text, str) and hypothesis_text:
                    model_data[model_name]['refs'].append(reference_text)
                    model_data[model_name]['hyps'].append(hypothesis_text)
                    matched_files_count += 1  # Global counter increment
                    model_counters[model_name] += 1  # Model-specific counter
                    
                    print(f"  ✓ SUCCESS: {model_name} count = {model_counters[model_name]}")
                else:
                    print(f"  ✗ EMPTY: {model_name} count unchanged = {model_counters[model_name]}")
                    unmatched_files.append(os.path.basename(txt_file))
            except Exception as e:
                print(f"  ✗ ERROR: {model_name} count unchanged = {model_counters[model_name]} - {e}")
                unmatched_files.append(os.path.basename(txt_file))
        else:
            print(f"  ✗ UNMATCHED: count unchanged")
            unmatched_files.append(os.path.basename(txt_file))
        
        print()
    
    # Final summary
    print("=== Final Counter Summary ===")
    print(f"Global matched_files_count: {matched_files_count}")
    print()
    
    print("Per-model counters (from tracking):")
    total_from_tracking = 0
    for model in sorted(model_counters.keys()):
        count = model_counters[model]
        total_from_tracking += count
        print(f"  {model}: {count} files")
    
    print()
    print("Per-model counters (from model_data):")
    total_from_model_data = 0
    for model_name, data in sorted(model_data.items()):
        count = len(data['refs'])
        total_from_model_data += count
        print(f"  {model_name}: {count} files")
    
    print()
    print("=== Validation ===")
    print(f"Global counter: {matched_files_count}")
    print(f"Sum from tracking: {total_from_tracking}")
    print(f"Sum from model_data: {total_from_model_data}")
    
    if matched_files_count == total_from_tracking == total_from_model_data:
        print("✅ All counters are consistent!")
    else:
        print("❌ Counter inconsistency detected!")
    
    # Check if counters reset for each model
    print()
    print("=== Counter Reset Analysis ===")
    print("Each model maintains its own counter independently.")
    print("Counters do NOT reset when switching between models.")
    print("Each model accumulates its files throughout the entire process.")
    
    return model_data, model_counters

if __name__ == '__main__':
    test_model_counters() 