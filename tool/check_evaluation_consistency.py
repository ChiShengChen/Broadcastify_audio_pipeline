#!/usr/bin/env python3
"""
Check evaluation consistency and potential issues
"""

import os
import pandas as pd
import jiwer
import glob
from collections import defaultdict

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

def run_evaluation(transcript_dir, ground_truth_file, model_prefixes):
    """Run evaluation and return detailed results."""
    
    # A standard transformation for both reference and hypothesis strings
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip()
    ])
    
    ground_truth_map = load_ground_truth(ground_truth_file)
    if ground_truth_map is None:
        return None
    
    # Find all transcript files
    all_txt_files = []
    for directory in [transcript_dir]:
        path = os.path.join(directory, '**', '*.txt')
        found_files = glob.glob(path, recursive=True)
        all_txt_files.extend(found_files)
    
    # Process files
    model_data = defaultdict(lambda: {'refs': [], 'hyps': [], 'files': []})
    unmatched_files = []
    
    for txt_file in all_txt_files:
        model_name, gt_key = parse_filename(txt_file, model_prefixes)
        
        if model_name and gt_key in ground_truth_map:
            try:
                reference_text = transformation(ground_truth_map[gt_key])
                
                with open(txt_file, 'r', encoding='utf-8') as f:
                    hypothesis_text = transformation(f.read())
                
                if isinstance(reference_text, str) and hypothesis_text:
                    model_data[model_name]['refs'].append(reference_text)
                    model_data[model_name]['hyps'].append(hypothesis_text)
                    model_data[model_name]['files'].append(os.path.basename(txt_file))
                else:
                    print(f"Warning: Empty or invalid content in {txt_file}")
            except Exception as e:
                print(f"Error processing {txt_file}: {e}")
        else:
            unmatched_files.append(os.path.basename(txt_file))
    
    # Calculate metrics
    results = []
    for model_name, data in sorted(model_data.items()):
        if not data['refs']:
            continue
        
        try:
            output = jiwer.process_words(data['refs'], data['hyps'])
            
            result = {
                'Model': model_name,
                'WER': output.wer,
                'MER': output.mer,
                'WIL': output.wil,
                'Substitutions': output.substitutions,
                'Deletions': output.deletions,
                'Insertions': output.insertions,
                'Hits': output.hits,
                'Total_Words_in_Reference': output.hits + output.substitutions + output.deletions,
                'Total_Files_Matched': len(data['refs']),
                'Files': data['files']
            }
            results.append(result)
        except Exception as e:
            print(f"Error calculating metrics for {model_name}: {e}")
    
    return results, unmatched_files

def main():
    # Configuration
    transcript_dir = "pipeline_csside_results_20250726_070330/merged_transcripts"
    ground_truth_file = "long_audio_test_dataset/long_audio_ground_truth.csv"
    model_prefixes = ['large-v3', 'wav2vec-xls-r', 'parakeet-tdt-0.6b-v2', 'canary-1b']
    
    print("=== Evaluation Consistency Check ===")
    print()
    
    # Run evaluation
    results, unmatched_files = run_evaluation(transcript_dir, ground_truth_file, model_prefixes)
    
    if results is None:
        print("Failed to run evaluation")
        return
    
    # Display results
    print("=== Evaluation Results ===")
    for result in results:
        print(f"\n{result['Model']}:")
        print(f"  Total_Files_Matched: {result['Total_Files_Matched']}")
        print(f"  WER: {result['WER']:.4f}")
        print(f"  MER: {result['MER']:.4f}")
        print(f"  Total_Words_in_Reference: {result['Total_Words_in_Reference']}")
        print(f"  Files: {result['Files']}")
    
    # Compare with existing results
    existing_results_file = "pipeline_csside_results_20250726_070330/asr_evaluation_results.csv"
    if os.path.exists(existing_results_file):
        print("\n=== Comparison with Existing Results ===")
        existing_df = pd.read_csv(existing_results_file)
        
        for _, existing_row in existing_df.iterrows():
            model = existing_row['Model']
            existing_files = existing_row['Total_Files_Matched']
            existing_wer = existing_row['WER']
            
            # Find corresponding result
            current_result = next((r for r in results if r['Model'] == model), None)
            
            if current_result:
                current_files = current_result['Total_Files_Matched']
                current_wer = current_result['WER']
                
                print(f"\n{model}:")
                print(f"  Existing Files: {existing_files}, Current Files: {current_files}")
                print(f"  Existing WER: {existing_wer:.4f}, Current WER: {current_wer:.4f}")
                
                if existing_files != current_files:
                    print(f"  ⚠️  FILE COUNT MISMATCH!")
                if abs(existing_wer - current_wer) > 0.001:
                    print(f"  ⚠️  WER MISMATCH!")
            else:
                print(f"\n{model}: Not found in current results")
    
    # Check for potential issues
    print("\n=== Potential Issues Analysis ===")
    
    # Check if all models have the same number of files
    file_counts = [r['Total_Files_Matched'] for r in results]
    if len(set(file_counts)) > 1:
        print("⚠️  Different models have different file counts!")
        for result in results:
            print(f"  {result['Model']}: {result['Total_Files_Matched']} files")
    else:
        print("✓ All models have the same number of files")
    
    # Check for unmatched files
    if unmatched_files:
        print(f"\n⚠️  Found {len(unmatched_files)} unmatched files:")
        for file in unmatched_files[:10]:  # Show first 10
            print(f"  {file}")
        if len(unmatched_files) > 10:
            print(f"  ... and {len(unmatched_files) - 10} more")
    else:
        print("\n✓ No unmatched files found")
    
    # Check for empty or problematic files
    print("\n=== File Content Analysis ===")
    for result in results:
        model = result['Model']
        files = result['Files']
        
        print(f"\n{model} files:")
        for file in files:
            file_path = os.path.join(transcript_dir, model, file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if not content:
                        print(f"  ⚠️  {file}: EMPTY")
                    elif len(content) < 10:
                        print(f"  ⚠️  {file}: VERY SHORT ({len(content)} chars)")
                    else:
                        print(f"  ✓ {file}: {len(content)} chars, {len(content.split())} words")
                except Exception as e:
                    print(f"  ❌ {file}: ERROR - {e}")
            else:
                print(f"  ❌ {file}: NOT FOUND")

if __name__ == '__main__':
    main() 