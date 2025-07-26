#!/usr/bin/env python3
"""
Test script to compare evaluation results with and without ground truth preprocessing
"""

import os
import pandas as pd
import subprocess
import tempfile
import shutil
from datetime import datetime

def run_evaluation_with_ground_truth(ground_truth_file, output_dir, transcript_dir):
    """Run evaluation with specified ground truth file"""
    
    # Create temporary output directory
    temp_output_dir = os.path.join(output_dir, f"temp_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    output_file = os.path.join(temp_output_dir, "asr_evaluation_results.csv")
    
    # Run evaluation
    cmd = [
        "python3", "evaluate_asr.py",
        "--transcript_dirs", transcript_dir,
        "--ground_truth_file", ground_truth_file,
        "--output_file", output_file
    ]
    
    print(f"Running evaluation with: {ground_truth_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Load and return results
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            return df, temp_output_dir
        else:
            print(f"Error: Output file not created: {output_file}")
            return None, temp_output_dir
    else:
        print(f"Error running evaluation: {result.stderr}")
        return None, temp_output_dir

def compare_results(original_results, preprocessed_results):
    """Compare results from original and preprocessed ground truth"""
    
    print("\n=== Comparison Results ===")
    
    if original_results is None or preprocessed_results is None:
        print("Error: One or both evaluations failed")
        return
    
    # Merge results for comparison
    comparison = pd.merge(
        original_results, 
        preprocessed_results, 
        on='Model', 
        suffixes=('_original', '_preprocessed')
    )
    
    print("\nModel Comparison:")
    print("=" * 80)
    
    for _, row in comparison.iterrows():
        model = row['Model']
        print(f"\n{model}:")
        
        # File count comparison
        files_orig = row['Total_Files_Matched_original']
        files_prep = row['Total_Files_Matched_preprocessed']
        print(f"  Files Matched: {files_orig} -> {files_prep}")
        
        if files_orig != files_prep:
            print(f"  ⚠️  FILE COUNT DIFFERENCE!")
        
        # WER comparison
        wer_orig = row['WER_original']
        wer_prep = row['WER_preprocessed']
        wer_diff = wer_prep - wer_orig
        print(f"  WER: {wer_orig:.4f} -> {wer_prep:.4f} (diff: {wer_diff:+.4f})")
        
        if abs(wer_diff) > 0.01:
            if wer_diff < 0:
                print(f"  ✓ WER IMPROVED by {abs(wer_diff):.4f}")
            else:
                print(f"  ⚠️  WER WORSENED by {wer_diff:.4f}")
        
        # MER comparison
        mer_orig = row['MER_original']
        mer_prep = row['MER_preprocessed']
        mer_diff = mer_prep - mer_orig
        print(f"  MER: {mer_orig:.4f} -> {mer_prep:.4f} (diff: {mer_diff:+.4f})")
        
        # Word count comparison
        words_orig = row['Total_Words_in_Reference_original']
        words_prep = row['Total_Words_in_Reference_preprocessed']
        print(f"  Total Words: {words_orig} -> {words_prep}")
        
        if words_orig != words_prep:
            print(f"  ⚠️  WORD COUNT DIFFERENCE!")

def main():
    # Configuration
    original_ground_truth = "long_audio_test_dataset/long_audio_ground_truth.csv"
    transcript_dir = "pipeline_csside_results_20250726_070330/merged_transcripts"
    output_dir = "preprocessing_test_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Ground Truth Preprocessing Impact Test ===")
    print(f"Original ground truth: {original_ground_truth}")
    print(f"Transcript directory: {transcript_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Step 1: Create preprocessed ground truth
    print("Step 1: Creating preprocessed ground truth...")
    preprocessed_ground_truth = os.path.join(output_dir, "preprocessed_ground_truth.csv")
    
    cmd = [
        "python3", "preprocess_ground_truth.py",
        "--input_file", original_ground_truth,
        "--output_file", preprocessed_ground_truth
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error creating preprocessed ground truth: {result.stderr}")
        return
    
    print("✓ Preprocessed ground truth created successfully")
    print()
    
    # Step 2: Run evaluation with original ground truth
    print("Step 2: Running evaluation with original ground truth...")
    original_results, temp_dir_orig = run_evaluation_with_ground_truth(
        original_ground_truth, output_dir, transcript_dir
    )
    
    if original_results is None:
        print("❌ Original evaluation failed")
        return
    
    print("✓ Original evaluation completed")
    print()
    
    # Step 3: Run evaluation with preprocessed ground truth
    print("Step 3: Running evaluation with preprocessed ground truth...")
    preprocessed_results, temp_dir_prep = run_evaluation_with_ground_truth(
        preprocessed_ground_truth, output_dir, transcript_dir
    )
    
    if preprocessed_results is None:
        print("❌ Preprocessed evaluation failed")
        return
    
    print("✓ Preprocessed evaluation completed")
    print()
    
    # Step 4: Compare results
    print("Step 4: Comparing results...")
    compare_results(original_results, preprocessed_results)
    
    # Step 5: Save comparison report
    print("\nStep 5: Saving comparison report...")
    
    if original_results is not None and preprocessed_results is not None:
        # Merge for detailed comparison
        comparison = pd.merge(
            original_results, 
            preprocessed_results, 
            on='Model', 
            suffixes=('_original', '_preprocessed')
        )
        
        # Add difference columns
        comparison['WER_Difference'] = comparison['WER_preprocessed'] - comparison['WER_original']
        comparison['MER_Difference'] = comparison['MER_preprocessed'] - comparison['MER_original']
        comparison['Files_Difference'] = comparison['Total_Files_Matched_preprocessed'] - comparison['Total_Files_Matched_original']
        comparison['Words_Difference'] = comparison['Total_Words_in_Reference_preprocessed'] - comparison['Total_Words_in_Reference_original']
        
        # Save comparison
        comparison_file = os.path.join(output_dir, "preprocessing_comparison.csv")
        comparison.to_csv(comparison_file, index=False)
        print(f"✓ Comparison saved to: {comparison_file}")
        
        # Save individual results
        original_results.to_csv(os.path.join(output_dir, "original_evaluation.csv"), index=False)
        preprocessed_results.to_csv(os.path.join(output_dir, "preprocessed_evaluation.csv"), index=False)
        
        print(f"✓ Original results saved to: {os.path.join(output_dir, 'original_evaluation.csv')}")
        print(f"✓ Preprocessed results saved to: {os.path.join(output_dir, 'preprocessed_evaluation.csv')}")
    
    # Cleanup temporary directories
    if os.path.exists(temp_dir_orig):
        shutil.rmtree(temp_dir_orig)
    if os.path.exists(temp_dir_prep):
        shutil.rmtree(temp_dir_prep)
    
    print("\n=== Test Completed ===")
    print(f"All results saved in: {output_dir}")

if __name__ == '__main__':
    main() 