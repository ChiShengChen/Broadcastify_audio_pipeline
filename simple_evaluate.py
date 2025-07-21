#!/usr/bin/env python3
"""
Simple ASR evaluation script that doesn't depend on pandas
"""

import os
import csv
import glob
import argparse
from collections import defaultdict

def load_ground_truth(filepath):
    """Load ground truth CSV into a dictionary."""
    ground_truth = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Filename') and row.get('transcript'):
                    ground_truth[row['Filename']] = row['transcript']
        print(f"Loaded {len(ground_truth)} ground truth entries")
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return {}

def simple_wer(reference, hypothesis):
    """Calculate simple Word Error Rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Simple Levenshtein distance for words
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    errors = dp[m][n]
    total_words = len(ref_words)
    
    return errors / total_words if total_words > 0 else 1.0

def main():
    parser = argparse.ArgumentParser(description="Simple ASR evaluation")
    parser.add_argument("--transcript_dir", required=True, help="Directory with transcript files")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    
    args = parser.parse_args()
    
    # Load ground truth
    ground_truth = load_ground_truth(args.ground_truth)
    if not ground_truth:
        print("Failed to load ground truth")
        return
    
    # Find transcript files
    transcript_files = glob.glob(os.path.join(args.transcript_dir, "*.txt"))
    print(f"Found {len(transcript_files)} transcript files")
    
    # Group by model
    model_results = defaultdict(list)
    
    for transcript_file in transcript_files:
        basename = os.path.basename(transcript_file)
        
        # Parse model name and original filename
        if '_' in basename:
            parts = basename.split('_', 1)
            model_name = parts[0]
            original_name = parts[1].replace('.txt', '.wav')
            
            # Read transcript
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    hypothesis = f.read().strip()
                
                # Find matching ground truth
                if original_name in ground_truth:
                    reference = ground_truth[original_name]
                    wer = simple_wer(reference, hypothesis)
                    model_results[model_name].append({
                        'file': original_name,
                        'reference': reference,
                        'hypothesis': hypothesis,
                        'wer': wer
                    })
                    print(f"Processed {model_name}: {original_name} (WER: {wer:.3f})")
                else:
                    print(f"No ground truth for {original_name}")
                    
            except Exception as e:
                print(f"Error processing {transcript_file}: {e}")
    
    # Calculate average WER for each model
    results = []
    for model_name, files in model_results.items():
        if files:
            avg_wer = sum(f['wer'] for f in files) / len(files)
            results.append({
                'Model': model_name,
                'Files_Processed': len(files),
                'Average_WER': avg_wer,
                'Min_WER': min(f['wer'] for f in files),
                'Max_WER': max(f['wer'] for f in files)
            })
            print(f"\n{model_name}:")
            print(f"  Files: {len(files)}")
            print(f"  Avg WER: {avg_wer:.3f}")
            print(f"  Min WER: {min(f['wer'] for f in files):.3f}")
            print(f"  Max WER: {max(f['wer'] for f in files):.3f}")
    
    # Save results
    if results:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Model', 'Files_Processed', 'Average_WER', 'Min_WER', 'Max_WER'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")
    else:
        print("No results to save")

if __name__ == "__main__":
    main() 