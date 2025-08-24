#!/usr/bin/env python3
"""
Evaluate ASR Selection results against ground truth.
This script evaluates the combined results from ASR Selection mode,
where LLM has already chosen the better ASR result for each file.
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
import re
from typing import Dict, List, Tuple
import numpy as np

def extract_base_filename(filename: str) -> str:
    """Extract base filename from ASR result filename."""
    # Remove model prefix (e.g., "canary-1b_", "large-v3_")
    if filename.startswith("canary-1b_"):
        base = filename[10:]  # Remove "canary-1b_"
    elif filename.startswith("large-v3_"):
        base = filename[9:]   # Remove "large-v3_"
    else:
        base = filename
    
    # Remove .txt extension
    if base.endswith(".txt"):
        base = base[:-4]
    
    return base

def load_ground_truth(ground_truth_file: str) -> Dict[str, str]:
    """Load ground truth data from CSV file."""
    print(f"Loading ground truth from: {ground_truth_file}")
    
    try:
        df = pd.read_csv(ground_truth_file)
        ground_truth = {}
        
        for _, row in df.iterrows():
            filename = row['Filename']
            transcript = row['transcript']
            
            # Remove .wav extension for matching
            if filename.endswith('.wav'):
                filename = filename[:-4]
            
            ground_truth[filename] = transcript
        
        print(f"Loaded {len(ground_truth)} ground truth entries")
        return ground_truth
        
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return {}

def load_transcripts(transcript_dir: str) -> Dict[str, str]:
    """Load transcript files from directory."""
    print(f"Loading transcripts from: {transcript_dir}")
    
    transcripts = {}
    transcript_path = Path(transcript_dir)
    
    if not transcript_path.exists():
        print(f"Error: Transcript directory does not exist: {transcript_dir}")
        return transcripts
    
    # Find all .txt files
    txt_files = list(transcript_path.glob("*.txt"))
    print(f"Found {len(txt_files)} transcript files")
    
    for file_path in txt_files:
        try:
            # Read transcript content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:  # Only include non-empty files
                # Extract base filename for matching
                base_filename = extract_base_filename(file_path.name)
                transcripts[base_filename] = content
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Successfully loaded {len(transcripts)} transcripts")
    return transcripts

def calculate_wer(reference: str, hypothesis: str) -> Tuple[float, int, int, int, int]:
    """
    Calculate Word Error Rate (WER) and related metrics.
    Returns: (WER, substitutions, deletions, insertions, hits)
    """
    # Simple word-level tokenization
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Dynamic programming for edit distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Backtrack to get edit operations
    substitutions = deletions = insertions = hits = 0
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            deletions += 1
            i -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1):
            insertions += 1
            j -= 1
        else:
            substitutions += 1
            i -= 1
            j -= 1
    
    total_errors = substitutions + deletions + insertions
    wer = total_errors / len(ref_words) if ref_words else 0
    
    return wer, substitutions, deletions, insertions, hits

def evaluate_asr_selection(transcript_dir: str, ground_truth_file: str, output_file: str):
    """Evaluate ASR Selection results against ground truth."""
    
    # Load data
    ground_truth = load_ground_truth(ground_truth_file)
    transcripts = load_transcripts(transcript_dir)
    
    if not ground_truth:
        print("Error: No ground truth data loaded")
        return
    
    if not transcripts:
        print("Error: No transcript data loaded")
        return
    
    # Match transcripts with ground truth
    matched_files = 0
    total_wer = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_hits = 0
    total_ref_words = 0
    
    results = []
    
    print("\nMatching transcripts with ground truth...")
    
    for base_filename, transcript in transcripts.items():
        if base_filename in ground_truth:
            matched_files += 1
            reference = ground_truth[base_filename]
            
            # Calculate WER
            wer, subs, dels, ins, hits = calculate_wer(reference, transcript)
            
            total_wer += wer
            total_substitutions += subs
            total_deletions += dels
            total_insertions += ins
            total_hits += hits
            total_ref_words += len(reference.split())
            
            results.append({
                'filename': base_filename,
                'wer': wer,
                'substitutions': subs,
                'deletions': dels,
                'insertions': ins,
                'hits': hits,
                'ref_words': len(reference.split()),
                'hyp_words': len(transcript.split())
            })
            
            if matched_files <= 5:  # Show first few matches
                print(f"  Matched: {base_filename}")
    
    if matched_files == 0:
        print("No files matched with ground truth!")
        print("Example transcript filenames:")
        for i, filename in enumerate(list(transcripts.keys())[:5]):
            print(f"  {filename}")
        print("Example ground truth filenames:")
        for i, filename in enumerate(list(ground_truth.keys())[:5]):
            print(f"  {filename}")
        return
    
    # Calculate overall metrics
    avg_wer = total_wer / matched_files if matched_files > 0 else 0
    avg_mer = (total_substitutions + total_deletions) / total_ref_words if total_ref_words > 0 else 0
    avg_wil = (total_substitutions + total_deletions + total_insertions) / total_ref_words if total_ref_words > 0 else 0
    
    print(f"\nEvaluation Results:")
    print(f"Files matched: {matched_files}")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Average MER: {avg_mer:.4f} ({avg_mer*100:.2f}%)")
    print(f"Average WIL: {avg_wil:.4f} ({avg_wil*100:.2f}%)")
    print(f"Total substitutions: {total_substitutions}")
    print(f"Total deletions: {total_deletions}")
    print(f"Total insertions: {total_insertions}")
    print(f"Total hits: {total_hits}")
    print(f"Total reference words: {total_ref_words}")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Save summary
    summary_file = output_file.replace('.csv', '_summary.csv')
    summary_data = {
        'Metric': ['WER', 'MER', 'WIL', 'Substitutions', 'Deletions', 'Insertions', 'Hits', 'Total_Reference_Words', 'Files_Matched'],
        'Value': [avg_wer, avg_mer, avg_wil, total_substitutions, total_deletions, total_insertions, total_hits, total_ref_words, matched_files]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")
    
    return avg_wer, avg_mer, avg_wil

def main():
    parser = argparse.ArgumentParser(description='Evaluate ASR Selection results against ground truth')
    parser.add_argument('--transcript_dir', required=True, help='Directory containing transcript files')
    parser.add_argument('--ground_truth_file', required=True, help='Path to ground truth CSV file')
    parser.add_argument('--output_file', required=True, help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    print("=== ASR Selection Evaluation ===")
    print(f"Transcript directory: {args.transcript_dir}")
    print(f"Ground truth file: {args.ground_truth_file}")
    print(f"Output file: {args.output_file}")
    print()
    
    evaluate_asr_selection(args.transcript_dir, args.ground_truth_file, args.output_file)

if __name__ == "__main__":
    main()
