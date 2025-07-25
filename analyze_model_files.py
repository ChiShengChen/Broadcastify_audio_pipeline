#!/usr/bin/env python3

import os
import pandas as pd
import glob
import argparse
from collections import defaultdict
from datetime import datetime

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

def analyze_model_files(transcript_dir, ground_truth_file, output_file=None):
    """Analyze files processed by each model and generate detailed report."""
    
    model_prefixes = ['large-v3', 'wav2vec-xls-r', 'parakeet-tdt-0.6b-v2', 'canary-1b']
    
    # Load ground truth
    ground_truth_map = load_ground_truth(ground_truth_file)
    if ground_truth_map is None:
        return
    
    # Find all transcript files
    path = os.path.join(transcript_dir, '**', '*.txt')
    all_txt_files = glob.glob(path, recursive=True)
    
    # Analyze files by model
    model_analysis = defaultdict(lambda: {
        'files': [],
        'gt_keys': set(),
        'missing_gt': set(),
        'empty_files': [],
        'error_files': [],
        'successful_files': []
    })
    
    # Expected ground truth keys
    expected_gt_keys = set(ground_truth_map.keys())
    
    print("=== Model File Analysis ===")
    print(f"Ground truth files: {len(expected_gt_keys)}")
    print(f"Expected files: {sorted(expected_gt_keys)}")
    print(f"Total transcript files found: {len(all_txt_files)}")
    print()
    
    # Process each transcript file
    for txt_file in all_txt_files:
        model_name, gt_key = parse_filename(txt_file, model_prefixes)
        
        if not model_name:
            print(f"Warning: Could not parse model name from {os.path.basename(txt_file)}")
            continue
            
        model_analysis[model_name]['files'].append(txt_file)
        
        if gt_key in expected_gt_keys:
            model_analysis[model_name]['gt_keys'].add(gt_key)
            
            # Check file content
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    model_analysis[model_name]['empty_files'].append(os.path.basename(txt_file))
                else:
                    model_analysis[model_name]['successful_files'].append(os.path.basename(txt_file))
            except Exception as e:
                model_analysis[model_name]['error_files'].append(f"{os.path.basename(txt_file)} (error: {e})")
        else:
            if gt_key:
                model_analysis[model_name]['missing_gt'].add(gt_key)
    
    # Generate detailed report
    report_lines = []
    report_lines.append("Model File Processing Analysis")
    report_lines.append("=" * 50)
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Transcript Directory: {transcript_dir}")
    report_lines.append(f"Ground Truth File: {ground_truth_file}")
    report_lines.append("")
    
    # Summary table
    report_lines.append("Summary Table:")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Model':<20} {'Total':<8} {'Success':<8} {'Empty':<8} {'Errors':<8} {'Missing GT':<10}")
    report_lines.append("-" * 80)
    
    for model in sorted(model_analysis.keys()):
        total = len(model_analysis[model]['files'])
        successful = len(model_analysis[model]['successful_files'])
        empty = len(model_analysis[model]['empty_files'])
        errors = len(model_analysis[model]['error_files'])
        missing_gt = len(expected_gt_keys - model_analysis[model]['gt_keys'])
        
        report_lines.append(f"{model:<20} {total:<8} {successful:<8} {empty:<8} {errors:<8} {missing_gt:<10}")
    
    report_lines.append("")
    
    # Detailed analysis per model
    for model in sorted(model_analysis.keys()):
        report_lines.append(f"=== {model} ===")
        report_lines.append(f"Total files found: {len(model_analysis[model]['files'])}")
        report_lines.append(f"Successful files: {len(model_analysis[model]['successful_files'])}")
        report_lines.append(f"Empty files: {len(model_analysis[model]['empty_files'])}")
        report_lines.append(f"Error files: {len(model_analysis[model]['error_files'])}")
        report_lines.append(f"Missing ground truth: {len(expected_gt_keys - model_analysis[model]['gt_keys'])}")
        
        # List successful files
        if model_analysis[model]['successful_files']:
            report_lines.append("Successful files:")
            for file in sorted(model_analysis[model]['successful_files']):
                report_lines.append(f"  ✓ {file}")
        
        # List empty files
        if model_analysis[model]['empty_files']:
            report_lines.append("Empty files:")
            for file in sorted(model_analysis[model]['empty_files']):
                report_lines.append(f"  ✗ {file}")
        
        # List error files
        if model_analysis[model]['error_files']:
            report_lines.append("Error files:")
            for file in sorted(model_analysis[model]['error_files']):
                report_lines.append(f"  ✗ {file}")
        
        # List missing ground truth
        missing_gt = expected_gt_keys - model_analysis[model]['gt_keys']
        if missing_gt:
            report_lines.append("Missing ground truth files:")
            for gt_key in sorted(missing_gt):
                report_lines.append(f"  ✗ {gt_key}")
        
        report_lines.append("")
    
    # Comparison analysis
    report_lines.append("=== Model Comparison ===")
    all_models = set(model_analysis.keys())
    
    # Find models with different file counts
    file_counts = {model: len(model_analysis[model]['successful_files']) for model in all_models}
    max_files = max(file_counts.values()) if file_counts else 0
    
    report_lines.append(f"Expected files per model: {len(expected_gt_keys)}")
    report_lines.append(f"Maximum successful files: {max_files}")
    report_lines.append("")
    
    for model in sorted(all_models):
        count = file_counts[model]
        if count < max_files:
            missing = max_files - count
            report_lines.append(f"{model}: Missing {missing} files")
        elif count == max_files:
            report_lines.append(f"{model}: ✓ All files processed")
        else:
            report_lines.append(f"{model}: ⚠ Unexpected - {count} files (more than expected)")
    
    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save report if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\nDetailed report saved to: {output_file}")
    
    return model_analysis

def main():
    parser = argparse.ArgumentParser(description="Analyze files processed by each ASR model")
    parser.add_argument("--transcript_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--ground_truth_file", required=True, help="Ground truth CSV file")
    parser.add_argument("--output_file", help="Output file for detailed report")
    
    args = parser.parse_args()
    
    analyze_model_files(args.transcript_dir, args.ground_truth_file, args.output_file)

if __name__ == '__main__':
    main() 