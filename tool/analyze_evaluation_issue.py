#!/usr/bin/env python3
"""
Detailed analysis of ASR evaluation issues
"""

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

def analyze_transcript_content(txt_file):
    """Analyze the content of a transcript file"""
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            return {
                'empty': True,
                'length': 0,
                'words': 0,
                'special_chars': [],
                'has_content': False
            }
        
        # Check for special characters that might cause issues
        special_chars = ['%', '-', '&', '(', ')', '[', ']', '{', '}', '!', '@', '#', '$', '^', '*', '+', '=', '|', '\\', '/', '?', '<', '>', '~', '`', '"', "'"]
        found_chars = [char for char in special_chars if char in content]
        
        words = len(content.split())
        
        return {
            'empty': False,
            'length': len(content),
            'words': words,
            'special_chars': found_chars,
            'has_content': True,
            'preview': content[:100] + '...' if len(content) > 100 else content
        }
    except Exception as e:
        return {
            'error': str(e),
            'has_content': False
        }

def main():
    # Configuration
    transcript_dir = "pipeline_csside_results_20250726_070330/merged_transcripts"
    ground_truth_file = "long_audio_test_dataset/long_audio_ground_truth.csv"
    model_prefixes = ['large-v3', 'wav2vec-xls-r', 'parakeet-tdt-0.6b-v2', 'canary-1b']
    
    print("=== Detailed ASR Evaluation Analysis ===")
    print()
    
    # Load ground truth
    ground_truth_map = load_ground_truth(ground_truth_file)
    if ground_truth_map is None:
        return
    
    # Find all transcript files
    all_txt_files = []
    for directory in [transcript_dir]:
        path = os.path.join(directory, '**', '*.txt')
        found_files = glob.glob(path, recursive=True)
        all_txt_files.extend(found_files)
    
    # Analyze each model
    model_analysis = defaultdict(lambda: {
        'files': [],
        'matched': [],
        'unmatched': [],
        'empty_files': [],
        'error_files': [],
        'content_analysis': []
    })
    
    for txt_file in all_txt_files:
        model_name, gt_key = parse_filename(txt_file, model_prefixes)
        
        if model_name:
            model_analysis[model_name]['files'].append(txt_file)
            
            # Analyze content
            content_analysis = analyze_transcript_content(txt_file)
            model_analysis[model_name]['content_analysis'].append((txt_file, content_analysis))
            
            if 'error' in content_analysis:
                model_analysis[model_name]['error_files'].append(txt_file)
            elif content_analysis['empty']:
                model_analysis[model_name]['empty_files'].append(txt_file)
            
            if gt_key in ground_truth_map:
                model_analysis[model_name]['matched'].append((txt_file, gt_key))
            else:
                model_analysis[model_name]['unmatched'].append(txt_file)
    
    # Report detailed analysis
    for model in model_prefixes:
        analysis = model_analysis[model]
        
        print(f"=== {model} Analysis ===")
        print(f"Total files: {len(analysis['files'])}")
        print(f"Matched with ground truth: {len(analysis['matched'])}")
        print(f"Unmatched: {len(analysis['unmatched'])}")
        print(f"Empty files: {len(analysis['empty_files'])}")
        print(f"Error files: {len(analysis['error_files'])}")
        
        # Check if all matched files have content
        valid_matches = 0
        for txt_file, gt_key in analysis['matched']:
            content_info = next((info for file, info in analysis['content_analysis'] if file == txt_file), None)
            if content_info and content_info.get('has_content', False) and not content_info.get('empty', True):
                valid_matches += 1
            else:
                print(f"  Warning: {os.path.basename(txt_file)} has no valid content")
        
        print(f"Valid matches with content: {valid_matches}")
        print()
        
        # Show content analysis for first few files
        print("Content analysis (first 3 files):")
        for i, (txt_file, content_info) in enumerate(analysis['content_analysis'][:3]):
            basename = os.path.basename(txt_file)
            if 'error' in content_info:
                print(f"  {basename}: ERROR - {content_info['error']}")
            elif content_info['empty']:
                print(f"  {basename}: EMPTY")
            else:
                print(f"  {basename}: {content_info['words']} words, {content_info['length']} chars")
                if content_info['special_chars']:
                    print(f"    Special chars: {content_info['special_chars']}")
                print(f"    Preview: {content_info['preview']}")
        print()
    
    # Check for potential issues in ground truth
    print("=== Ground Truth Analysis ===")
    for filename, transcript in ground_truth_map.items():
        if not isinstance(transcript, str) or not transcript.strip():
            print(f"Warning: {filename} has empty or invalid transcript")
            continue
        
        # Check for special characters in ground truth
        special_chars = ['%', '-', '&', '(', ')', '[', ']', '{', '}', '!', '@', '#', '$', '^', '*', '+', '=', '|', '\\', '/', '?', '<', '>', '~', '`', '"', "'"]
        found_chars = [char for char in special_chars if char in transcript]
        
        if found_chars:
            print(f"  {filename}: contains special chars {found_chars}")
            print(f"    Preview: {transcript[:100]}...")
    
    # Simulate the evaluation process
    print("\n=== Simulated Evaluation Process ===")
    for model in model_prefixes:
        analysis = model_analysis[model]
        
        # Collect valid reference-hypothesis pairs
        refs = []
        hyps = []
        valid_count = 0
        
        for txt_file, gt_key in analysis['matched']:
            content_info = next((info for file, info in analysis['content_analysis'] if file == txt_file), None)
            
            if content_info and content_info.get('has_content', False) and not content_info.get('empty', True):
                try:
                    reference_text = transformation(ground_truth_map[gt_key])
                    
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        hypothesis_text = transformation(f.read())
                    
                    if isinstance(reference_text, str) and hypothesis_text:
                        refs.append(reference_text)
                        hyps.append(hypothesis_text)
                        valid_count += 1
                    else:
                        print(f"  {model}: {os.path.basename(txt_file)} - transformation failed")
                except Exception as e:
                    print(f"  {model}: {os.path.basename(txt_file)} - error: {e}")
            else:
                print(f"  {model}: {os.path.basename(txt_file)} - no valid content")
        
        print(f"{model}: {valid_count} valid pairs for evaluation")
        
        if valid_count > 0:
            try:
                output = jiwer.process_words(refs, hyps)
                print(f"  WER: {output.wer:.4f}")
                print(f"  MER: {output.mer:.4f}")
                print(f"  Total words in reference: {output.hits + output.substitutions + output.deletions}")
            except Exception as e:
                print(f"  Error calculating metrics: {e}")
        print()

if __name__ == '__main__':
    main() 