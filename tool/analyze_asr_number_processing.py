#!/usr/bin/env python3
"""
Analyze how different ASR models process numbers
"""

import os
import re
import pandas as pd
from collections import defaultdict
import glob

def extract_numbers_from_text(text):
    """Extract all numbers from text and categorize them"""
    numbers = []
    
    # Different number patterns
    patterns = {
        'time_format': r'(\d{1,2}):(\d{2})',  # 11:17, 20:28
        'phone_code': r'(\d)-(\d)-(\d)',      # 6-1-2
        'address_number': r'\b(\d{3,4})\b',   # 4560, 132
        'single_digit': r'\b(\d)\b',          # 1, 2, 3
        'double_digit': r'\b(\d{2})\b',       # 11, 20, 95
        'large_number': r'\b(\d{5,})\b',      # 1541, 20241201
    }
    
    for pattern_name, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            numbers.append({
                'type': pattern_name,
                'value': match.group(0),
                'position': match.start(),
                'context': text[max(0, match.start()-20):match.end()+20]
            })
    
    return numbers

def analyze_number_processing(transcript_dir, ground_truth_file):
    """Analyze how different ASR models process numbers"""
    
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_file)
    gt_dict = dict(zip(gt_df['Filename'], gt_df['transcript']))
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(transcript_dir) 
                  if os.path.isdir(os.path.join(transcript_dir, d))]
    
    print("=== ASR Model Number Processing Analysis ===")
    print(f"Models found: {model_dirs}")
    print()
    
    # Analyze each model
    model_analysis = {}
    
    for model in model_dirs:
        print(f"=== Analyzing {model} ===")
        model_dir = os.path.join(transcript_dir, model)
        transcript_files = glob.glob(os.path.join(model_dir, "*.txt"))
        
        model_stats = {
            'total_files': len(transcript_files),
            'number_instances': [],
            'number_types': defaultdict(int),
            'processing_patterns': defaultdict(list)
        }
        
        for transcript_file in transcript_files:
            filename = os.path.basename(transcript_file)
            # Extract original filename from transcript filename
            # Format: model_original_filename.txt
            original_filename = '_'.join(filename.split('_')[1:]).replace('.txt', '.wav')
            
            if original_filename in gt_dict:
                # Read transcript
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_text = f.read().strip()
                
                # Read ground truth
                gt_text = gt_dict[original_filename]
                
                # Extract numbers from both
                transcript_numbers = extract_numbers_from_text(transcript_text)
                gt_numbers = extract_numbers_from_text(gt_text)
                
                # Analyze number processing
                for gt_num in gt_numbers:
                    gt_value = gt_num['value']
                    gt_type = gt_num['type']
                    
                    # Check if this number appears in transcript
                    found_in_transcript = False
                    transcript_versions = []
                    
                    for trans_num in transcript_numbers:
                        if trans_num['value'] == gt_value:
                            found_in_transcript = True
                            transcript_versions.append(trans_num['value'])
                        elif trans_num['type'] == gt_type:
                            # Same type but different value
                            transcript_versions.append(trans_num['value'])
                    
                    model_stats['number_instances'].append({
                        'file': original_filename,
                        'gt_number': gt_value,
                        'gt_type': gt_type,
                        'found_in_transcript': found_in_transcript,
                        'transcript_versions': transcript_versions,
                        'gt_context': gt_num['context'],
                        'transcript_text': transcript_text[:200] + '...' if len(transcript_text) > 200 else transcript_text
                    })
                    
                    model_stats['number_types'][gt_type] += 1
                    
                    if found_in_transcript:
                        model_stats['processing_patterns']['correct'].append(gt_value)
                    else:
                        model_stats['processing_patterns']['missing'].append(gt_value)
        
        model_analysis[model] = model_stats
        
        # Print summary for this model
        print(f"Files processed: {model_stats['total_files']}")
        print(f"Number instances analyzed: {len(model_stats['number_instances'])}")
        print(f"Number types found:")
        for num_type, count in model_stats['number_types'].items():
            print(f"  {num_type}: {count}")
        print(f"Processing accuracy:")
        correct = len(model_stats['processing_patterns']['correct'])
        missing = len(model_stats['processing_patterns']['missing'])
        total = correct + missing
        if total > 0:
            accuracy = correct / total * 100
            print(f"  Correct: {correct}/{total} ({accuracy:.1f}%)")
            print(f"  Missing: {missing}/{total} ({100-accuracy:.1f}%)")
        print()
    
    return model_analysis

def generate_detailed_report(model_analysis, output_file):
    """Generate detailed report of number processing analysis"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ASR Model Number Processing Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        for model, stats in model_analysis.items():
            f.write(f"Model: {model}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total files: {stats['total_files']}\n")
            f.write(f"Number instances: {len(stats['number_instances'])}\n\n")
            
            # Number type breakdown
            f.write("Number types found:\n")
            for num_type, count in stats['number_types'].items():
                f.write(f"  {num_type}: {count}\n")
            f.write("\n")
            
            # Processing accuracy
            correct = len(stats['processing_patterns']['correct'])
            missing = len(stats['processing_patterns']['missing'])
            total = correct + missing
            if total > 0:
                accuracy = correct / total * 100
                f.write(f"Processing accuracy: {accuracy:.1f}%\n")
                f.write(f"  Correct: {correct}/{total}\n")
                f.write(f"  Missing: {missing}/{total}\n\n")
            
            # Detailed examples
            f.write("Detailed examples:\n")
            for i, instance in enumerate(stats['number_instances'][:10]):  # Show first 10
                f.write(f"\nExample {i+1}:\n")
                f.write(f"  File: {instance['file']}\n")
                f.write(f"  Ground truth number: {instance['gt_number']} ({instance['gt_type']})\n")
                f.write(f"  Found in transcript: {instance['found_in_transcript']}\n")
                if instance['transcript_versions']:
                    f.write(f"  Transcript versions: {instance['transcript_versions']}\n")
                f.write(f"  GT context: {instance['gt_context']}\n")
                f.write(f"  Transcript preview: {instance['transcript_text'][:100]}...\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"Detailed report saved to: {output_file}")

def compare_number_processing_across_models(model_analysis):
    """Compare number processing across different models"""
    
    print("=== Cross-Model Number Processing Comparison ===")
    
    # Collect all unique numbers across all models
    all_numbers = set()
    for model, stats in model_analysis.items():
        for instance in stats['number_instances']:
            all_numbers.add(instance['gt_number'])
    
    print(f"Total unique numbers found: {len(all_numbers)}")
    print()
    
    # Create comparison table
    comparison_data = []
    for number in sorted(all_numbers):
        row = {'number': number}
        for model in model_analysis.keys():
            # Find this number in the model's data
            found = False
            for instance in model_analysis[model]['number_instances']:
                if instance['gt_number'] == number:
                    found = instance['found_in_transcript']
                    break
            row[model] = '✓' if found else '✗'
        comparison_data.append(row)
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(comparison_data)
    print("Number processing comparison:")
    print(df.to_string(index=False))
    
    # Calculate overall accuracy per model
    print("\nOverall accuracy by model:")
    for model in model_analysis.keys():
        correct = len(model_analysis[model]['processing_patterns']['correct'])
        missing = len(model_analysis[model]['processing_patterns']['missing'])
        total = correct + missing
        if total > 0:
            accuracy = correct / total * 100
            print(f"  {model}: {accuracy:.1f}% ({correct}/{total})")
    
    return df

def main():
    # Configuration
    transcript_dir = "pipeline_csside_results_20250726_070330/merged_transcripts"
    ground_truth_file = "long_audio_test_dataset/long_audio_ground_truth.csv"
    output_file = "asr_number_processing_analysis.txt"
    
    if not os.path.exists(transcript_dir):
        print(f"Error: Transcript directory not found: {transcript_dir}")
        return
    
    if not os.path.exists(ground_truth_file):
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return
    
    # Run analysis
    model_analysis = analyze_number_processing(transcript_dir, ground_truth_file)
    
    # Generate detailed report
    generate_detailed_report(model_analysis, output_file)
    
    # Compare across models
    comparison_df = compare_number_processing_across_models(model_analysis)
    
    # Save comparison to CSV
    comparison_df.to_csv("asr_number_processing_comparison.csv", index=False)
    print(f"\nComparison saved to: asr_number_processing_comparison.csv")
    
    print(f"\nAnalysis complete!")
    print(f"Detailed report: {output_file}")
    print(f"Comparison table: asr_number_processing_comparison.csv")

if __name__ == '__main__':
    main() 