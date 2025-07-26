#!/usr/bin/env python3

import os
import sys
import json
import argparse
import pandas as pd
from collections import defaultdict

def parse_model_analysis(analysis_file):
    """Parse model_file_analysis.txt to extract successful files for each model"""
    model_files = defaultdict(set)
    current_model = None
    in_successful_section = False
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('=== ') and line.endswith(' ==='):
                current_model = line.strip('= ').strip()
                in_successful_section = False
            elif line.strip() == 'Successful files:':
                in_successful_section = True
            elif in_successful_section and line.strip().startswith('✓'):
                filename = line.strip()[2:].strip()
                if current_model:
                    # Extract the base filename (remove model prefix and .txt extension)
                    if '_' in filename:
                        parts = filename.split('_', 1)
                        if len(parts) == 2:
                            base_filename = parts[1].replace('.txt', '')
                            model_files[current_model].add(base_filename)
                        else:
                            base_filename = filename.replace('.txt', '')
                            model_files[current_model].add(base_filename)
                    else:
                        base_filename = filename.replace('.txt', '')
                        model_files[current_model].add(base_filename)
            elif in_successful_section and not line.strip():
                in_successful_section = False
    return model_files

def get_expected_files(ground_truth_file):
    gt_df = pd.read_csv(ground_truth_file)
    expected_files = set()
    for _, row in gt_df.iterrows():
        filename = row['Filename']
        base_name = os.path.splitext(filename)[0]
        expected_files.add(base_name)
    return expected_files

def find_missing_files(model_files, expected_files, models):
    missing_files = defaultdict(list)
    for model in models:
        successful_files = model_files.get(model, set())
        for expected_file in expected_files:
            if expected_file not in successful_files:
                missing_files[model].append(expected_file)
    return missing_files

def analyze_audio_files(audio_dir, missing_files):
    analysis = {}
    for model, files in missing_files.items():
        analysis[model] = {}
        for file in files:
            audio_file = os.path.join(audio_dir, f"{file}.wav")
            analysis[model][file] = {
                'audio_exists': os.path.exists(audio_file),
                'audio_size': os.path.getsize(audio_file) if os.path.exists(audio_file) else 0,
                'possible_reasons': []
            }
            if not os.path.exists(audio_file):
                analysis[model][file]['possible_reasons'].append("Audio file not found")
            else:
                size_mb = analysis[model][file]['audio_size'] / (1024 * 1024)
                if size_mb > 100:
                    analysis[model][file]['possible_reasons'].append("Very large audio file (>100MB)")
                elif size_mb < 0.1:
                    analysis[model][file]['possible_reasons'].append("Very small audio file (<100KB)")
                try:
                    import librosa
                    duration = librosa.get_duration(path=audio_file)
                    if duration > 600:
                        analysis[model][file]['possible_reasons'].append(f"Very long audio ({duration:.1f}s)")
                    elif duration < 1:
                        analysis[model][file]['possible_reasons'].append(f"Very short audio ({duration:.1f}s)")
                except:
                    analysis[model][file]['possible_reasons'].append("Could not analyze audio duration")
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Analyze missing ASR files")
    parser.add_argument("--analysis_file", required=True, help="model_file_analysis.txt path")
    parser.add_argument("--ground_truth_file", required=True, help="Ground truth CSV file")
    parser.add_argument("--audio_dir", required=True, help="Original audio directory")
    parser.add_argument("--models", required=True, help="Comma-separated list of models")
    parser.add_argument("--output_file", required=True, help="Output JSON file")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(',')]
    model_files = parse_model_analysis(args.analysis_file)
    expected_files = get_expected_files(args.ground_truth_file)
    missing_files = find_missing_files(model_files, expected_files, models)
    audio_analysis = analyze_audio_files(args.audio_dir, missing_files)
    result = {
        'missing_files': dict(missing_files),
        'audio_analysis': audio_analysis,
        'summary': {
            'total_expected': len(expected_files),
            'models_analyzed': models
        }
    }
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("=== Missing Files Analysis ===")
    for model in models:
        missing_count = len(missing_files[model])
        successful_count = len(model_files.get(model, set()))
        print(f"{model}: {successful_count} successful, {missing_count} missing")
        if missing_count > 0:
            for file in missing_files[model][:5]:
                print(f"  - {file}")
            if len(missing_files[model]) > 5:
                print(f"  ... and {len(missing_files[model]) - 5} more")

if __name__ == '__main__':
    main()
