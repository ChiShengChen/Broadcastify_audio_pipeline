#!/usr/bin/env python3

import json
import os
import argparse

def load_missing_analysis(analysis_file):
    """Load missing files analysis"""
    with open(analysis_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_rerun_script(missing_analysis, output_dir, pipeline_dir, ground_truth_file):
    """Create script to re-run ASR for missing files"""
    
    script_content = []
    script_content.append("#!/bin/bash")
    script_content.append("set -e")
    script_content.append("")
    script_content.append("# Auto-generated script to re-run ASR for missing files")
    script_content.append("")
    
    # Create output directories
    script_content.append("# Create output directories")
    script_content.append(f"mkdir -p {output_dir}/audio_segments")
    script_content.append(f"mkdir -p {output_dir}/asr_transcripts")
    script_content.append(f"mkdir -p {output_dir}/merged_transcripts")
    script_content.append("")
    
    # Process each model
    for model, missing_files in missing_analysis['missing_files'].items():
        if not missing_files:
            continue
            
        script_content.append(f"echo '=== Processing {model} ==='")
        
        # Create model-specific directory
        model_dir = f"{output_dir}/audio_segments/{model}"
        script_content.append(f"mkdir -p {model_dir}")
        
        # Copy audio files for this model
        for file in missing_files:
            audio_file = f"$(dirname '{ground_truth_file}')/{file}.wav"
            target_file = f"{model_dir}/{file}.wav"
            script_content.append(f"if [ -f '{audio_file}' ]; then")
            script_content.append(f"    cp '{audio_file}' '{target_file}'")
            script_content.append(f"    echo 'Copied {file}.wav for {model}'")
            script_content.append("else")
            script_content.append(f"    echo 'Warning: Audio file {file}.wav not found'")
            script_content.append("fi")
        
        script_content.append("")
        
        # Run ASR for this model
        script_content.append(f"echo 'Running ASR for {model}...'")
        script_content.append(f"python3 run_all_asrs.py {model_dir}")
        script_content.append("")
        
        # Move transcripts to final location
        script_content.append(f"echo 'Moving transcripts for {model}...'")
        for file in missing_files:
            transcript_file = f"{model_dir}/{model}_{file}.txt"
            target_file = f"{output_dir}/asr_transcripts/{model}_{file}.txt"
            script_content.append(f"if [ -f '{transcript_file}' ]; then")
            script_content.append(f"    mv '{transcript_file}' '{target_file}'")
            script_content.append(f"    echo 'Moved {model}_{file}.txt'")
            script_content.append("else")
            script_content.append(f"    echo 'Warning: Transcript {model}_{file}.txt not generated'")
            script_content.append("fi")
        
        script_content.append("")
    
    # Copy existing transcripts for reference
    script_content.append("echo '=== Copying existing transcripts for reference ==='")
    script_content.append(f"if [ -d '{pipeline_dir}/merged_transcripts' ]; then")
    script_content.append(f"    cp -r '{pipeline_dir}/merged_transcripts'/* '{output_dir}/merged_transcripts/' 2>/dev/null || true")
    script_content.append("    echo 'Copied existing transcripts'")
    script_content.append("fi")
    script_content.append("")
    
    # Merge transcripts if needed
    script_content.append("echo '=== Merging transcripts ==='")
    script_content.append(f"if [ -d '{pipeline_dir}/long_audio_segments' ]; then")
    script_content.append(f"    python3 merge_split_transcripts.py \\")
    script_content.append(f"        --input_dir '{output_dir}/asr_transcripts' \\")
    script_content.append(f"        --output_dir '{output_dir}/merged_transcripts' \\")
    script_content.append(f"        --metadata_dir '{pipeline_dir}/long_audio_segments'")
    script_content.append("    echo 'Transcript merging completed'")
    script_content.append("fi")
    script_content.append("")
    
    # Run evaluation
    script_content.append("echo '=== Running evaluation ==='")
    script_content.append(f"python3 evaluate_asr.py \\")
    script_content.append(f"    --transcript_dirs '{output_dir}/merged_transcripts' \\")
    script_content.append(f"    --ground_truth_file '{ground_truth_file}' \\")
    script_content.append(f"    --output_file '{output_dir}/asr_evaluation_results_fixed.csv'")
    script_content.append("")
    
    # Run model file analysis
    script_content.append("echo '=== Running model file analysis ==='")
    script_content.append(f"python3 analyze_model_files.py \\")
    script_content.append(f"    --transcript_dir '{output_dir}/merged_transcripts' \\")
    script_content.append(f"    --ground_truth_file '{ground_truth_file}' \\")
    script_content.append(f"    --output_file '{output_dir}/model_file_analysis_fixed.txt'")
    script_content.append("")
    
    script_content.append("echo '=== Re-run completed ==='")
    
    return "\n".join(script_content)

def main():
    parser = argparse.ArgumentParser(description="Generate re-run script")
    parser.add_argument("--analysis_file", required=True, help="Missing analysis JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--pipeline_dir", required=True, help="Original pipeline directory")
    parser.add_argument("--ground_truth_file", required=True, help="Ground truth file")
    parser.add_argument("--script_file", required=True, help="Output script file")
    
    args = parser.parse_args()
    
    # Load analysis
    missing_analysis = load_missing_analysis(args.analysis_file)
    
    # Generate script
    script_content = create_rerun_script(
        missing_analysis, 
        args.output_dir, 
        args.pipeline_dir, 
        args.ground_truth_file
    )
    
    # Write script
    with open(args.script_file, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(args.script_file, 0o755)
    
    print(f"Re-run script generated: {args.script_file}")

if __name__ == '__main__':
    main()
