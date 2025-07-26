#!/usr/bin/env python3

import json
import os
import argparse
from datetime import datetime

def load_missing_analysis(analysis_file):
    """Load missing files analysis"""
    with open(analysis_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_report(missing_analysis, output_file):
    """Generate detailed analysis report"""
    
    report_lines = []
    report_lines.append("Missing ASR Files Analysis Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary
    report_lines.append("SUMMARY")
    report_lines.append("-" * 20)
    total_missing = sum(len(files) for files in missing_analysis['missing_files'].values())
    report_lines.append(f"Total missing files across all models: {total_missing}")
    report_lines.append(f"Models analyzed: {', '.join(missing_analysis['summary']['models_analyzed'])}")
    report_lines.append(f"Expected files per model: {missing_analysis['summary']['total_expected']}")
    report_lines.append("")
    
    # Per-model analysis
    report_lines.append("PER-MODEL ANALYSIS")
    report_lines.append("-" * 20)
    
    for model in missing_analysis['summary']['models_analyzed']:
        missing_files = missing_analysis['missing_files'].get(model, [])
        missing_count = len(missing_files)
        success_count = missing_analysis['summary']['total_expected'] - missing_count
        
        report_lines.append(f"{model}:")
        report_lines.append(f"  - Successful: {success_count}")
        report_lines.append(f"  - Missing: {missing_count}")
        if missing_count > 0:
            report_lines.append(f"  - Success rate: {success_count/missing_analysis['summary']['total_expected']*100:.1f}%")
        report_lines.append("")
    
    # Detailed missing files analysis
    report_lines.append("DETAILED MISSING FILES ANALYSIS")
    report_lines.append("-" * 35)
    
    for model, missing_files in missing_analysis['missing_files'].items():
        if not missing_files:
            continue
            
        report_lines.append(f"\n{model.upper()}:")
        report_lines.append(f"Missing {len(missing_files)} files:")
        
        for file in missing_files:
            analysis = missing_analysis['audio_analysis'][model][file]
            report_lines.append(f"  - {file}.wav")
            
            if analysis['possible_reasons']:
                report_lines.append(f"    Possible reasons:")
                for reason in analysis['possible_reasons']:
                    report_lines.append(f"      * {reason}")
            
            if analysis['audio_exists']:
                size_mb = analysis['audio_size'] / (1024 * 1024)
                report_lines.append(f"    File size: {size_mb:.2f} MB")
            else:
                report_lines.append(f"    File size: NOT FOUND")
    
    # Common issues summary
    report_lines.append("\nCOMMON ISSUES SUMMARY")
    report_lines.append("-" * 25)
    
    issue_counts = {}
    for model in missing_analysis['audio_analysis']:
        for file, analysis in missing_analysis['audio_analysis'][model].items():
            for reason in analysis['possible_reasons']:
                issue_counts[reason] = issue_counts.get(reason, 0) + 1
    
    for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"  - {issue}: {count} files")
    
    # Recommendations
    report_lines.append("\nRECOMMENDATIONS")
    report_lines.append("-" * 15)
    
    if total_missing > 0:
        report_lines.append("1. Check audio file integrity and format")
        report_lines.append("2. Verify audio files are not corrupted")
        report_lines.append("3. Consider splitting very long audio files (>10 minutes)")
        report_lines.append("4. Check available disk space and memory")
        report_lines.append("5. Verify ASR model dependencies are properly installed")
        report_lines.append("6. Run the generated re-run script to process missing files")
    else:
        report_lines.append("✓ All files processed successfully!")
    
    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Analysis report generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate analysis report")
    parser.add_argument("--analysis_file", required=True, help="Missing analysis JSON file")
    parser.add_argument("--output_file", required=True, help="Output report file")
    
    args = parser.parse_args()
    
    # Load analysis
    missing_analysis = load_missing_analysis(args.analysis_file)
    
    # Generate report
    generate_report(missing_analysis, args.output_file)

if __name__ == '__main__':
    main()
