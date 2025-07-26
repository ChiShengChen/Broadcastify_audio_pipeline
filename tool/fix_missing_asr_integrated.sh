#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Integrated Fix Missing ASR Files Script ---
# 1. 分析 model_file_analysis.txt，找出每個模型缺漏的檔案
# 2. 產生 rerun script 只補跑缺漏檔案
# 3. 產生詳細報告（含可能原因）
# 4. 提供範例用法

# --- Configuration ---
PYTHON_EXEC="python3"

# Usage
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "\nUsage: $0 --pipeline_output_dir DIR --fix_output_dir DIR --ground_truth_file FILE [--models MODEL1,MODEL2]"
    echo "\nOptions:"
    echo "  --pipeline_output_dir DIR    Pipeline output directory containing model_file_analysis.txt"
    echo "  --fix_output_dir DIR         Output directory for fixed results"
    echo "  --ground_truth_file FILE     Ground truth CSV file"
    echo "  --models MODEL1,MODEL2       Comma-separated list of models to fix (default: all)"
    echo "  -h, --help                   Show this help"
    echo "\nExamples:"
    echo "  $0 --pipeline_output_dir /path/to/pipeline_results --fix_output_dir /path/to/fix_results --ground_truth_file /path/to/gt.csv"
    echo "  $0 --pipeline_output_dir /path/to/pipeline_results --fix_output_dir /path/to/fix_results --ground_truth_file /path/to/gt.csv --models large-v3,canary-1b"
    echo "\nWhat this script does:"
    echo "1. 分析 model_file_analysis.txt，找出缺漏檔案"
    echo "2. 產生詳細報告與 rerun script"
    echo "3. 報告包含缺漏原因分析與修復建議"
    echo "4. rerun script 只補跑缺漏檔案，並自動整理結果"
    exit 0
fi

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --pipeline_output_dir)
            PIPELINE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --fix_output_dir)
            FIX_OUTPUT_DIR="$2"
            shift 2
            ;;
        --ground_truth_file)
            GROUND_TRUTH_FILE="$2"
            shift 2
            ;;
        --models)
            MODELS_TO_FIX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# --- Validate required arguments ---
if [[ -z "$PIPELINE_OUTPUT_DIR" ]]; then
    echo "Error: --pipeline_output_dir is required"
    exit 1
fi
if [[ -z "$FIX_OUTPUT_DIR" ]]; then
    echo "Error: --fix_output_dir is required"
    exit 1
fi
if [[ -z "$GROUND_TRUTH_FILE" ]]; then
    echo "Error: --ground_truth_file is required"
    exit 1
fi

# Set default models if not specified
if [[ -z "$MODELS_TO_FIX" ]]; then
    MODELS_TO_FIX="large-v3,canary-1b,parakeet-tdt-0.6b-v2,wav2vec-xls-r"
fi
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS_TO_FIX"

# --- Paths ---
MODEL_ANALYSIS_FILE="$PIPELINE_OUTPUT_DIR/model_file_analysis.txt"
MERGED_TRANSCRIPTS_DIR="$PIPELINE_OUTPUT_DIR/merged_transcripts"

# --- Check files ---
if [[ ! -d "$PIPELINE_OUTPUT_DIR" ]]; then
    echo "Error: Pipeline output directory not found: $PIPELINE_OUTPUT_DIR"
    exit 1
fi
if [[ ! -f "$MODEL_ANALYSIS_FILE" ]]; then
    echo "Error: model_file_analysis.txt not found in: $PIPELINE_OUTPUT_DIR"
    exit 1
fi
mkdir -p "$FIX_OUTPUT_DIR"

# --- Step 1: Analyze Missing Files ---
echo "--- Step 1: Analyzing Missing Files ---"
cat > "$FIX_OUTPUT_DIR/analyze_missing.py" << 'EOF'
import os
import sys
import json
import argparse
import pandas as pd
from collections import defaultdict

def parse_model_analysis(analysis_file):
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
EOF

# Run the analysis
echo "Running missing files analysis..."
$PYTHON_EXEC "$FIX_OUTPUT_DIR/analyze_missing.py" \
    --analysis_file "$MODEL_ANALYSIS_FILE" \
    --ground_truth_file "$GROUND_TRUTH_FILE" \
    --audio_dir "$(dirname "$GROUND_TRUTH_FILE")" \
    --models "$MODELS_TO_FIX" \
    --output_file "$FIX_OUTPUT_DIR/missing_analysis.json"
echo "Missing files analysis completed"
echo ""

# --- Step 2: Generate Re-run Script ---
echo "--- Step 2: Generating Re-run Script ---"
cat > "$FIX_OUTPUT_DIR/generate_rerun.py" << 'EOF'
import json
import os
import argparse

def load_missing_analysis(analysis_file):
    with open(analysis_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_rerun_script(missing_analysis, output_dir, pipeline_dir, ground_truth_file):
    script_content = []
    script_content.append("#!/bin/bash")
    script_content.append("set -e")
    script_content.append("")
    script_content.append("# Auto-generated script to re-run ASR for missing files")
    script_content.append("")
    script_content.append("# Create output directories")
    script_content.append(f"mkdir -p {output_dir}/audio_segments")
    script_content.append(f"mkdir -p {output_dir}/asr_transcripts")
    script_content.append(f"mkdir -p {output_dir}/merged_transcripts")
    script_content.append("")
    for model, missing_files in missing_analysis['missing_files'].items():
        if not missing_files:
            continue
        script_content.append(f"echo '=== Processing {model} ==='")
        model_dir = f"{output_dir}/audio_segments/{model}"
        script_content.append(f"mkdir -p {model_dir}")
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
        script_content.append(f"echo 'Running ASR for {model}...'")
        script_content.append(f"python3 run_all_asrs.py {model_dir}")
        script_content.append("")
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
    script_content.append("echo '=== Copying existing transcripts for reference ==='")
    script_content.append(f"if [ -d '{pipeline_dir}/merged_transcripts' ]; then")
    script_content.append(f"    cp -r '{pipeline_dir}/merged_transcripts'/* '{output_dir}/merged_transcripts/' 2>/dev/null || true")
    script_content.append("    echo 'Copied existing transcripts'")
    script_content.append("fi")
    script_content.append("")
    script_content.append("echo '=== Merging transcripts ==='")
    script_content.append(f"if [ -d '{pipeline_dir}/long_audio_segments' ]; then")
    script_content.append(f"    python3 merge_split_transcripts.py \\")
    script_content.append(f"        --input_dir '{output_dir}/asr_transcripts' \\")
    script_content.append(f"        --output_dir '{output_dir}/merged_transcripts' \\")
    script_content.append(f"        --metadata_dir '{pipeline_dir}/long_audio_segments'")
    script_content.append("    echo 'Transcript merging completed'")
    script_content.append("fi")
    script_content.append("")
    script_content.append("echo '=== Running evaluation ==='")
    script_content.append(f"python3 evaluate_asr.py \\")
    script_content.append(f"    --transcript_dirs '{output_dir}/merged_transcripts' \\")
    script_content.append(f"    --ground_truth_file '{ground_truth_file}' \\")
    script_content.append(f"    --output_file '{output_dir}/asr_evaluation_results_fixed.csv'")
    script_content.append("")
    script_content.append("echo '=== Running model file analysis ==='")
    script_content.append(f"python3 tool/analyze_model_files.py \\")
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
    missing_analysis = load_missing_analysis(args.analysis_file)
    script_content = create_rerun_script(
        missing_analysis, 
        args.output_dir, 
        args.pipeline_dir, 
        args.ground_truth_file
    )
    with open(args.script_file, 'w') as f:
        f.write(script_content)
    os.chmod(args.script_file, 0o755)
    print(f"Re-run script generated: {args.script_file}")
if __name__ == '__main__':
    main()
EOF

# Generate re-run script
echo "Generating re-run script..."
$PYTHON_EXEC "$FIX_OUTPUT_DIR/generate_rerun.py" \
    --analysis_file "$FIX_OUTPUT_DIR/missing_analysis.json" \
    --output_dir "$FIX_OUTPUT_DIR" \
    --pipeline_dir "$PIPELINE_OUTPUT_DIR" \
    --ground_truth_file "$GROUND_TRUTH_FILE" \
    --script_file "$FIX_OUTPUT_DIR/rerun_missing_asr.sh"
echo "Re-run script generated: $FIX_OUTPUT_DIR/rerun_missing_asr.sh"
echo ""

# --- Step 3: Generate Analysis Report ---
echo "--- Step 3: Generating Analysis Report ---"
cat > "$FIX_OUTPUT_DIR/generate_report.py" << 'EOF'
import json
import os
import argparse
from datetime import datetime

def load_missing_analysis(analysis_file):
    with open(analysis_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_report(missing_analysis, output_file):
    report_lines = []
    report_lines.append("Missing ASR Files Analysis Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("SUMMARY")
    report_lines.append("-" * 20)
    total_missing = sum(len(files) for files in missing_analysis['missing_files'].values())
    report_lines.append(f"Total missing files across all models: {total_missing}")
    report_lines.append(f"Models analyzed: {', '.join(missing_analysis['summary']['models_analyzed'])}")
    report_lines.append(f"Expected files per model: {missing_analysis['summary']['total_expected']}")
    report_lines.append("")
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
    if total_missing > 0:
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
        report_lines.append("\nCOMMON ISSUES SUMMARY")
        report_lines.append("-" * 25)
        issue_counts = {}
        for model in missing_analysis['audio_analysis']:
            for file, analysis in missing_analysis['audio_analysis'][model].items():
                for reason in analysis['possible_reasons']:
                    issue_counts[reason] = issue_counts.get(reason, 0) + 1
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  - {issue}: {count} files")
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
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"Analysis report generated: {output_file}")
def main():
    parser = argparse.ArgumentParser(description="Generate analysis report")
    parser.add_argument("--analysis_file", required=True, help="Missing analysis JSON file")
    parser.add_argument("--output_file", required=True, help="Output report file")
    args = parser.parse_args()
    missing_analysis = load_missing_analysis(args.analysis_file)
    generate_report(missing_analysis, args.output_file)
if __name__ == '__main__':
    main()
EOF

echo "Generating detailed analysis report..."
$PYTHON_EXEC "$FIX_OUTPUT_DIR/generate_report.py" \
    --analysis_file "$FIX_OUTPUT_DIR/missing_analysis.json" \
    --output_file "$FIX_OUTPUT_DIR/missing_files_report.txt"
echo "Analysis report generated: $FIX_OUTPUT_DIR/missing_files_report.txt"
echo ""

# --- Step 4: Display Results ---
echo "--- Step 4: Display Results ---"
echo "=== Summary ==="
if command -v jq > /dev/null 2>&1; then
    TOTAL_MISSING=$(jq -r '[.missing_files[] | length] | add' "$FIX_OUTPUT_DIR/missing_analysis.json")
    echo "Total missing files: $TOTAL_MISSING"
    echo ""
    echo "Missing files per model:"
    for model in "${MODEL_ARRAY[@]}"; do
        MISSING_COUNT=$(jq -r ".missing_files[\"$model\"] | length" "$FIX_OUTPUT_DIR/missing_analysis.json")
        echo "  $model: $MISSING_COUNT"
    done
else
    echo "Install jq for better JSON parsing, or check the generated files:"
    echo "  - $FIX_OUTPUT_DIR/missing_analysis.json"
    echo "  - $FIX_OUTPUT_DIR/missing_files_report.txt"
fi
echo ""
echo "=== Generated Files ==="
echo "1. Missing analysis: $FIX_OUTPUT_DIR/missing_analysis.json"
echo "2. Re-run script: $FIX_OUTPUT_DIR/rerun_missing_asr.sh"
echo "3. Detailed report: $FIX_OUTPUT_DIR/missing_files_report.txt"
echo ""
echo "=== Next Steps ==="
echo "1. Review the detailed report: $FIX_OUTPUT_DIR/missing_files_report.txt"
echo "2. Run the re-run script: $FIX_OUTPUT_DIR/rerun_missing_asr.sh"
echo "3. Check the fixed results in: $FIX_OUTPUT_DIR"
echo ""
if [[ -f "$FIX_OUTPUT_DIR/missing_files_report.txt" ]]; then
    echo "=== Quick Preview of Report ==="
    head -20 "$FIX_OUTPUT_DIR/missing_files_report.txt"
    echo "..."
    echo ""
fi
echo "=== Script Completed ==="

# --- Example Usage ---
if [[ "$1" == "--example" ]]; then
    echo "\n=== Example Usage ==="
    echo "./fix_missing_asr_integrated.sh \\"
    echo "    --pipeline_output_dir /path/to/pipeline_results_20250726_070330 \\"
    echo "    --fix_output_dir /path/to/fix_results \\"
    echo "    --ground_truth_file /path/to/long_audio_ground_truth.csv"
    echo "\n# Or指定模型："
    echo "./fix_missing_asr_integrated.sh \\"
    echo "    --pipeline_output_dir /path/to/pipeline_results_20250726_070330 \\"
    echo "    --fix_output_dir /path/to/fix_results \\"
    echo "    --ground_truth_file /path/to/long_audio_ground_truth.csv \\"
    echo "    --models large-v3,canary-1b"
    echo ""
    exit 0
fi 