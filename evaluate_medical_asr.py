import os
import pandas as pd
import jiwer
import glob
import argparse
from collections import defaultdict
# python3 evaluate_medical_asr.py \
# --hypotheses_dir /media/meow/One\ Touch/ems_call/medical_asr_recording_dataset/audio/ \
# --reference_dir /media/meow/One\ Touch/ems_call/medical_asr_recording_dataset/reference_transcripts/ \
# --output_file medical_asr_evaluation_results.csv
# A standard transformation for both reference and hypothesis strings
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
])

def load_ground_truth(reference_dir):
    """Loads all ground truth transcripts from a directory."""
    ground_truth_map = {}
    reference_files = glob.glob(os.path.join(reference_dir, '*.txt'))
    if not reference_files:
        print(f"Error: No reference .txt files found in {reference_dir}")
        return None
    
    for ref_file in reference_files:
        basename = os.path.basename(ref_file)
        with open(ref_file, 'r', encoding='utf-8') as f:
            ground_truth_map[basename] = f.read().strip()
            
    return ground_truth_map

def parse_hypothesis_filename(filepath, known_model_prefixes):
    """
    Parses a hypothesis filepath to extract model name and original transcript filename.
    e.g., '.../audio/large-v3_train_0000.txt' -> ('large-v3', 'train_0000.txt')
    Returns: (model_name, reference_filename) or (None, None)
    """
    basename = os.path.basename(filepath)
    for prefix in known_model_prefixes:
        if basename.startswith(prefix + '_'):
            model_name = prefix
            reference_filename = basename[len(prefix) + 1:]
            return model_name, reference_filename
    return None, None

def main():
    """Main function to run the ASR evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate medical ASR model outputs.")
    parser.add_argument(
        "--hypotheses_dir",
        type=str,
        required=True,
        help="Directory containing the ASR-generated transcript .txt files (hypotheses)."
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        required=True,
        help="Directory containing the ground truth transcript .txt files (references)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the final evaluation results CSV file."
    )
    parser.add_argument(
        '--model_prefixes',
        nargs='+',
        default=['large-v3', 'wav2vec-xls-r', 'parakeet-tdt-0.6b-v2', 'canary-1b'],
        help='Known model prefixes used in the filenames.'
    )
    args = parser.parse_args()

    ground_truth_map = load_ground_truth(args.reference_dir)
    if ground_truth_map is None:
        return

    model_data = defaultdict(lambda: {'refs': [], 'hyps': []})

    print(f"Searching for hypothesis files in: {args.hypotheses_dir}")
    hypothesis_files = glob.glob(os.path.join(args.hypotheses_dir, '*.txt'))
    print(f"Found {len(hypothesis_files)} total .txt files.")

    # Exclude reference files from hypothesis files
    reference_filenames = set(os.path.basename(f) for f in glob.glob(os.path.join(args.reference_dir, '*.txt')))
    hypothesis_files = [f for f in hypothesis_files if os.path.basename(f) not in reference_filenames]
    print(f"Found {len(hypothesis_files)} hypothesis files to process.")


    matched_files_count = 0
    unmatched_files = []

    for hyp_file in hypothesis_files:
        model_name, ref_filename = parse_hypothesis_filename(hyp_file, args.model_prefixes)
        
        if model_name and ref_filename in ground_truth_map:
            reference_text = transformation(ground_truth_map[ref_filename])
            
            with open(hyp_file, 'r', encoding='utf-8') as f:
                hypothesis_text = transformation(f.read())

            if isinstance(reference_text, str) and hypothesis_text:
                model_data[model_name]['refs'].append(reference_text)
                model_data[model_name]['hyps'].append(hypothesis_text)
                matched_files_count += 1
        else:
            if ref_filename:
                unmatched_files.append(os.path.basename(hyp_file))


    print(f"\nFinished processing. Matched {matched_files_count} files with ground truth.")
    if unmatched_files:
        print(f"Could not match {len(unmatched_files)} files. Examples: {unmatched_files[:5]}")

    if not model_data:
        print("No model data could be collected. Exiting.")
        return

    all_results = []
    print("\nCalculating metrics for each model...")
    for model_name, data in sorted(model_data.items()):
        print(f" - Model: {model_name} ({len(data['refs'])} files)")
        if not data['refs']:
            continue
            
        output = jiwer.process_words(data['refs'], data['hyps'])

        result = {
            'Model': model_name,
            'WER': output.wer,
            'MER': output.mer,
            'WIL': output.wil,
            'Substitutions': output.substitutions,
            'Deletions': output.deletions,
            'Insertions': output.insertions,
            'Hits': output.hits,
            'Total_Words_in_Reference': output.hits + output.substitutions + output.deletions,
            'Total_Files_Matched': len(data['refs'])
        }
        all_results.append(result)

    if not all_results:
        print("\nNo results could be calculated. Please check file names and content.")
        return

    results_df = pd.DataFrame(all_results).sort_values(by='WER').reset_index(drop=True)
    results_df.to_csv(args.output_file, index=False)
    
    print(f"\nEvaluation complete. Results saved to {args.output_file}")
    print("--- Medical ASR Evaluation Report ---")
    print(results_df.to_string())
    print("-------------------------------------")

if __name__ == '__main__':
    main() 