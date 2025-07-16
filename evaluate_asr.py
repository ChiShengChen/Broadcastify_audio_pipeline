import os
import pandas as pd
import jiwer
import glob
from collections import defaultdict

# --- Configuration ---
GROUND_TRUTH_FILE = '/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv'
# Directories containing the text files from various models
TRANSCRIPT_DIRS = [
    # '/media/meow/One Touch/ems_call/random_samples_1',
    # '/media/meow/One Touch/ems_call/random_samples_2'
    '/media/meow/One Touch/ems_call/random_samples_1_preprocessed',
    '/media/meow/One Touch/ems_call/random_samples_2_preprocessed'
]
OUTPUT_CSV_FILE = '/media/meow/One Touch/ems_call/asr_preprocessed_evaluation_results.csv'

# Known model prefixes used in the filenames.
# The script will try to match these at the start of filenames.
KNOWN_MODEL_PREFIXES = ['large-v3', 'wav2vec-xls-r', 'parakeet-tdt-0.6b-v2', 'canary-1b']
# ---------------------

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

def parse_filename(filepath):
    """
    Parses a filepath like '.../large-v3_202412010133-841696-14744_call_2.txt'
    and returns the model name and the ground truth key (e.g., '..._call_2.wav').
    Returns: (model_name, ground_truth_key) or (None, None)
    """
    basename = os.path.basename(filepath)
    for prefix in KNOWN_MODEL_PREFIXES:
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

def main():
    """Main function to run the ASR evaluation."""
    ground_truth_map = load_ground_truth(GROUND_TRUTH_FILE)
    if ground_truth_map is None:
        return

    # A dictionary to hold lists of references and hypotheses for each model
    # e.g., {'large-v3': {'refs': [...], 'hyps': [...]}, ...}
    model_data = defaultdict(lambda: {'refs': [], 'hyps': []})

    print("Searching for transcript files...")
    all_txt_files = []
    for directory in TRANSCRIPT_DIRS:
        path = os.path.join(directory, '*.txt')
        found_files = glob.glob(path)
        print(f"Found {len(found_files)} .txt files in {directory}")
        all_txt_files.extend(found_files)

    print(f"\nProcessing a total of {len(all_txt_files)} files...")
    matched_files_count = 0
    unmatched_files = []

    for txt_file in all_txt_files:
        model_name, gt_key = parse_filename(txt_file)
        
        if model_name and gt_key in ground_truth_map:
            # Pre-process the text here before appending
            reference_text = transformation(ground_truth_map[gt_key])
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                hypothesis_text = transformation(f.read())

            if isinstance(reference_text, str) and hypothesis_text:
                model_data[model_name]['refs'].append(reference_text)
                model_data[model_name]['hyps'].append(hypothesis_text)
                matched_files_count += 1
        else:
            unmatched_files.append(os.path.basename(txt_file))


    print(f"\nFinished processing. Matched {matched_files_count} files with ground truth.")
    if unmatched_files:
        print(f"Could not match {len(unmatched_files)} files. Examples: {unmatched_files[:5]}")

    if not model_data:
        print("No model data could be collected. Exiting.")
        return

    # --- Calculate metrics for each model ---
    all_results = []
    print("\nCalculating metrics for each model...")
    for model_name, data in sorted(model_data.items()):
        print(f" - Model: {model_name} ({len(data['refs'])} files)")
        if not data['refs']:
            continue

        # The transformation is now applied before calling this function,
        # so we pass the raw lists of strings directly.
        output = jiwer.process_words(
            data['refs'],
            data['hyps']
        )

        result = {
            'Model': model_name,
            'WER': output.wer,
            'MER': output.mer,
            'WIL': output.wil,
            'Substitutions': output.substitutions,
            'Deletions': output.deletions,
            'Insertions': output.insertions,
            'Hits': output.hits,
            # In older jiwer versions, the total number of words in the reference
            # is the sum of hits, substitutions, and deletions.
            'Total_Words_in_Reference': output.hits + output.substitutions + output.deletions,
            'Total_Files_Matched': len(data['refs'])
        }
        all_results.append(result)

    if not all_results:
        print("\nNo results could be calculated. Please check file names and content.")
        return

    # --- Save and display results ---
    results_df = pd.DataFrame(all_results).sort_values(by='WER').reset_index(drop=True)
    results_df.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print(f"\nEvaluation complete. Results saved to {OUTPUT_CSV_FILE}")
    print("--- ASR Evaluation Report ---")
    print(results_df.to_string())
    print("-----------------------------")


if __name__ == '__main__':
    main() 