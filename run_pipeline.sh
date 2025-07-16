#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Workflow Overview ---
# This script automates a two-step ASR evaluation process:
# 1. `run_all_asrs.py`: Transcribes audio files from AUDIO_DIR and saves the
#    resulting .txt transcripts back into the SAME AUDIO_DIR. No temporary
#    directory is used, and no files are deleted automatically.
# 2. `evaluate_asr.py`: Compares the generated .txt transcripts against a
#    ground truth file to calculate WER and other metrics, saving the
#    results to OUTPUT_FILE.

# --- User Configuration ---
# Directory containing the .wav files to be transcribed.
# Transcript .txt files will also be saved here.
AUDIO_DIR="/media/meow/One Touch/ems_call/random_samples_1_preprocessed"

# Path to the ground truth CSV file for evaluation.
# Must contain 'Filename' and 'transcript' columns.
GROUND_TRUTH_FILE="/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv"

# Path to save the final evaluation report CSV.
OUTPUT_FILE="asr_evaluation_results.csv"

# Python interpreter to use.
PYTHON_EXEC="python3"
# --------------------


# --- Step 1: Install Dependencies ---
echo "--- Checking and installing required Python libraries... ---"
# Install all dependencies for both Python scripts.
$PYTHON_EXEC -m pip install pandas jiwer torch transformers torchaudio "nemo_toolkit[asr]" openai-whisper tqdm

echo -e "\n--- Dependencies installation complete ---"


# --- Step 2: Generate ASR Transcripts ---
# `run_all_asrs.py` will process all .wav files in AUDIO_DIR.
# The output transcripts (e.g., 'model-name_audio-file.txt') are saved
# directly alongside the original .wav files in AUDIO_DIR.
echo -e "\n--- Running ASR transcription on files in: $AUDIO_DIR ---"
$PYTHON_EXEC ems_call/run_all_asrs.py "$AUDIO_DIR"
echo -e "\n--- Transcription finished. TXT files saved in $AUDIO_DIR ---"


# --- Step 3: Evaluate Transcripts and Generate Report ---
# `evaluate_asr.py` reads the transcripts from AUDIO_DIR, compares them
# to the ground truth, and generates a final report.
echo -e "\n--- Evaluating transcripts against ground truth ---"
$PYTHON_EXEC ems_call/evaluate_asr.py \
    --transcript_dirs "$AUDIO_DIR" \
    --ground_truth_file "$GROUND_TRUTH_FILE" \
    --output_file "$OUTPUT_FILE"

echo -e "\n--- Script executed successfully ---"
echo "Evaluation report saved to: $OUTPUT_FILE" 