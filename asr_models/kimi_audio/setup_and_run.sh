#!/bin/bash
set -e

# ==============================================================================
# Kimi-Audio Docker Setup & Run Script
# ==============================================================================
#
# This script automates the setup process for running Kimi-Audio in a
# Docker container. It will:
#
# 1. Configure necessary paths.
# 2. Create a cache directory on an external drive to store models.
# 3. Modify the Dockerfile to use a more compatible CUDA version.
# 4. Create a Python script for batch audio transcription.
# 5. Build the Docker image.
# 6. Provide the final `docker run` command to start processing.
#
# ==> PREREQUISITES <==
# - Docker
# - NVIDIA Drivers
# - NVIDIA Container Toolkit
#
# ==> HOW TO USE <==
# 1. Place this script in the root of your Kimi-Audio project directory.
# 2. Modify the CONFIGURATION variables below to match your system's paths.
# 3. Make the script executable: chmod +x setup_and_run.sh
# 4. Run the script: ./setup_and_run.sh
#
# ==============================================================================

# --- CONFIGURATION: PLEASE EDIT THESE PATHS ---

# The mount point of your large external drive.
# On your current machine, this is "/media/meow/One Touch".
# Change this to match the target machine's path.
EXTERNAL_DRIVE_PATH="/media/meow/One Touch"

# The name of the dataset folder on your external drive.
DATASET_FOLDER_NAME="ems_call/long_calls_filtered"

# The name for the Docker image we will build.
DOCKER_IMAGE_NAME="kimi-audio"

# --- END OF CONFIGURATION ---


# --- SCRIPT LOGIC (No need to edit below this line) ---

# Get the absolute path of the project directory (where this script is located)
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define full paths based on the configuration above
CACHE_DIR="$EXTERNAL_DRIVE_PATH/huggingface_cache"
DATASET_DIR="$EXTERNAL_DRIVE_PATH/$DATASET_FOLDER_NAME"

echo "================================================="
echo " Kimi-Audio Docker Setup"
echo "================================================="
echo "Project Directory: $PROJECT_DIR"
echo "Model Cache Directory: $CACHE_DIR"
echo "Dataset Directory: $DATASET_DIR"
echo "Docker Image Name: $DOCKER_IMAGE_NAME"
echo "-------------------------------------------------"
echo

# Step 1: Create and set permissions for the model cache directory
echo "[STEP 1/4] Setting up model cache directory..."
if [ -d "$CACHE_DIR" ]; then
    echo "Cache directory already exists. Setting permissions."
else
    echo "Creating cache directory at $CACHE_DIR"
    mkdir -p "$CACHE_DIR"
fi
chmod 777 "$CACHE_DIR"
echo "✓ Cache directory is ready."
echo

# Step 2: Modify Dockerfile for better compatibility
echo "[STEP 2/4] Modifying Dockerfile for CUDA 12.1.1 compatibility..."
# This command replaces the CUDA version to prevent driver incompatibility issues.
sed -i 's/FROM nvidia\/cuda:12.8.1-cudnn-devel-ubuntu22.04/FROM nvidia\/cuda:12.1.1-cudnn8-devel-ubuntu22.04/' "$PROJECT_DIR/Dockerfile"
echo "✓ Dockerfile has been updated."
echo

# Step 3: Create the batch ASR python script
echo "[STEP 3/4] Creating batch transcription script (batch_english_asr.py)..."
cat > "$PROJECT_DIR/batch_english_asr.py" <<'EOF'
import os
import argparse
import glob
from pathlib import Path
import torch
from kimia_infer.api.kimia import KimiAudio

def run_batch_transcription(source_dir, model_path="moonshotai/Kimi-Audio-7B-Instruct"):
    """
    Traverses a directory, transcribes all .wav files using Kimi-Audio,
    and saves the transcriptions in the same folder.
    """
    print("=" * 80)
    print(f"Kimi-Audio English ASR Batch Transcription Tool")
    print(f"Source Directory: {source_dir}")
    print(f"Using Model: {model_path}")
    print("=" * 80)
    print()

    # Check for GPU
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name()}")
    else:
        print("⚠ No GPU detected, using CPU (processing will be slower).")
        print("  Note: Kimi-Audio is computationally intensive and a GPU is highly recommended.")
    print()
    
    # --- 1. Load Model ---
    print("Loading Kimi-Audio model... (This may take a moment)")
    try:
        model = KimiAudio(
            model_path=model_path,
            load_detokenizer=True, # Detokenizer not strictly needed for ASR
        )
        print("✓ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Please ensure you have a stable internet connection for the first run,")
        print("   and that the model path is correct.")
        return
        
    print()

    # --- 2. Define Sampling Parameters ---
    sampling_params = {
        "text_temperature": 0.0,
        "text_top_k": 1,
    }

    # --- 3. Find and Process Audio Files ---
    print("Starting transcription process...")
    
    wav_files = glob.glob(os.path.join(source_dir, '**', '*.wav'), recursive=True)
    
    if not wav_files:
        print(f"No .wav files found in {source_dir}. Please check the directory.")
        return

    print(f"Found {len(wav_files)} .wav files to process.")
    print()

    for audio_path in wav_files:
        output_txt_path = Path(audio_path).with_suffix('.txt')

        if os.path.exists(output_txt_path):
            print(f"⏭️ Skipping (already exists): {output_txt_path}")
            continue
            
        print(f"🎤 Processing: {audio_path}")
        
        try:
            messages = [
                {"role": "user", "message_type": "text", "content": "This is a speech-to-text task, please transcribe the content of the audio into text."},
                {"role": "user", "message_type": "audio", "content": audio_path},
            ]

            _, text_output = model.generate(messages, **sampling_params, output_type="text")
            
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(text_output)
            
            print(f"   📄 Saved transcription to: {output_txt_path}")

        except Exception as e:
            print(f"   ❌ Error processing {audio_path}: {e}")
            error_file_path = Path(audio_path).with_suffix('.error.txt')
            with open(error_file_path, 'w', encoding='utf-8') as f:
                f.write(str(e))
            print(f"   📄 Saved error details to: {error_file_path}")

        print("-" * 40)

    print("\nTranscription process finished!")
    print("Results have been saved in their respective source folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch English ASR using Kimi-Audio.")
    parser.add_argument("source_dir", type=str, help="Directory containing sub-folders of .wav files.")
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct", help="Path or Hugging Face name of the Kimi-Audio model.")
    
    args = parser.parse_args()
    
    run_batch_transcription(args.source_dir, args.model_path)
EOF
echo "✓ Python script created successfully."
echo

# Step 4: Build the Docker image
echo "[STEP 4/4] Building the Docker image '$DOCKER_IMAGE_NAME'..."
echo "This may take a while, please be patient."
docker build -t "$DOCKER_IMAGE_NAME" "$PROJECT_DIR"
echo "✓ Docker image built successfully!"
echo

# Final instructions
echo "================================================="
echo "🎉 SETUP COMPLETE! 🎉"
echo "================================================="
echo
echo "You can now run the transcription process using the command below."
echo "It will mount all necessary directories into the container."
echo
echo "---"
echo "👇 To start the container, run this command:"
echo "---"
echo
echo "docker run --gpus all -it --rm \\
  -v \"$CACHE_DIR\":/root/.cache/huggingface \\
  -v \"$PROJECT_DIR\":/app \\
  -v \"$DATASET_DIR\":/data \\
  $DOCKER_IMAGE_NAME"
echo
echo "---"
echo "🚀 Once inside the container, run this command to start transcribing:"
echo "---"
echo
echo "python /app/batch_english_asr.py /data"
echo 