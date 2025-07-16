# -*- coding: utf-8 -*-
import os
import whisper

# List of model names
# need 11G VRAM
model_names = [
    "tiny",
    "medium",
    "large-v2", #out of mem on 3080 (10GB)
    "large-v3", 
    "large-v3-turbo"
]

# Path to the input directory containing audio files
# input_directory = "/media/meow/Elements/ems_call/data/data_cliped_from_2024all_filtered"  # Replace with your input directory path
# input_directory = "/media/meow/Elements/ems_call/data/data_2024all_n3_keep10s"  # Replace with your input directory path
input_directory = "/media/meow/One Touch/ems_call/test_data_30"  # Replace with your input directory path


# Base output directory
# base_output_dir = "/media/meow/Elements/ems_call/data/20250123_all_transcriptions"
base_output_dir = "/media/meow/One Touch/ems_call/test_data_30/all_30_transcriptions"


# Ensure the base output directory exists
os.makedirs(base_output_dir, exist_ok=True)

# Process each model
for model_name in model_names:
    print(f"Processing with model: {model_name}")

    # Load the model
    model = whisper.load_model(model_name)

    # Create a directory for the model's output
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Process each audio file in the input directory
    for audio_file in os.listdir(input_directory):
        if audio_file.endswith(('.wav', '.mp3', '.m4a')):  # Add other audio formats if needed
            audio_path = os.path.join(input_directory, audio_file)
            print(f"Transcribing file: {audio_path}")

            # Transcribe the audio
            result = model.transcribe(audio_path)

            # Define the output file path
            output_file = os.path.join(model_output_dir, f"{os.path.splitext(audio_file)[0]}.txt")

            # Save the transcription
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])

            print(f"Transcription saved to: {output_file}")
