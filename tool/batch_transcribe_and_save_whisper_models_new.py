# -*- coding: utf-8 -*-
import os
import whisper

# List of model names
model_names = [
    "tiny",
    "medium",
    "large-v2",
    "large-v3",
    "large-v3-turbo"
]

# Path configurations
input_directory = "/media/meow/One Touch/ems_call/test_data_30"
base_output_dir = "/media/meow/One Touch/ems_call/test_data_30/all_30_transcriptions"

# Get all immediate subdirectories
subdir_names = next(os.walk(input_directory))[1]

# Process each model
for model_name in model_names:
    print(f"\n{'='*40}")
    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name)
    
    # Process each subdirectory
    for subdir_name in subdir_names:
        subdir_path = os.path.join(input_directory, subdir_name)
        output_dir = os.path.join(base_output_dir, subdir_name, model_name)
        
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing subdirectory: {subdir_name}")
        print(f"Output directory: {output_dir}")

        # Process each audio file in the subdirectory
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.wav', '.mp3', '.m4a')):
                audio_path = os.path.join(subdir_path, filename)
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_path = os.path.join(output_dir, txt_filename)

                # Skip if transcription already exists
                if os.path.exists(txt_path):
                    print(f"Skipping existing transcription: {txt_path}")
                    continue

                print(f"Transcribing: {filename}")
                try:
                    result = model.transcribe(audio_path)
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(result['text'])
                    print(f"Saved transcription to: {txt_path}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

print("\nAll transcriptions completed!")