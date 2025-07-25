#!/usr/bin/env python3

import os
import shutil
import tempfile

def create_test_scenario():
    """Create a test scenario with missing files for some models."""
    
    # Create temporary directory structure
    test_dir = tempfile.mkdtemp()
    print(f"Created test directory: {test_dir}")
    
    # Create model subdirectories
    models = ['large-v3', 'wav2vec-xls-r', 'parakeet-tdt-0.6b-v2', 'canary-1b']
    for model in models:
        os.makedirs(os.path.join(test_dir, model), exist_ok=True)
    
    # Create test files - simulate missing files for some models
    test_files = {
        'large-v3': ['large-v3_long_audio_group_001.txt', 'large-v3_long_audio_group_002.txt', 'large-v3_long_audio_group_003.txt'],
        'wav2vec-xls-r': ['wav2vec-xls-r_long_audio_group_001.txt', 'wav2vec-xls-r_long_audio_group_002.txt'],  # Missing one file
        'parakeet-tdt-0.6b-v2': ['parakeet-tdt-0.6b-v2_long_audio_group_001.txt'],  # Missing two files
        'canary-1b': ['canary-1b_long_audio_group_001.txt', 'canary-1b_long_audio_group_002.txt', 'canary-1b_long_audio_group_003.txt']
    }
    
    # Create files with some content
    for model, files in test_files.items():
        for file in files:
            file_path = os.path.join(test_dir, model, file)
            with open(file_path, 'w') as f:
                f.write(f"Test content for {file}")
    
    # Create an empty file to test empty file detection
    empty_file = os.path.join(test_dir, 'large-v3', 'large-v3_long_audio_group_004.txt')
    with open(empty_file, 'w') as f:
        pass  # Empty file
    
    print("Test scenario created:")
    for model, files in test_files.items():
        print(f"  {model}: {len(files)} files")
    
    return test_dir

if __name__ == '__main__':
    test_dir = create_test_scenario()
    print(f"\nTest directory: {test_dir}")
    print("You can now run analyze_model_files.py on this directory to test missing file detection.") 