#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage of VAD Pipeline
=============================

This script demonstrates how to use the new VAD (Voice Activity Detection)
pipeline with your existing ASR workflow.
"""

import os
from pathlib import Path
import json

def example_vad_only():
    """Example: Run VAD only to extract speech segments"""
    print("=== Example 1: VAD Only ===")
    
    # Configure paths
    input_dir = "/media/meow/One Touch/ems_call/random_samples_1_preprocessed"
    output_dir = "/media/meow/One Touch/ems_call/vad_output_example"
    
    # Command to run VAD only
    cmd = f"""
    python3 ems_call/vad_pipeline.py \\
        --input_dir "{input_dir}" \\
        --output_dir "{output_dir}" \\
        --speech_threshold 0.6 \\
        --min_speech_duration 1.0
    """
    
    print("Command to run:")
    print(cmd)
    print()
    
    # Expected output structure
    print("Expected output structure:")
    print("vad_output_example/")
    print("├── audio_file_1/")
    print("│   ├── segment_001.wav")
    print("│   ├── segment_002.wav")
    print("│   └── audio_file_1_vad_metadata.json")
    print("├── audio_file_2/")
    print("│   └── ...")
    print("└── vad_processing_summary.json")
    print()


def example_vad_plus_asr():
    """Example: Run complete VAD + ASR pipeline"""
    print("=== Example 2: VAD + ASR Pipeline ===")
    
    # Configure paths
    input_dir = "/media/meow/One Touch/ems_call/random_samples_1_preprocessed"
    output_dir = "/media/meow/One Touch/ems_call/vad_asr_output_example"
    
    # Command to run complete pipeline
    cmd = f"""
    python3 ems_call/run_vad_asr_pipeline.py \\
        --input_dir "{input_dir}" \\
        --output_dir "{output_dir}" \\
        --models large-v3 canary-1b \\
        --speech_threshold 0.5 \\
        --min_speech_duration 0.5
    """
    
    print("Command to run:")
    print(cmd)
    print()
    
    # Expected output structure
    print("Expected output structure:")
    print("vad_asr_output_example/")
    print("├── vad_segments/          # VAD extracted segments")
    print("│   ├── audio_file_1/")
    print("│   │   ├── segment_001.wav")
    print("│   │   └── segment_002.wav")
    print("│   └── vad_processing_summary.json")
    print("├── transcripts/           # ASR transcripts for segments")
    print("│   ├── large-v3_audio_file_1_segment_001.txt")
    print("│   ├── canary-1b_audio_file_1_segment_001.txt")
    print("│   └── ...")
    print("└── final_results/         # Consolidated transcripts")
    print("    ├── audio_file_1/")
    print("    │   ├── large-v3_audio_file_1.txt")
    print("    │   └── canary-1b_audio_file_1.txt")
    print("    └── processing_summary.json")
    print()


def example_skip_vad():
    """Example: Run ASR without VAD (original workflow)"""
    print("=== Example 3: ASR Without VAD (Original Workflow) ===")
    
    # Configure paths
    input_dir = "/media/meow/One Touch/ems_call/random_samples_1_preprocessed"
    output_dir = "/media/meow/One Touch/ems_call/asr_only_output_example"
    
    # Command to run ASR without VAD
    cmd = f"""
    python3 ems_call/run_vad_asr_pipeline.py \\
        --input_dir "{input_dir}" \\
        --output_dir "{output_dir}" \\
        --skip_vad \\
        --models large-v3 wav2vec-xls-r
    """
    
    print("Command to run:")
    print(cmd)
    print()
    
    print("This processes original files directly without VAD preprocessing.")
    print()


def example_shell_script():
    """Example: Using the enhanced shell script"""
    print("=== Example 4: Using Enhanced Shell Script ===")
    
    # Configure paths
    input_dir = "/media/meow/One Touch/ems_call/random_samples_1_preprocessed"
    output_dir = "/media/meow/One Touch/ems_call/pipeline_output_example"
    ground_truth = "/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv"
    
    print("Command options:")
    
    # Full pipeline with VAD
    cmd1 = f"""
    bash ems_call/run_vad_pipeline.sh \\
        --input_dir "{input_dir}" \\
        --output_dir "{output_dir}" \\
        --ground_truth "{ground_truth}" \\
        --models "large-v3 canary-1b" \\
        --speech-threshold 0.6
    """
    print("1. Full pipeline with VAD:")
    print(cmd1)
    print()
    
    # Pipeline without VAD
    cmd2 = f"""
    bash ems_call/run_vad_pipeline.sh \\
        --input_dir "{input_dir}" \\
        --output_dir "{output_dir}" \\
        --no-vad \\
        --models "large-v3"
    """
    print("2. Pipeline without VAD:")
    print(cmd2)
    print()
    
    # VAD only
    cmd3 = f"""
    bash ems_call/run_vad_pipeline.sh \\
        --input_dir "{input_dir}" \\
        --output_dir "{output_dir}" \\
        --skip-asr \\
        --skip-evaluation
    """
    print("3. VAD processing only:")
    print(cmd3)
    print()


def show_configuration_options():
    """Show available configuration options"""
    print("=== Configuration Options ===")
    
    print("VAD Parameters:")
    print("  --speech_threshold FLOAT    Speech detection threshold (0.0-1.0, default: 0.5)")
    print("  --min_speech_duration FLOAT Minimum speech segment duration (seconds, default: 0.5)")
    print("  --min_silence_duration FLOAT Minimum silence to split segments (seconds, default: 0.3)")
    print("  --chunk_size INT            VAD processing chunk size (default: 512)")
    print()
    
    print("ASR Models (select one or more):")
    print("  large-v3                    Whisper Large v3 (best accuracy)")
    print("  canary-1b                   NVIDIA Canary 1B")
    print("  parakeet-tdt-0.6b-v2       NVIDIA Parakeet TDT 0.6B")
    print("  wav2vec-xls-r               Facebook Wav2Vec2 XLS-R")
    print()
    
    print("Processing Options:")
    print("  --skip_vad                  Skip VAD preprocessing")
    print("  --skip-asr                  Skip ASR transcription")
    print("  --skip-evaluation           Skip accuracy evaluation")
    print()


def performance_comparison():
    """Show expected performance improvements with VAD"""
    print("=== Expected Performance Benefits with VAD ===")
    print()
    print("1. Processing Speed:")
    print("   - Without VAD: Process entire audio files")
    print("   - With VAD: Process only speech segments (~30-70% of total audio)")
    print("   - Expected speedup: 1.5-3x faster")
    print()
    
    print("2. Accuracy:")
    print("   - Reduced background noise and silence")
    print("   - Better focus on actual speech content")
    print("   - Potentially improved WER (Word Error Rate)")
    print()
    
    print("3. Storage:")
    print("   - Intermediate files contain only speech segments")
    print("   - Reduced storage requirements for processed audio")
    print()
    
    print("4. Analysis:")
    print("   - VAD provides speech/silence ratio statistics")
    print("   - Segment-level metadata for detailed analysis")
    print("   - Time-stamped speech segments")
    print()


def main():
    """Main function to display all examples"""
    print("VAD Pipeline Examples")
    print("====================")
    print()
    
    example_vad_only()
    example_vad_plus_asr()
    example_skip_vad()
    example_shell_script()
    show_configuration_options()
    performance_comparison()
    
    print("=== Getting Started ===")
    print()
    print("1. Make sure you have all dependencies installed:")
    print("   pip install torch torchaudio transformers nemo_toolkit[asr] openai-whisper")
    print()
    print("2. Try VAD only first to see speech segments:")
    print(f"   python3 ems_call/vad_pipeline.py --input_dir YOUR_INPUT_DIR --output_dir /tmp/vad_test")
    print()
    print("3. Run the complete pipeline:")
    print(f"   bash ems_call/run_vad_pipeline.sh -i YOUR_INPUT_DIR -o YOUR_OUTPUT_DIR")
    print()
    print("4. Check results in the output directory and summary files")
    print()


if __name__ == '__main__':
    main() 