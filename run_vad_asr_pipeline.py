#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated VAD + ASR Pipeline
=============================

This script combines Voice Activity Detection (VAD) preprocessing with ASR transcription.
It first extracts speech segments from audio files, then runs multiple ASR models on
only the speech segments, improving efficiency and potentially accuracy.

Usage:
    python3 ems_call/run_vad_asr_pipeline.py --input_dir /path/to/audio --output_dir /path/to/output
    python3 ems_call/run_vad_asr_pipeline.py --input_dir /path/to/audio --output_dir /path/to/output --skip_vad
"""

import os
import sys
import glob
import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import warnings

# Add the current directory to Python path to import local modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import VAD pipeline
from vad_pipeline import VADPipeline

# Import ASR components (from run_all_asrs.py)
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torchaudio
    import torchaudio.transforms as T
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')

# ASR Model Configuration
MODELS = {
    'wav2vec-xls-r': {
        'path': 'facebook/wav2vec2-base-960h',
        'framework': 'transformers'
    },
    'canary-1b': {
        'path': 'nvidia/canary-1b',
        'framework': 'nemo'
    },
    'parakeet-tdt-0.6b-v2': {
        'path': 'nvidia/parakeet-ctc-0.6b',
        'framework': 'nemo'
    },
    'large-v3': {
        'path': 'large-v3',
        'framework': 'whisper'
    }
}

class VADASRPipeline:
    """Integrated VAD + ASR Pipeline"""
    
    def __init__(self, 
                 vad_params: dict = None, 
                 asr_models: list = None,
                 device: str = None):
        """
        Initialize the integrated pipeline
        
        Args:
            vad_params: VAD configuration parameters
            asr_models: List of ASR models to use (default: all available)
            device: Device to use for processing
        """
        # Set device
        self.device = device or self.get_device()
        
        # Initialize VAD pipeline
        vad_params = vad_params or {}
        self.vad_pipeline = VADPipeline(**vad_params)
        
        # Set ASR models
        self.asr_models = asr_models or list(MODELS.keys())
        
        # Check dependencies
        self.check_dependencies()
        
        print(f"VAD+ASR Pipeline initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - ASR Models: {', '.join(self.asr_models)}")
    
    def get_device(self):
        """Check for CUDA availability"""
        if torch.cuda.is_available():
            print("CUDA detected. Using GPU for processing.")
            return "cuda"
        else:
            print("CUDA not detected. Using CPU (this will be slower).")
            return "cpu"
    
    def check_dependencies(self):
        """Check if required frameworks are available"""
        missing = []
        
        for model in self.asr_models:
            framework = MODELS[model]['framework']
            if framework == 'transformers' and not TRANSFORMERS_AVAILABLE:
                missing.append('transformers')
            elif framework == 'nemo' and not NEMO_AVAILABLE:
                missing.append('nemo_toolkit')
            elif framework == 'whisper' and not WHISPER_AVAILABLE:
                missing.append('openai-whisper')
        
        if missing:
            print(f"ERROR: Missing dependencies: {', '.join(set(missing))}")
            print("Please install required packages:")
            for dep in set(missing):
                if dep == 'transformers':
                    print("  pip install transformers torch torchaudio")
                elif dep == 'nemo_toolkit':
                    print("  pip install nemo_toolkit[asr]")
                elif dep == 'openai-whisper':
                    print("  pip install openai-whisper")
            return False
        
        print("All required dependencies are available.")
        return True
    
    def process_with_vad_and_asr(self, input_dir: str, output_dir: str, skip_vad: bool = False):
        """
        Complete pipeline: VAD preprocessing + ASR transcription
        
        Args:
            input_dir: Directory containing input audio files
            output_dir: Output directory for results
            skip_vad: If True, skip VAD and process original files directly
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not skip_vad:
            # Step 1: VAD Preprocessing
            print("=== Step 1: VAD Preprocessing ===")
            vad_output_dir = output_path / "vad_segments"
            vad_summary = self.vad_pipeline.process_directory(str(input_path), str(vad_output_dir))
            
            if 'error' in vad_summary:
                print(f"VAD processing failed: {vad_summary['error']}")
                return
            
            print(f"VAD completed: {vad_summary['successful']} files processed")
            
            # Step 2: Collect VAD segments for ASR
            print("\n=== Step 2: Collecting VAD segments for ASR ===")
            segment_files = list(vad_output_dir.rglob("segment_*.wav"))
        else:
            # Skip VAD - use original files
            print("=== Skipping VAD - Processing original files ===")
            segment_files = []
            for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                segment_files.extend(input_path.rglob(f'*{ext}'))
                segment_files.extend(input_path.rglob(f'*{ext.upper()}'))
        
        if not segment_files:
            print("No audio files found for ASR processing")
            return
        
        print(f"Found {len(segment_files)} audio files for ASR processing")
        
        # Step 3: ASR Transcription
        print(f"\n=== Step 3: ASR Transcription ===")
        asr_output_dir = output_path / "transcripts"
        asr_output_dir.mkdir(exist_ok=True)
        
        # Copy files to ASR directory for processing
        import shutil
        asr_input_dir = output_path / "asr_input"
        asr_input_dir.mkdir(exist_ok=True)
        
        for i, segment_file in enumerate(tqdm(segment_files, desc="Preparing files for ASR")):
            # Create a simplified filename for ASR processing
            if skip_vad:
                new_name = f"original_{i:04d}_{segment_file.stem}.wav"
            else:
                # Extract original file info and segment info
                parent_name = segment_file.parent.name
                segment_name = segment_file.stem
                new_name = f"{parent_name}_{segment_name}.wav"
            
            target_path = asr_input_dir / new_name
            shutil.copy2(segment_file, target_path)
        
        # Run ASR on all models
        self.run_asr_transcription(str(asr_input_dir), str(asr_output_dir))
        
        # Step 4: Consolidate results
        print(f"\n=== Step 4: Consolidating results ===")
        self.consolidate_results(str(asr_input_dir), str(asr_output_dir), str(output_path), skip_vad)
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Results saved to: {output_path}")
        if not skip_vad:
            print(f"  - VAD segments: {vad_output_dir}")
        print(f"  - ASR transcripts: {asr_output_dir}")
        print(f"  - Final results: {output_path / 'final_results'}")
    
    def run_asr_transcription(self, input_dir: str, output_dir: str):
        """Run ASR transcription on audio files"""
        wav_files = sorted(glob.glob(os.path.join(input_dir, '*.wav')))
        
        if not wav_files:
            print("No WAV files found for ASR processing")
            return
        
        for model_prefix in self.asr_models:
            if model_prefix not in MODELS:
                print(f"Warning: Unknown model {model_prefix}, skipping")
                continue
                
            config = MODELS[model_prefix]
            model_path = config['path']
            framework = config['framework']
            
            print(f"\n--- Processing with {model_prefix} ({framework}) ---")
            
            try:
                if framework == 'transformers':
                    self.transcribe_with_transformers(model_path, wav_files, output_dir, model_prefix)
                elif framework == 'nemo':
                    self.transcribe_with_nemo(model_path, wav_files, output_dir, model_prefix)
                elif framework == 'whisper':
                    self.transcribe_with_whisper(model_path, wav_files, output_dir, model_prefix)
            except Exception as e:
                print(f"Error with model {model_prefix}: {e}")
                continue
    
    def transcribe_with_transformers(self, model_path, wav_files, output_dir, model_prefix):
        """Transcribe using Transformers models"""
        print(f"Loading Transformers model: {model_path}")
        try:
            model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
            processor = Wav2Vec2Processor.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        for audio_path in tqdm(wav_files, desc=f"Transcribing with {model_prefix}"):
            output_path = os.path.join(output_dir, f"{model_prefix}_{os.path.basename(audio_path).replace('.wav', '.txt')}")
            
            if os.path.exists(output_path):
                continue
                
            try:
                speech_array, sampling_rate = torchaudio.load(audio_path)
                if sampling_rate != 16000:
                    resampler = T.Resample(orig_freq=sampling_rate, new_freq=16000)
                    speech_array = resampler(speech_array)
                
                inputs = processor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = model(inputs.input_values.to(self.device)).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcription.strip())
            except Exception as e:
                print(f"Error transcribing {os.path.basename(audio_path)}: {e}")
        
        del model, processor
        if "cuda" in self.device:
            torch.cuda.empty_cache()

    def transcribe_with_nemo(self, model_path, wav_files, output_dir, model_prefix):
        """Transcribe using NeMo models"""
        print(f"Loading NeMo model: {model_path}")
        try:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model_path, map_location=self.device)
            asr_model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        batch_size = 16
        for i in tqdm(range(0, len(wav_files), batch_size), desc=f"Transcribing with {model_prefix}"):
            batch_paths = wav_files[i:i + batch_size]
            unprocessed_paths = []
            
            for path in batch_paths:
                output_path = os.path.join(output_dir, f"{model_prefix}_{os.path.basename(path).replace('.wav', '.txt')}")
                if not os.path.exists(output_path):
                    unprocessed_paths.append(path)
            
            if not unprocessed_paths:
                continue

            try:
                transcriptions = asr_model.transcribe(
                    audio=unprocessed_paths,
                    batch_size=batch_size,
                    return_hypotheses=True
                )

                if transcriptions:
                    for path, hyp in zip(unprocessed_paths, transcriptions):
                        text_result = hyp.text if hasattr(hyp, 'text') else ""
                        output_path = os.path.join(output_dir, f"{model_prefix}_{os.path.basename(path).replace('.wav', '.txt')}")
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(text_result.strip())
            except Exception as e:
                print(f"Error during batch transcription: {e}")
        
        del asr_model
        if "cuda" in self.device:
            torch.cuda.empty_cache()

    def transcribe_with_whisper(self, model_name, wav_files, output_dir, model_prefix):
        """Transcribe using Whisper models"""
        print(f"Loading Whisper model: {model_name}")
        try:
            model = whisper.load_model(model_name, device=self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        for audio_path in tqdm(wav_files, desc=f"Transcribing with {model_prefix}"):
            output_path = os.path.join(output_dir, f"{model_prefix}_{os.path.basename(audio_path).replace('.wav', '.txt')}")
            
            if os.path.exists(output_path):
                continue
                
            try:
                result = model.transcribe(audio_path, language="en", fp16=("cuda" in self.device))
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result["text"].strip())
            except Exception as e:
                print(f"Error transcribing {os.path.basename(audio_path)}: {e}")
        
        del model
        if "cuda" in self.device:
            torch.cuda.empty_cache()
    
    def consolidate_results(self, asr_input_dir: str, asr_output_dir: str, final_output_dir: str, skip_vad: bool):
        """Consolidate transcription results by original file"""
        final_results_dir = Path(final_output_dir) / "final_results"
        final_results_dir.mkdir(exist_ok=True)
        
        # Group transcriptions by original file
        transcript_files = list(Path(asr_output_dir).glob("*.txt"))
        
        # Organize by original file and model
        file_groups = {}
        
        for transcript_file in transcript_files:
            # Parse filename: {model}_{original_file}_{segment_info}.txt
            parts = transcript_file.stem.split('_', 2)
            if len(parts) >= 2:
                model = parts[0]
                # Remove model prefix to get original info
                remaining = '_'.join(parts[1:])
                
                if skip_vad:
                    # For original files: original_{index}_{filename}
                    if remaining.startswith('original_'):
                        original_file = '_'.join(remaining.split('_')[2:])  # Remove 'original_{index}'
                    else:
                        original_file = remaining
                else:
                    # For VAD segments: {original_file}_segment_{number}
                    if '_segment_' in remaining:
                        original_file = remaining.split('_segment_')[0]
                    else:
                        original_file = remaining
                
                if original_file not in file_groups:
                    file_groups[original_file] = {}
                if model not in file_groups[original_file]:
                    file_groups[original_file][model] = []
                
                # Read transcript content
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                file_groups[original_file][model].append(content)
        
        # Create consolidated transcripts for each original file
        for original_file, models in file_groups.items():
            file_result_dir = final_results_dir / original_file
            file_result_dir.mkdir(exist_ok=True)
            
            for model, segments in models.items():
                # Combine all segments for this model
                if skip_vad:
                    # For original files, just use the single transcript
                    combined_text = segments[0] if segments else ""
                else:
                    # For VAD segments, combine all segments
                    combined_text = " ".join(segments)
                
                # Save combined transcript
                output_file = file_result_dir / f"{model}_{original_file}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
        
        print(f"Consolidated results for {len(file_groups)} original files")
        
        # Create summary
        summary = {
            'processing_mode': 'original_files' if skip_vad else 'vad_segments',
            'original_files_processed': len(file_groups),
            'models_used': list(self.asr_models),
            'files': {}
        }
        
        for original_file, models in file_groups.items():
            summary['files'][original_file] = {
                'models': list(models.keys()),
                'segment_counts': {model: len(segments) for model, segments in models.items()}
            }
        
        # Save summary
        summary_file = final_results_dir / 'processing_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Processing summary saved to: {summary_file}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Integrated VAD + ASR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with VAD preprocessing
    python3 ems_call/run_vad_asr_pipeline.py --input_dir /path/to/audio --output_dir /path/to/output
    
    # Skip VAD and process original files
    python3 ems_call/run_vad_asr_pipeline.py --input_dir /path/to/audio --output_dir /path/to/output --skip_vad
    
    # Custom VAD parameters
    python3 ems_call/run_vad_asr_pipeline.py --input_dir /path/to/audio --output_dir /path/to/output \\
        --speech_threshold 0.7 --min_speech_duration 1.0
    
    # Select specific ASR models
    python3 ems_call/run_vad_asr_pipeline.py --input_dir /path/to/audio --output_dir /path/to/output \\
        --models large-v3 canary-1b
        """
    )
    
    # Input/Output
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    
    # Processing options
    parser.add_argument('--skip_vad', action='store_true', help='Skip VAD preprocessing and process original files')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()), default=list(MODELS.keys()),
                       help='ASR models to use (default: all available)')
    
    # VAD Parameters
    parser.add_argument('--speech_threshold', type=float, default=0.5, help='VAD speech threshold (default: 0.5)')
    parser.add_argument('--min_speech_duration', type=float, default=0.5, help='Minimum speech duration (default: 0.5s)')
    parser.add_argument('--min_silence_duration', type=float, default=0.3, help='Minimum silence duration (default: 0.3s)')
    parser.add_argument('--chunk_size', type=int, default=512, help='VAD chunk size (default: 512)')
    
    args = parser.parse_args()
    
    # Prepare VAD parameters
    vad_params = {
        'speech_threshold': args.speech_threshold,
        'min_speech_duration': args.min_speech_duration,
        'min_silence_duration': args.min_silence_duration,
        'chunk_size': args.chunk_size,
    }
    
    # Initialize pipeline
    pipeline = VADASRPipeline(
        vad_params=vad_params,
        asr_models=args.models
    )
    
    # Run pipeline
    pipeline.process_with_vad_and_asr(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        skip_vad=args.skip_vad
    )


if __name__ == '__main__':
    main() 