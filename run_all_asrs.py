import os
import glob
import torch
import argparse
from tqdm import tqdm
import warnings

# python3 ems_call/run_all_asrs.py /media/meow/One\ Touch/ems_call/random_samples_1_preprocessed/

# --- Suppress Warnings ---
# Suppress specific warnings from libraries to keep the output clean.
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')

# --- Model & Framework Imports ---
# We import everything here and handle potential import errors.
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

# --- Model Configuration ---
# Maps model prefixes to their local paths and the framework required to run them.
MODELS = {
    'wav2vec-xls-r': {
        'path': 'facebook/wav2vec2-base-960h', # Changed to a fine-tuned version
        'framework': 'transformers'
    },
    'canary-1b': {
        'path': 'nvidia/canary-1b', # Changed from local path to Hub ID
        'framework': 'nemo'
    },
    'parakeet-tdt-0.6b-v2': {
        'path': 'nvidia/parakeet-ctc-0.6b', # Changed from local path to Hub ID
        'framework': 'nemo'
    },
    'large-v3': {
        'path': 'large-v3', # Whisper uses model name, not path
        'framework': 'whisper'
    }
}

# --- Helper Functions ---

def get_device():
    """Checks for CUDA availability."""
    if torch.cuda.is_available():
        print("CUDA detected. Using GPU for transcription.")
        return "cuda"
    else:
        print("CUDA not detected. Using CPU (this will be very slow).")
        return "cpu"

def check_dependencies():
    """Checks if all required frameworks are installed."""
    print("Checking required frameworks...")
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: 'transformers' or 'torchaudio' not found. Please run 'pip install transformers torch torchaudio'.")
        return False
    if not NEMO_AVAILABLE:
        print("ERROR: 'nemo_toolkit' not found. Please run 'pip install nemo_toolkit[asr]'.")
        return False
    if not WHISPER_AVAILABLE:
        print("ERROR: 'openai-whisper' not found. Please run 'pip install openai-whisper'.")
        return False
    print("All required frameworks are available.")
    return True

# --- Transcription Functions by Framework ---

def transcribe_with_transformers(model_path, wav_files, device, model_prefix):
    """Handles transcription for models using the Transformers library (e.g., Wav2Vec2)."""
    print(f"Loading Transformers model from: {model_path}")
    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
        processor = Wav2Vec2Processor.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading Transformers model: {e}")
        return

    for audio_path in tqdm(wav_files, desc=f"Model: {model_prefix}"):
        output_path = os.path.join(os.path.dirname(audio_path), f"{model_prefix}_{os.path.basename(audio_path).replace('.wav', '.txt')}")
        if os.path.exists(output_path):
            continue
        try:
            speech_array, sampling_rate = torchaudio.load(audio_path)
            if sampling_rate != 16000:
                resampler = T.Resample(orig_freq=sampling_rate, new_freq=16000)
                speech_array = resampler(speech_array)
            
            inputs = processor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription.strip())
        except Exception as e:
            print(f"Error transcribing {os.path.basename(audio_path)}: {e}")
    del model, processor
    if "cuda" in device: torch.cuda.empty_cache()

def transcribe_with_nemo(model_path, wav_files, device, model_prefix):
    """Handles transcription for NeMo models (e.g., Canary, Parakeet)."""
    print(f"Loading NeMo model from: {model_path}")
    try:
        # Note: NeMo models often need the .nemo file path directly.
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_path, map_location=device)
        asr_model.to(device)
    except Exception as e:
        print(f"Error loading NeMo model: {e}")
        return

    # Process files in batches for efficiency
    batch_size = 16
    for i in tqdm(range(0, len(wav_files), batch_size), desc=f"Model: {model_prefix}"):
        batch_paths = wav_files[i:i + batch_size]
        
        # Filter out files that already have a transcript
        unprocessed_paths = []
        for path in batch_paths:
            output_path = os.path.join(os.path.dirname(path), f"{model_prefix}_{os.path.basename(path).replace('.wav', '.txt')}")
            if not os.path.exists(output_path):
                unprocessed_paths.append(path)
        
        if not unprocessed_paths:
            continue

        try:
            # Force NeMo to return Hypothesis objects for consistent output processing,
            # which aligns with the provided example scripts.
            transcriptions = asr_model.transcribe(
                audio=unprocessed_paths,
                batch_size=batch_size,
                return_hypotheses=True  # Ensure we get Hypothesis objects
            )

            # The result is now guaranteed to be a list of Hypothesis objects.
            if transcriptions:
                for path, hyp in zip(unprocessed_paths, transcriptions):
                    # Extract the text from the hypothesis object before writing.
                    text_result = hyp.text if hasattr(hyp, 'text') else ""
                    output_path = os.path.join(os.path.dirname(path), f"{model_prefix}_{os.path.basename(path).replace('.wav', '.txt')}")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text_result.strip())
        except Exception as e:
            print(f"Error during batch transcription: {e}")
    del asr_model
    if "cuda" in device: torch.cuda.empty_cache()


def transcribe_with_whisper(model_name, wav_files, device, model_prefix):
    """Handles transcription using the openai-whisper library."""
    print(f"Loading Whisper model: {model_name}")
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return

    for audio_path in tqdm(wav_files, desc=f"Model: {model_prefix}"):
        output_path = os.path.join(os.path.dirname(audio_path), f"{model_prefix}_{os.path.basename(audio_path).replace('.wav', '.txt')}")
        if os.path.exists(output_path):
            continue
        try:
            result = model.transcribe(audio_path, language="en", fp16=("cuda" in device))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result["text"].strip())
        except Exception as e:
            print(f"Error transcribing {os.path.basename(audio_path)}: {e}")
    del model
    if "cuda" in device: torch.cuda.empty_cache()

# --- Main Execution ---

def main(wav_folder, selected_models=None):
    """Main function to orchestrate the transcription process.
    
    Args:
        wav_folder (str): Path to folder containing WAV files
        selected_models (list): List of specific models to run. If None, run all models.
    """
    if not check_dependencies():
        return

    print(f"\nTarget audio folder: {wav_folder}")
    if not os.path.isdir(wav_folder):
        print(f"ERROR: Folder not found: {wav_folder}")
        return

    wav_files = glob.glob(os.path.join(wav_folder, '*.wav'))
    if not wav_files:
        print(f"No .wav files found in {wav_folder}.")
        return
    print(f"Found {len(wav_files)} .wav files to process.")

    device = get_device()

    # Determine which models to run
    if selected_models:
        models_to_run = {k: v for k, v in MODELS.items() if k in selected_models}
        print(f"Running selected models: {', '.join(selected_models)}")
    else:
        models_to_run = MODELS
        print("Running all available models")

    for model_prefix, config in models_to_run.items():
        print(f"\n--- Starting transcription for model: {model_prefix} ---")
        model_path = config['path']
        framework = config['framework']
        
        if framework == 'transformers':
            # Transformers can load from Hub ID directly
            transcribe_with_transformers(model_path, wav_files, device, model_prefix)
        
        elif framework == 'nemo':
            # NeMo can load from Hub ID directly
            transcribe_with_nemo(model_path, wav_files, device, model_prefix)

        elif framework == 'whisper':
            transcribe_with_whisper(model_path, wav_files, device, model_prefix)

        else:
            print(f"WARNING: Unknown framework '{framework}' for model {model_prefix}. Skipping.")
            
        print(f"--- Finished transcription for model: {model_prefix} ---")

    print(f"\nProcessed {len(models_to_run)} model(s).")
    print(f"Transcription .txt files saved in: {wav_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run multiple local ASR models on a folder of WAV files.")
    parser.add_argument("wav_folder", type=str, help="Path to the folder containing .wav files to process.")
    parser.add_argument("--models", type=str, nargs='+', 
                       choices=list(MODELS.keys()), 
                       help="Select specific ASR models to run. Available models: " + ", ".join(MODELS.keys()))
    args = parser.parse_args()
    main(args.wav_folder, selected_models=args.models) 