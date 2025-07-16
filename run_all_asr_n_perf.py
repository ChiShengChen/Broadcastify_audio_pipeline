import os
import gc
import time
import argparse
import logging
import torch
import librosa
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import jiwer
import glob
import shutil
from pathlib import Path
from collections import defaultdict

# --- Model Specific Imports ---
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
except ImportError:
    Wav2Vec2ForCTC = None
try:
    from nemo.collections.asr.models import EncDecMultiTaskModel, EncDecCTCModel
except ImportError:
    EncDecMultiTaskModel = None
    EncDecCTCModel = None
try:
    import whisper
except ImportError:
    whisper = None

# --- NVML for GPU monitoring ---
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_SUPPORT = True
except (ImportError, pynvml.NVMLError):
    GPU_SUPPORT = False

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model & Benchmark Configuration ---
MODELS_CONFIG = {
    "whisper": {
        "name": "OpenAI Whisper",
        "model_id": "large-v3",
        "loader": lambda model_id, device: whisper.load_model(model_id, device=device) if whisper else None,
        "transcriber": lambda model, path: model.transcribe(path, language="en")["text"],
    },
    "wav2vec2": {
        "name": "Meta Wav2Vec2",
        "model_id": "facebook/wav2vec2-large-960h-lv60-self",
        "loader": lambda model_id, device: (
            Wav2Vec2ForCTC.from_pretrained(model_id).to(device),
            Wav2Vec2Processor.from_pretrained(model_id)
        ) if Wav2Vec2ForCTC else (None, None),
        "transcriber": lambda model_tuple, path: transcribe_wav2vec2(model_tuple, path),
    },
    "canary": {
        "name": "NVIDIA Canary",
        "model_id": "nvidia/canary-1b",
        "loader": lambda model_id, device: EncDecMultiTaskModel.from_pretrained(model_id).to(device) if EncDecMultiTaskModel else None,
        "transcriber": lambda model, path: model.transcribe(audio=[path], batch_size=1)[0].text,
    },
    "parakeet": {
        "name": "NVIDIA Parakeet",
        "model_id": "nvidia/parakeet-ctc-0.6b",
        "loader": lambda model_id, device: EncDecCTCModel.from_pretrained(model_id).to(device) if EncDecCTCModel else None,
        "transcriber": lambda model, path: model.transcribe(audio=[path], batch_size=1)[0].text,
    }
}

# ==============================================================================
# PERFORMANCE BENCHMARKING FUNCTIONS
# ==============================================================================

def transcribe_wav2vec2(model_tuple, audio_path):
    """Helper for Wav2Vec2 transcription."""
    model, processor = model_tuple
    device = model.device
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

def get_gpu_memory_usage(device_index=0):
    if not GPU_SUPPORT or not torch.cuda.is_available(): return 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def get_audio_files(source_dir, num_files):
    logger.info(f"Searching for .wav files in {source_dir}...")
    all_files = list(Path(source_dir).rglob("*.wav"))
    if not all_files:
        logger.error(f"No .wav files found in {source_dir}. Please check the path.")
        return []
    
    selected_files = all_files if num_files == -1 else all_files[:num_files]
    logger.info(f"Found {len(all_files)} files. Selected {len(selected_files)} for benchmarking.")
    return selected_files

def benchmark_model(model_key, config, audio_files, transcript_output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"--- Benchmarking {config['name']} on {device} ---")

    ram_before_load = get_ram_usage()
    vram_before_load = get_gpu_memory_usage()
    
    model = config["loader"](config["model_id"], device)
    
    if (isinstance(model, tuple) and model[0] is None) or (not isinstance(model, tuple) and model is None):
        logger.error(f"Failed to load model {config['name']}. Library might be missing. Skipping.")
        return None

    static_ram = get_ram_usage() - ram_before_load
    static_vram = get_gpu_memory_usage() - vram_before_load
    logger.info(f"Model loading memory usage: {static_ram:.2f} MiB RAM, {static_vram:.2f} MiB VRAM")

    results = []
    for i, audio_path in enumerate(audio_files):
        try:
            logger.info(f"Processing file {i+1}/{len(audio_files)}: {audio_path.name}")
            audio_duration = librosa.get_duration(path=str(audio_path))

            start_time = time.time()
            ram_before = get_ram_usage()
            vram_before = get_gpu_memory_usage()
            
            # Get transcription text
            transcription = config["transcriber"](model, str(audio_path))
            
            # Save transcription to file for WER calculation later
            # Using .stem ensures we don't get 'file.wav.txt'
            transcript_path = transcript_output_dir / f"{model_key}_{audio_path.stem}.txt"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcription)

            end_time = time.time()
            processing_time = end_time - start_time
            throughput = audio_duration / processing_time
            
            peak_ram = get_ram_usage() - ram_before_load
            peak_vram = get_gpu_memory_usage() - vram_before_load

            results.append({
                "processing_time": processing_time, "throughput": throughput,
                "peak_ram_mb": peak_ram, "peak_vram_mb": peak_vram,
            })
            logger.info(f"Time: {processing_time:.2f}s, Throughput: {throughput:.2f}x RTF")

        except Exception as e:
            logger.error(f"Error processing {audio_path.name}: {e}", exc_info=True)
            continue
    
    del model
    gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

    if not results:
        logger.warning(f"No files were successfully processed for {config['name']}.")
        return None
        
    return {
        "Model": config["name"],
        "Avg_Processing_Time_s": np.mean([r["processing_time"] for r in results]),
        "Avg_Throughput_RTF": np.mean([r["throughput"] for r in results]),
        "Avg_Peak_RAM_MiB": np.mean([r["peak_ram_mb"] for r in results]),
        "Avg_Peak_VRAM_MiB": np.mean([r["peak_vram_mb"] for r in results]),
    }

def plot_performance_results(df, output_path):
    logger.info(f"Generating performance plot and saving to {output_path}...")
    df_sorted = df.sort_values("Avg_Throughput_RTF", ascending=False)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle('ASR Model Performance Benchmark', fontsize=16)
    
    # Throughput, VRAM, RAM plots
    palettes = ["viridis", "plasma", "magma"]
    y_metrics = ["Avg_Throughput_RTF", "Avg_Peak_VRAM_MiB", "Avg_Peak_RAM_MiB"]
    titles = ["Average Throughput (Real-Time Factor)", "Average Peak VRAM Usage", "Average Peak RAM Usage"]
    y_labels = ["Audio secs / processing sec (Higher is Better)", "VRAM (MiB) (Lower is Better)", "RAM (MiB) (Lower is Better)"]
    fmts = ['%.2f', '%.0f', '%.0f']

    for i in range(3):
        sns.barplot(x="Model", y=y_metrics[i], data=df_sorted, ax=axes[i], palette=palettes[i])
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(y_labels[i])
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt=fmts[i])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path)
    plt.close()
    logger.info("Performance plot saved successfully.")

# ==============================================================================
# ACCURACY EVALUATION (WER) FUNCTIONS
# ==============================================================================

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.Strip()
])

def load_ground_truth(filepath):
    try:
        df = pd.read_csv(filepath)
        df.dropna(subset=['Filename', 'transcript'], inplace=True)
        return pd.Series(df.transcript.values, index=df.Filename).to_dict()
    except Exception as e:
        logger.error(f"Error loading ground truth file {filepath}: {e}")
        return None

def evaluate_accuracy(transcript_dir, ground_truth_file, output_csv_path):
    logger.info("\n--- Starting Accuracy Evaluation (WER) ---")
    ground_truth_map = load_ground_truth(ground_truth_file)
    if ground_truth_map is None: return

    model_data = defaultdict(lambda: {'refs': [], 'hyps': []})
    model_keys = MODELS_CONFIG.keys() # Use keys from our main config

    all_txt_files = glob.glob(str(transcript_dir / '*.txt'))
    logger.info(f"Found {len(all_txt_files)} transcript files to evaluate.")

    matched_count = 0
    for txt_file in all_txt_files:
        basename = os.path.basename(txt_file)
        model_key, gt_key = None, None

        for key in model_keys:
            if basename.startswith(key + '_'):
                model_key = key
                # Robustly get the base filename (e.g., "some_file_name")
                gt_key_base = basename[len(key) + 1:].replace('.txt', '')
                # Construct the ground truth key (e.g., "some_file_name.wav")
                gt_key = gt_key_base + '.wav'
                break
        
        if model_key and gt_key in ground_truth_map:
            reference_text = transformation(ground_truth_map[gt_key])
            with open(txt_file, 'r', encoding='utf-8') as f:
                hypothesis_text = transformation(f.read())

            if isinstance(reference_text, str) and hypothesis_text:
                model_data[MODELS_CONFIG[model_key]['name']]['refs'].append(reference_text)
                model_data[MODELS_CONFIG[model_key]['name']]['hyps'].append(hypothesis_text)
                matched_count += 1
        else:
            if model_key:
                 logger.warning(f"Could not find ground truth for transcript. Model: {model_key}, Expected GT Key: {gt_key}")
    
    logger.info(f"Successfully matched {matched_count} transcripts with ground truth entries.")

    if not model_data:
        logger.error("No model data could be collected for accuracy evaluation. Exiting.")
        return

    all_results = []
    for model_name, data in sorted(model_data.items()):
        if not data['refs']: continue
        output = jiwer.process_words(data['refs'], data['hyps'])
        all_results.append({
            'Model': model_name, 'WER': output.wer, 'MER': output.mer, 'WIL': output.wil,
            'Substitutions': output.substitutions, 'Deletions': output.deletions,
            'Insertions': output.insertions, 'Hits': output.hits,
            'Total_Words_in_Reference': output.hits + output.substitutions + output.deletions,
            'Total_Files_Matched': len(data['refs'])
        })

    if not all_results:
        logger.error("No accuracy results could be calculated.")
        return

    results_df = pd.DataFrame(all_results).sort_values(by='WER').reset_index(drop=True)
    results_df.to_csv(output_csv_path, index=False)
    logger.info(f"Accuracy evaluation complete. Results saved to {output_csv_path}")
    logger.info("\n--- Accuracy (WER) Results ---")
    print(results_df.to_string())

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run a full benchmark and accuracy evaluation for ASR models.")
    parser.add_argument(
        "--source_dir", 
        type=str, 
        default="/media/meow/One Touch/ems_call/long_calls_filtered",
        help="Directory containing audio files (.wav)."
    )
    parser.add_argument(
        "--num_files", 
        type=int, 
        default=5,
        help="Number of audio files to process. Use -1 for all."
    )
    parser.add_argument(
        "--ground_truth_file", 
        type=str, 
        default="/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv",
        help="Path to the ground truth CSV file for WER evaluation."
    )
    args = parser.parse_args()

    # --- Setup Directories ---
    output_dir = Path("results")
    transcript_dir = Path("temp_transcripts")
    output_dir.mkdir(exist_ok=True)
    transcript_dir.mkdir(exist_ok=True)

    # --- Get Audio Files ---
    audio_files = get_audio_files(args.source_dir, args.num_files)
    if not audio_files:
        shutil.rmtree(transcript_dir)
        return

    # --- Part 1: Performance Benchmarking (and save transcripts) ---
    logger.info("\n--- Starting Performance Benchmarking ---")
    all_perf_results = []
    for model_key, config in MODELS_CONFIG.items():
        results = benchmark_model(model_key, config, audio_files, transcript_dir)
        if results:
            all_perf_results.append(results)

    if all_perf_results:
        perf_df = pd.DataFrame(all_perf_results)
        perf_csv_path = output_dir / "asr_performance_benchmark.csv"
        perf_plot_path = output_dir / "asr_performance_benchmark.png"
        
        logger.info("\n--- Performance Benchmark Results ---")
        print(perf_df.to_string())
        perf_df.to_csv(perf_csv_path, index=False)
        logger.info(f"\nPerformance results saved to {perf_csv_path}")
        plot_performance_results(perf_df, perf_plot_path)
    else:
        logger.error("No models could be benchmarked.")

    # --- Part 2: Accuracy Evaluation ---
    accuracy_csv_path = output_dir / "asr_accuracy_evaluation.csv"
    evaluate_accuracy(transcript_dir, args.ground_truth_file, accuracy_csv_path)

    # --- Cleanup ---
    logger.info(f"Cleaning up temporary transcript directory: {transcript_dir}")
    shutil.rmtree(transcript_dir)
    logger.info("Done.")

if __name__ == "__main__":
    print("Ensure you have installed all required libraries: pandas, matplotlib, seaborn, psutil, librosa, torch, transformers, nemo_toolkit[asr], openai-whisper, pynvml, jiwer")
    main()
    if GPU_SUPPORT:
        pynvml.nvmlShutdown() 