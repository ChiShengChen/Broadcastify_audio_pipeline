
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
from pathlib import Path

# --- Model Specific Imports ---
# It's okay if some of these fail, the script will only run benchmarks for installed libraries.
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
        "is_nemo": False,
    },
    "wav2vec2": {
        "name": "Meta Wav2Vec2",
        "model_id": "facebook/wav2vec2-large-960h-lv60-self",
        "loader": lambda model_id, device: (
            Wav2Vec2ForCTC.from_pretrained(model_id).to(device),
            Wav2Vec2Processor.from_pretrained(model_id)
        ) if Wav2Vec2ForCTC else (None, None),
        "transcriber": lambda model_tuple, path: transcribe_wav2vec2(model_tuple, path),
        "is_nemo": False,
    },
    "canary": {
        "name": "NVIDIA Canary",
        "model_id": "nvidia/canary-1b",
        "loader": lambda model_id, device: EncDecMultiTaskModel.from_pretrained(model_id).to(device) if EncDecMultiTaskModel else None,
        "transcriber": lambda model, path: model.transcribe(audio=[path], batch_size=1)[0].text,
        "is_nemo": True,
    },
    "parakeet": {
        "name": "NVIDIA Parakeet",
        "model_id": "nvidia/parakeet-ctc-0.6b",
        "loader": lambda model_id, device: EncDecCTCModel.from_pretrained(model_id).to(device) if EncDecCTCModel else None,
        "transcriber": lambda model, path: model.transcribe(audio=[path], batch_size=1)[0].text,
        "is_nemo": True,
    }
}

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
    """Returns the current GPU memory usage in MiB."""
    if not GPU_SUPPORT or not torch.cuda.is_available():
        return 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2

def get_ram_usage():
    """Returns the current process's RAM usage in MiB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def get_audio_files(source_dir, num_files):
    """Gathers a list of audio files for benchmarking."""
    logger.info(f"Searching for .wav files in {source_dir}...")
    all_files = list(Path(source_dir).rglob("*.wav"))
    if not all_files:
        logger.error(f"No .wav files found in {source_dir}. Please check the path.")
        return []
    
    if num_files == -1:
        selected_files = all_files
        logger.info(f"Found {len(all_files)} files. Selected all of them for a full benchmark.")
    else:
        selected_files = all_files[:num_files]
        logger.info(f"Found {len(all_files)} files. Selected {len(selected_files)} for benchmarking.")
        
    return selected_files

def benchmark_model(model_key, config, audio_files):
    """Loads a model, runs transcription on files, and measures performance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"--- Benchmarking {config['name']} on {device} ---")

    # --- Model Loading ---
    ram_before_load = get_ram_usage()
    vram_before_load = get_gpu_memory_usage()
    
    loader_func = config["loader"]
    model = loader_func(config["model_id"], device)
    
    if (isinstance(model, tuple) and model[0] is None) or (not isinstance(model, tuple) and model is None):
        logger.error(f"Failed to load model {config['name']}. Required library might be missing. Skipping.")
        return None

    ram_after_load = get_ram_usage()
    vram_after_load = get_gpu_memory_usage()
    
    static_ram = ram_after_load - ram_before_load
    static_vram = vram_after_load - vram_before_load
    logger.info(f"Model loading memory usage: {static_ram:.2f} MiB RAM, {static_vram:.2f} MiB VRAM")

    # --- Benchmarking Loop ---
    results = []
    transcriber_func = config["transcriber"]
    
    for i, audio_path in enumerate(audio_files):
        try:
            logger.info(f"Processing file {i+1}/{len(audio_files)}: {audio_path.name}")
            audio_duration = librosa.get_duration(path=str(audio_path))

            # Measure performance
            start_time = time.time()
            ram_before = get_ram_usage()
            vram_before = get_gpu_memory_usage()
            
            _ = transcriber_func(model, str(audio_path))
            
            end_time = time.time()
            ram_after = get_ram_usage()
            vram_after = get_gpu_memory_usage()

            processing_time = end_time - start_time
            throughput = audio_duration / processing_time
            
            # Record peak usage during transcription
            peak_ram = ram_after - ram_before_load
            peak_vram = vram_after - vram_before_load

            results.append({
                "processing_time": processing_time,
                "throughput": throughput,
                "peak_ram_mb": peak_ram,
                "peak_vram_mb": peak_vram,
            })
            logger.info(f"Time: {processing_time:.2f}s, Throughput: {throughput:.2f}x Real-Time")

        except Exception as e:
            logger.error(f"Error processing {audio_path.name}: {e}")
            continue
    
    # --- Cleanup ---
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Aggregate Results ---
    if not results:
        logger.warning(f"No files were successfully processed for {config['name']}.")
        return None
        
    avg_results = {
        "Model": config["name"],
        "Avg_Processing_Time_s": np.mean([r["processing_time"] for r in results]),
        "Avg_Throughput_RTF": np.mean([r["throughput"] for r in results]),
        "Avg_Peak_RAM_MiB": np.mean([r["peak_ram_mb"] for r in results]),
        "Avg_Peak_VRAM_MiB": np.mean([r["peak_vram_mb"] for r in results]),
    }
    return avg_results

def plot_results(df, output_path):
    """Generates and saves bar plots for the benchmark results."""
    logger.info(f"Generating plot and saving to {output_path}...")
    
    # Sort by throughput for better visualization
    df_sorted = df.sort_values("Avg_Throughput_RTF", ascending=False)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle('ASR Model Benchmark Comparison', fontsize=16)
    
    # Plot 1: Throughput (Higher is Better)
    sns.barplot(x="Model", y="Avg_Throughput_RTF", data=df_sorted, ax=axes[0], palette="viridis")
    axes[0].set_title("Average Throughput (Real-Time Factor)")
    axes[0].set_ylabel("Audio seconds processed per second (Higher is Better)")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.2f')

    # Plot 2: Peak VRAM Usage (Lower is Better)
    sns.barplot(x="Model", y="Avg_Peak_VRAM_MiB", data=df_sorted, ax=axes[1], palette="plasma")
    axes[1].set_title("Average Peak VRAM Usage")
    axes[1].set_ylabel("VRAM (MiB) (Lower is Better)")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.0f')

    # Plot 3: Peak RAM Usage (Lower is Better)
    sns.barplot(x="Model", y="Avg_Peak_RAM_MiB", data=df_sorted, ax=axes[2], palette="magma")
    axes[2].set_title("Average Peak RAM Usage")
    axes[2].set_ylabel("RAM (MiB) (Lower is Better)")
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt='%.0f')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path)
    plt.close()
    logger.info("Plot saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark various ASR models.")
    parser.add_argument(
        "--source_dir", 
        type=str, 
        default="/media/meow/One Touch/ems_call/long_calls_filtered",
        help="Directory containing audio files in subfolders."
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=5,
        help="Number of audio files to use for benchmarking. Use -1 to process all available files."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="asr_benchmark_results.csv",
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="asr_benchmark_comparison.png",
        help="Path to save the output plot image."
    )
    args = parser.parse_args()

    audio_files = get_audio_files(args.source_dir, args.num_files)
    if not audio_files:
        return

    all_results = []
    for model_key, config in MODELS_CONFIG.items():
        results = benchmark_model(model_key, config, audio_files)
        if results:
            all_results.append(results)

    if not all_results:
        logger.error("No models could be benchmarked. Please check your installations and paths.")
        return

    # --- Save results ---
    results_df = pd.DataFrame(all_results)
    logger.info("\n--- Benchmark Results ---")
    print(results_df.to_string())
    results_df.to_csv(args.output_csv, index=False)
    logger.info(f"\nResults saved to {args.output_csv}")

    # --- Plot results ---
    plot_results(results_df, args.output_plot)

if __name__ == "__main__":
    # Add a requirements check or advice
    print("Ensure you have installed all required libraries: pandas, matplotlib, seaborn, psutil, librosa, torch, transformers, nemo_toolkit[asr], openai-whisper, pynvml")
    main()
    if GPU_SUPPORT:
        pynvml.nvmlShutdown() 