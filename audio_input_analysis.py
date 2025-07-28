import soundfile as sf
import librosa
import os
import datetime

AUDIO_EXTS = [".wav", ".flac", ".mp3"]

# Define requirements for each ASR model
ASR_REQUIREMENTS = {
    "Whisper": {"min_sec": 0, "max_sec": float('inf'), "min_sr": 0, "max_sr": float('inf')},
    "Canary": {"min_sec": 0.5, "max_sec": 60, "min_sr": 16000, "max_sr": 16000},
    "Parakeet": {"min_sec": 1, "max_sec": float('inf'), "min_sr": 16000, "max_sr": 16000},
    "Wav2Vec2": {"min_sec": 0.1, "max_sec": float('inf'), "min_sr": 16000, "max_sr": 16000}
}

# Suggested actions for common issues
SUGGESTION = {
    "duration": {
        "too_short": "Pad the audio with silence to reach the minimum duration.",
        "too_long": "Split the audio into smaller chunks, each under the maximum duration limit."
    },
    "samplerate": {
        "mismatch": "Resample the audio to the required sampling rate (e.g., with sox or librosa)."
    }
}

# Mapping from internal model key to display name
MODEL_DISPLAY_NAMES = {
    "Whisper": "Whisper (large-v3)",
    "Canary": "Canary-1b (NeMo)",
    "Parakeet": "Parakeet-tdt-0.6b-v2 (NeMo)",
    "Wav2Vec2": "Wav2Vec2-xls-r (Transformers)"
}

def analyze_audio(filepath):
    """
    Analyze audio file for duration, sample rate, channel count, and volume.
    Returns (duration_sec, sample_rate, channels, max_volume, avg_volume)
    """
    try:
        data, samplerate = sf.read(filepath)
    except RuntimeError:
        # For formats not supported by soundfile, fall back to librosa
        data, samplerate = librosa.load(filepath, sr=None, mono=False)
    duration_sec = len(data) / samplerate if hasattr(data, '__len__') else 0
    channels = 1 if len(data.shape) == 1 else data.shape[0] if hasattr(data, 'shape') else 1
    # Compute volume statistics
    import numpy as np
    if channels == 1:
        arr = data
    else:
        arr = data if isinstance(data, np.ndarray) else data[0]
        arr = arr[0] if arr.ndim > 1 else arr
    max_vol = float(np.max(np.abs(arr)))
    avg_vol = float(np.mean(np.abs(arr)))
    return duration_sec, samplerate, channels, max_vol, avg_vol

def check_asr_compat(duration, samplerate, channels, max_vol, avg_vol):
    """
    Check audio compatibility with each ASR model.
    Returns a dictionary of results per model (status, issues, recommendations).
    """
    results = {}
    for asr, req in ASR_REQUIREMENTS.items():
        duration_ok = req["min_sec"] <= duration <= req["max_sec"]
        samplerate_ok = req["min_sr"] == 0 or samplerate == req["min_sr"]
        channel_ok = (asr == "Whisper") or (channels == 1)
        volume_ok = (asr not in {"Canary", "Wav2Vec2"}) or (max_vol >= 0.01)
        issues = []
        if not duration_ok:
            if duration < req["min_sec"]:
                issues.append(f"Duration too short: {duration:.2f}s (minimum: {req['min_sec']}s)")
            if duration > req["max_sec"]:
                issues.append(f"Duration too long: {duration:.2f}s (maximum: {req['max_sec']}s)")
        if not samplerate_ok:
            issues.append(f"Sample rate mismatch: {samplerate}Hz (required: {req['min_sr']}Hz)")
        if not channel_ok:
            issues.append(f"Channel mismatch: {channels} channels (required: 1)")
        if not volume_ok:
            issues.append(f"Volume too low: {max_vol:.4f} (minimum: 0.01)")
        if not issues:
            # Compatible: meets all requirements
            results[MODEL_DISPLAY_NAMES[asr]] = {"status": "COMPATIBLE", "issues": [], "recommend": []}
        else:
            # Incompatible: accumulate recommendations for fixing issues
            recommend = []
            if not samplerate_ok:
                recommend.append("Resample to 16000Hz")
            if not channel_ok:
                recommend.append("Convert to mono if stereo")
            if not volume_ok:
                recommend.append("Normalize volume to minimum 0.01")
            if duration < req["min_sec"]:
                recommend.append(f"Pad audio to minimum {req['min_sec']}s")
            if duration > req["max_sec"]:
                recommend.append(f"Split audio into segments of maximum {req['max_sec']}s")
            recommend.append("Split long audio into segments")
            recommend.append("Pad short audio to minimum duration")
            results[MODEL_DISPLAY_NAMES[asr]] = {"status": "INCOMPATIBLE", "issues": issues, "recommend": recommend}
    return results

def model_compatibility_summary(all_results, model_names):
    """
    Compute statistics for each model:
    - Number of compatible files
    - List of incompatible filenames
    """
    summary = {m: 0 for m in model_names}
    incompatible_files = {m: [] for m in model_names}
    total = len(all_results)
    for fname, info in all_results.items():
        for m in model_names:
            v = info['asr_results'][m]['status']
            if v == "COMPATIBLE":
                summary[m] += 1
            else:
                incompatible_files[m].append(fname)
    lines = ["COMPATIBILITY SUMMARY:\n----------------------------------------"]
    for m in model_names:
        percent = summary[m] / total * 100 if total > 0 else 0
        lines.append(f"{m}: {summary[m]}/{total} files ({percent:.1f}%)")
    lines.append("\nINCOMPATIBLE FILES BY MODEL:")
    for m in model_names:
        if incompatible_files[m]:
            files_str = ', '.join(incompatible_files[m])
            lines.append(f"{m}: {files_str}")
        else:
            lines.append(f"{m}: (all compatible)")
    return "\n".join(lines)

def generate_report(all_results, input_dir):
    """
    Generate a full English report with model statistics,
    incompatible filenames, and per-file analysis details.
    """
    # Get models in the preferred display order
    model_names = [MODEL_DISPLAY_NAMES[k] for k in ["Whisper", "Canary", "Parakeet", "Wav2Vec2"]]
    now = datetime.datetime.now().isoformat()
    summary_text = model_compatibility_summary(all_results, model_names)
    report = f"""\
================================================================================
AUDIO ANALYSIS REPORT FOR ASR MODEL COMPATIBILITY
================================================================================
Analysis Date: {now}
Input Directory: {input_dir}
Total Files Analyzed: {len(all_results)}

{summary_text}

"""
    # File-by-file detail section
    for fname, info in all_results.items():
        duration = info['duration']
        samplerate = info['samplerate']
        channels = info['channels']
        max_vol = info['max_vol']
        avg_vol = info['avg_vol']
        try:
            stat = os.stat(info['filepath'])
            size = stat.st_size
        except Exception:
            size = -1
        report += f"""
FILE: {fname}
--------------------------------------------------
Duration: {duration:.2f} seconds
Sample Rate: {samplerate} Hz
Channels: {channels}
File Size: {size:,} bytes
Max Volume: {max_vol:.4f}
Average Volume: {avg_vol:.4f}

MODEL COMPATIBILITY:
"""
        for asr, result in info['asr_results'].items():
            status = result['status']
            report += f"  {asr}: {status}"
            if status == "INCOMPATIBLE":
                report += "\n    Issues:"
                for issue in result['issues']:
                    report += f"\n      - {issue}"
                report += "\n    Recommendations:"
                for r in set(result['recommend']):  # Use set to avoid duplicate lines
                    report += f"\n      - {r}"
            report += "\n"
        report += "\n"
    report += "="*80 + "\nEND OF REPORT\n" + "="*80 + "\n"
    return report

def scan_and_analyze(directory):
    """
    Walk through the given directory, analyze all supported audio files,
    and aggregate results in a dictionary keyed by filename.
    """
    all_results = {}
    for root, _, files in os.walk(directory):
        for f in files:
            if any(f.lower().endswith(ext) for ext in AUDIO_EXTS):
                path = os.path.join(root, f)
                try:
                    duration, samplerate, channels, max_vol, avg_vol = analyze_audio(path)
                except Exception as e:
                    # If reading fails, mark all models as ERROR for this file
                    duration, samplerate, channels, max_vol, avg_vol = -1, -1, -1, -1, -1
                    asr_results = {MODEL_DISPLAY_NAMES[asr]: {
                        "status": "ERROR",
                        "issues": [f"Error reading file: {e}"],
                        "recommend": []
                    } for asr in ASR_REQUIREMENTS}
                    all_results[f] = {
                        "duration": duration,
                        "samplerate": samplerate,
                        "channels": channels,
                        "max_vol": max_vol,
                        "avg_vol": avg_vol,
                        "asr_results": asr_results,
                        "filepath": path
                    }
                    continue
                asr_results = check_asr_compat(duration, samplerate, channels, max_vol, avg_vol)
                all_results[f] = {
                    "duration": duration,
                    "samplerate": samplerate,
                    "channels": channels,
                    "max_vol": max_vol,
                    "avg_vol": avg_vol,
                    "asr_results": asr_results,
                    "filepath": path
                }
    return all_results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python audio_asr_folder_report.py <audio_folder>")
        exit(1)
    folder = sys.argv[1]
    all_results = scan_and_analyze(folder)
    report = generate_report(all_results, folder)
    with open("audio_asr_analysis_report.txt", "w") as f:
        f.write(report)
    print("Report saved to audio_asr_analysis_report.txt")
