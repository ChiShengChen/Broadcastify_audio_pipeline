# Audio Processing and Transcription Pipeline

This project is a comprehensive pipeline for processing audio data, from downloading raw files to generating text transcriptions. The workflow involves crawling, format conversion, voice activity detection (VAD), audio enhancement, and finally, transcription using various Whisper models.

## Prerequisites

Before running the scripts, ensure you have the necessary Python libraries installed. You can install them using pip:

```bash
pip install selenium requests pydub tqdm numpy scipy matplotlib torch torchaudio whisper
```

You will also need:
- **Firefox** installed for the Selenium web crawler.
- **FFmpeg** for audio format conversion (`pydub` and `ffmpeg` command in `bash_mp3_to_wav.sh` depend on it).

---

## Pipeline Workflow

The scripts should be run in the following order. **Important:** Before running each script, make sure to update the input and output directory paths within the file to match your environment.

### Step 1: Download Audio Data

- **Script:** `firefox_craw_data_v3.py`
- **Purpose:** Automatically logs into `broadcastify.com` and downloads audio archives as MP3 files for a specified date range.
- **Usage:**
    1.  Modify `LOGIN_URL`, `ARCHIVE_URL`, `USERNAME`, `PASSWORD`, and `DOWNLOAD_FOLDER` variables in the script.
    2.  Set the desired `startDate` and `endDate`.
    3.  Run the script: `python ems_call/firefox_craw_data_v3.py`

### Step 2: Convert MP3 to WAV

- **Script:** `mp3_to_wav.py` (Python) or `bash_mp3_to_wav.sh` (Bash)
- **Purpose:** Converts the downloaded MP3 files into WAV format, which is more suitable for detailed audio processing. The Python script offers more features like progress tracking and error handling.
- **Usage (Python):**
    1.  Update `input_folder` and `output_folder` in `mp3_to_wav.py`.
    2.  Run the script: `python ems_call/mp3_to_wav.py`

### Step 3: Analyze Audio Data (Optional)

- **Script:** `count_file_stat.py`
- **Purpose:** Analyzes a folder of WAV files to provide statistics, such as total file count, date range of the recordings, any missing dates, and a histogram of audio durations. This is useful for data verification.
- **Usage:**
    1.  Update `folder_path` and `save_path` at the bottom of the script.
    2.  Run the script: `python ems_call/count_file_stat.py`

- **Script:** `calculate_average_duration.py`
- **Purpose:** Scans a directory to count the total number of `.wav` files and calculate their average duration. Useful for getting a baseline understanding of raw data or VAD segments.
- **Usage:**
    1.  Update `TARGET_DIR` to the folder you want to analyze.
    2.  Run the script: `python ems_call/calculate_average_duration.py`

### Step 4: Detect Speech Segments (VAD)

- **Script:** `wav_vad_only_w_timepoint.py`
- **Purpose:** Uses the Silero VAD model to detect speech segments in the WAV files. It saves each detected segment as a separate, smaller WAV file and creates a corresponding text file (`*_time_point.txt`) logging the start and end times.
- **Usage:**
    1.  Update `input_folder` and `output_folder`.
    2.  Run the script: `python ems_call/wav_vad_only_w_timepoint.py`

### Step 5: Merge Segments into Coherent Calls

- **Script:** `merge_calls_by_timestamp.py`
- **Purpose:** Intelligently merges the fragmented speech segments from Step 4 into coherent calls. It identifies breaks between calls by analyzing the silence duration between segments, which is the recommended merging method.
- **Usage:**
    1.  Update `INPUT_DIR` and `OUTPUT_DIR`.
    2.  Adjust `CALL_BREAK_THRESHOLD_S` (e.g., 15 seconds) to define what constitutes a new call.
    3.  Run the script: `python ems_call/merge_calls_by_timestamp.py`

- **(Alternative) Script:** `concat_audio_w_adative_1min.py`
- **Purpose:** A simpler method to merge speech segments. It adds silence for short gaps (< 60s) or creates a new file for long gaps (> 60s).
- **Usage:**
    1.  Update `base_dir` and `output_dir`.
    2.  Run the script: `python ems_call/concat_audio_w_adative_1min.py`

### Step 6: Post-Processing and Analysis of Merged Calls (Optional)

This stage involves analyzing and filtering the calls merged in the previous step.

-   **Analyze Call Durations:**
    -   **Script:** `analyze_merged_call_stats.py`
    -   **Purpose:** Counts the total number of merged calls and categorizes them by duration (e.g., <1 min, 1-2 min, >2 min) to understand the distribution of call lengths.
    -   **Usage:** Update `TARGET_DIR` and run `python ems_call/analyze_merged_call_stats.py`.

-   **Filter Long Calls:**
    -   **Script:** `copy_long_calls.py`
    -   **Purpose:** Copies calls that are longer than a specified duration (e.g., 60 seconds) into a new directory for focused analysis, while preserving the folder structure.
    -   **Usage:** Update `SOURCE_DIR`, `DEST_DIR`, and `MIN_DURATION_S`, then run `python ems_call/copy_long_calls.py`.

-   **Generate Audio Properties Log:**
    -   **Script:** `analyze_audio_properties.py`
    -   **Purpose:** Scans a folder of audio files, extracts key properties (duration, sample rate, channels, bit depth), and saves the information into a detailed `.csv` log file.
    -   **Usage:** Update `TARGET_DIR` and `OUTPUT_LOG_FILE`, then run `python ems_call/analyze_audio_properties.py`.

### Step 7: Filter and Enhance Audio

- **Script:** `audio_filter_enhance_plot.py`
- **Purpose:** Applies various audio processing techniques (Wiener filter for noise reduction, high-pass filter, and a band-pass filter for speech enhancement) to the audio files. It saves the processed audio and generates plots for visual comparison of the original vs. enhanced audio.
- **Usage:**
    1.  Update `input_folder_path` to the directory containing the audio you want to process (e.g., the output from Step 5 or the filtered calls from Step 6).
    2.  Run the script: `python ems_call/audio_filter_enhance_plot.py`

### Step 8: Transcribe Audio to Text

- **Script:** `batch_transcribe_and_save_whisper_models_new.py`
- **Purpose:** Performs batch transcription on the processed audio files using different Whisper models (`tiny`, `medium`, `large-v2`, etc.). It organizes the text outputs into subdirectories based on the model used and skips files that have already been transcribed.
- **Usage:**
    1.  Update `input_directory` and `base_output_dir`. The input can be any directory with audio files, such as the output from Step 7.
    2.  Run the script: `python ems_call/batch_transcribe_and_save_whisper_models_new.py` 

---

## Utility and Experimental Scripts (`tool/` directory)

This directory contains additional scripts for various specialized tasks, experiments, and data verification.

-   **`firefox_craw_data_v3_c2.py`**
    -   **Purpose:** A variant of the main data crawling script, likely configured for different target dates or download directories.

-   **`raw_data_vad.py`**
    -   **Purpose:** An integrated script that processes a folder of `.mp3` files directly. It converts each MP3 to a temporary WAV file, performs VAD to extract speech segments, saves them, and then deletes the temporary file.

-   **`wav_vad_only.py`**
    -   **Purpose:** A variation of the main VAD script (`wav_vad_only_w_timepoint.py`). It performs speech activity detection but **does not** create a `_time_point.txt` log file.

-   **`concat_audio_w_2s.py`**
    -   **Purpose:** Concatenates audio segments from subdirectories into a single file per subdirectory, inserting a 2-second silent gap between each segment.

-   **`pick_tri_all10s.py`**
    -   **Purpose:** Scans audio files, detects non-silent parts, and extracts a 10-second clip from the beginning of each non-silent segment. Useful for creating a dataset of short audio samples.

-   **`check_files.py`**
    -   **Purpose:** A utility to check for missing files within a specified date range in a folder. It compares expected filenames (based on YYYYMMDD format) against existing files.

-   **`test_ping.py`**
    -   **Purpose:** A simple script to test network connectivity to `broadcastify.com` by sending a request and saving the HTML response. 