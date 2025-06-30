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

### Step 4: Detect Speech Segments (VAD)

- **Script:** `wav_vad_only_w_timepoint.py`
- **Purpose:** Uses the Silero VAD model to detect speech segments in the WAV files. It saves each detected segment as a separate, smaller WAV file and creates a corresponding text file (`*_time_point.txt`) logging the start and end times.
- **Usage:**
    1.  Update `input_folder` and `output_folder`.
    2.  Run the script: `python ems_call/wav_vad_only_w_timepoint.py`

### Step 5: Concatenate Speech Segments

- **Script:** `concat_audio_w_adative_1min.py`
- **Purpose:** Merges the speech segments created in the previous step. It intelligently adds silence for short gaps (< 60s) between segments or creates a new file for long gaps (> 60s).
- **Usage:**
    1.  Update `base_dir` and `output_dir`.
    2.  Run the script: `python ems_call/concat_audio_w_adative_1min.py`

### Step 6: Filter and Enhance Audio

- **Script:** `audio_filter_enhance_plot.py`
- **Purpose:** Applies various audio processing techniques (Wiener filter for noise reduction, high-pass filter, and a band-pass filter for speech enhancement) to the audio files. It saves the processed audio and generates plots for visual comparison of the original vs. enhanced audio.
- **Usage:**
    1.  Update `input_folder_path` to the directory containing the audio you want to process (e.g., the output from Step 5).
    2.  Run the script: `python ems_call/audio_filter_enhance_plot.py`

### Step 7: Transcribe Audio to Text

- **Script:** `batch_transcribe_and_save_whisper_models_new.py`
- **Purpose:** Performs batch transcription on the processed audio files using different Whisper models (`tiny`, `medium`, `large-v2`, etc.). It organizes the text outputs into subdirectories based on the model used and skips files that have already been transcribed.
- **Usage:**
    1.  Update `input_directory` and `base_output_dir`.
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
 
---  
## Related Repo:  
https://github.com/ChiShengChen/broadcastify_mp3_crawler  
https://github.com/ChiShengChen/whisper_audio_translate  
