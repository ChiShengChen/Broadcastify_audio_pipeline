# Debug Transcript Reading

Based on the `run_llm_pipeline.sh` script, I've created two debug tools to help you troubleshoot transcript reading issues.

## Overview

The `run_llm_pipeline.sh` script uses a specific approach to discover and read transcripts:

1. **Step 1: Find ASR Transcripts** - Look for transcripts in multiple possible locations
2. **Step 1.5: Whisper Filter** - Filter only Whisper (large-v3) results if enabled
3. **Step 2+: Process transcripts** - Pass to LLM processing

## Debug Tools

### 1. Bash Debug Script (Quick Check)

```bash
./debug_transcript_discovery.sh --asr_results_dir "/path/to/asr/results"
```

**Features:**
- Replicates exact logic from `run_llm_pipeline.sh`
- Fast discovery and validation
- Shows file counts and patterns
- Sample content validation

**Usage Examples:**
```bash
# Basic usage
./debug_transcript_discovery.sh --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902"

# Disable Whisper filtering
./debug_transcript_discovery.sh --asr_results_dir "/path/to/results" --disable_whisper_filter

# Verbose output
./debug_transcript_discovery.sh --asr_results_dir "/path/to/results" --verbose

# Help
./debug_transcript_discovery.sh --help
```

### 2. Python Debug Script (Detailed Analysis)

```bash
python3 debug_transcript_reading.py --asr_results_dir "/path/to/asr/results"
```

**Features:**
- Comprehensive file validation
- Content analysis and patterns
- Detailed error reporting
- JSON output support

**Usage Examples:**
```bash
# Basic usage
python3 debug_transcript_reading.py --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902"

# Disable Whisper filtering
python3 debug_transcript_reading.py --asr_results_dir "/path/to/results" --disable_whisper_filter

# Verbose output with JSON results
python3 debug_transcript_reading.py --asr_results_dir "/path/to/results" --verbose --output_file debug_results.json

# Help
python3 debug_transcript_reading.py --help
```

## Transcript Discovery Logic (from run_llm_pipeline.sh)

The script looks for transcripts in this order:

1. `$ASR_RESULTS_DIR/asr_transcripts/`
2. `$ASR_RESULTS_DIR/merged_transcripts/`
3. `$ASR_RESULTS_DIR/merged_segmented_transcripts/`
4. `$ASR_RESULTS_DIR/*.txt` (root directory)

### Whisper Filtering

When enabled (default), only files with `large-v3_` in the filename are processed:
- Pattern: `*large-v3_*.txt`
- Example: `large-v3_202412010133-841696-14744_call_2_segment_001.txt`

## Common Issues and Solutions

### Issue 1: "No transcript directories found"

**Symptoms:**
```
❌ Error: No transcript directories found in /path/to/results
Expected locations:
  - /path/to/results/asr_transcripts/
  - /path/to/results/merged_transcripts/
  - /path/to/results/merged_segmented_transcripts/
  - /path/to/results/*.txt (root directory)
```

**Solutions:**
1. Check if the ASR results directory path is correct
2. Verify the directory structure matches expected format
3. Look for `.txt` files in subdirectories

### Issue 2: "No Whisper files found"

**Symptoms:**
```
❌ No Whisper (large-v3) files found!
```

**Solutions:**
1. Disable Whisper filtering: `--disable_whisper_filter`
2. Check if files have the correct naming pattern (`large-v3_`)
3. Verify ASR pipeline generated Whisper results

### Issue 3: "No readable files"

**Symptoms:**
```
❌ No readable transcript files found
```

**Solutions:**
1. Check file permissions
2. Verify files are not empty
3. Check file encoding (should be UTF-8)
4. Verify files are not corrupted

### Issue 4: Empty or corrupted files

**Solutions:**
1. Re-run the ASR pipeline
2. Check disk space during ASR processing
3. Verify ASR models are working correctly

## Example Directory Structures

### Typical ASR Results Structure:
```
pipeline_results_20250729_033902/
├── asr_transcripts/
│   ├── large-v3_202412010133-841696-14744_call_1_segment_000.txt
│   ├── large-v3_202412010133-841696-14744_call_2_segment_001.txt
│   └── ...
├── merged_transcripts/
│   └── ...
└── evaluation_results.csv
```

### Alternative Structure:
```
simple_llm_results/
├── emergency_pages/
│   ├── large-v3_202412010133-841696-14744_call_1_segment_000_emergency_page.txt
│   └── ...
└── corrected_transcripts/
    └── ...
```

## Integration with LLM Pipeline

Once you've debugged transcript reading, the files are passed to:

1. **Medical Correction**: `llm_local_models.py --mode medical_correction`
2. **Emergency Page Generation**: `llm_local_models.py --mode emergency_page`

The `llm_local_models.py` script uses this function to find files:
```python
def find_transcript_files(input_dirs: List[str]) -> List[Path]:
    transcript_files = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            continue
        # Find all .txt files
        for txt_file in input_path.rglob("*.txt"):
            transcript_files.append(txt_file)
    return transcript_files
```

## Quick Debug Commands

```bash
# Test your specific directory
./debug_transcript_discovery.sh --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902"

# If no Whisper files found, try without filtering
./debug_transcript_discovery.sh --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902" --disable_whisper_filter

# For detailed analysis
python3 debug_transcript_reading.py --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250729_033902" --verbose
```

## Next Steps

After debugging:
1. Fix any identified issues
2. Run the full LLM pipeline with corrected paths
3. Use the same directory structure for consistent results

The debug tools will help you identify exactly where the transcript reading process fails and provide specific guidance for fixing the issues.