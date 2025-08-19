#!/bin/bash

# Debug Transcript Discovery Script
# Replicates the exact transcript discovery logic from run_llm_pipeline.sh

set -e

# Default values
ASR_RESULTS_DIR=""
ENABLE_WHISPER_FILTER=true
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --asr_results_dir)
            ASR_RESULTS_DIR="$2"
            shift 2
            ;;
        --enable_whisper_filter)
            ENABLE_WHISPER_FILTER=true
            shift
            ;;
        --disable_whisper_filter)
            ENABLE_WHISPER_FILTER=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Debug Transcript Discovery"
            echo ""
            echo "Usage: $0 --asr_results_dir DIR [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --asr_results_dir DIR         Directory containing ASR results"
            echo "  --enable_whisper_filter       Enable filtering for Whisper results (default)"
            echo "  --disable_whisper_filter      Disable Whisper filtering"
            echo "  --verbose                     Enable verbose output"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --asr_results_dir /path/to/asr/results"
            echo "  $0 --asr_results_dir /path/to/asr/results --disable_whisper_filter"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$ASR_RESULTS_DIR" ]; then
    echo "Error: --asr_results_dir is required"
    echo "Use -h or --help for usage information"
    exit 1
fi

if [ ! -d "$ASR_RESULTS_DIR" ]; then
    echo "Error: ASR results directory does not exist: $ASR_RESULTS_DIR"
    exit 1
fi

echo "=== Debug Transcript Discovery ==="
echo "ASR Results Directory: $ASR_RESULTS_DIR"
echo "Whisper Filter: $ENABLE_WHISPER_FILTER"
echo "Verbose: $VERBOSE"
echo ""

# --- Step 1: Find ASR Transcripts (EXACT COPY from run_llm_pipeline.sh) ---
echo "--- Step 1: Locating ASR Transcripts ---"

# Look for transcripts in various possible locations
TRANSCRIPT_DIRS=()
if [ -d "$ASR_RESULTS_DIR/asr_transcripts" ]; then
    TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR/asr_transcripts")
fi
if [ -d "$ASR_RESULTS_DIR/merged_transcripts" ]; then
    TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR/merged_transcripts")
fi
if [ -d "$ASR_RESULTS_DIR/merged_segmented_transcripts" ]; then
    TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR/merged_segmented_transcripts")
fi

# If no specific transcript directory found, check the root
if [ ${#TRANSCRIPT_DIRS[@]} -eq 0 ]; then
    # Check if there are .txt files in the root directory
    if find "$ASR_RESULTS_DIR" -maxdepth 1 -name "*.txt" | grep -q .; then
        TRANSCRIPT_DIRS+=("$ASR_RESULTS_DIR")
    fi
fi

if [ ${#TRANSCRIPT_DIRS[@]} -eq 0 ]; then
    echo "❌ Error: No transcript directories found in $ASR_RESULTS_DIR"
    echo "Expected locations:"
    echo "  - $ASR_RESULTS_DIR/asr_transcripts/"
    echo "  - $ASR_RESULTS_DIR/merged_transcripts/"
    echo "  - $ASR_RESULTS_DIR/merged_segmented_transcripts/"
    echo "  - $ASR_RESULTS_DIR/*.txt (root directory)"
    
    # List what's actually in the directory
    echo ""
    echo "Contents of $ASR_RESULTS_DIR:"
    ls -la "$ASR_RESULTS_DIR" | head -20
    exit 1
fi

echo "✅ Found transcript directories:"
for dir in "${TRANSCRIPT_DIRS[@]}"; do
    echo "  - $dir"
done

# Count total transcript files
TOTAL_TRANSCRIPTS=0
echo ""
echo "Transcript file counts by directory:"
for dir in "${TRANSCRIPT_DIRS[@]}"; do
    COUNT=$(find "$dir" -name "*.txt" | wc -l)
    TOTAL_TRANSCRIPTS=$((TOTAL_TRANSCRIPTS + COUNT))
    echo "  - $dir: $COUNT files"
    
    if [ "$VERBOSE" = true ]; then
        echo "    Sample files:"
        find "$dir" -name "*.txt" | head -5 | while read file; do
            echo "      * $(basename "$file")"
        done
        if [ "$COUNT" -gt 5 ]; then
            echo "      ... and $((COUNT - 5)) more files"
        fi
    fi
done

echo ""
echo "✅ Total transcript files found: $TOTAL_TRANSCRIPTS"

if [ "$TOTAL_TRANSCRIPTS" -eq 0 ]; then
    echo "❌ No .txt files found in transcript directories"
    exit 1
fi

# --- Step 1.5: Whisper Filter (EXACT COPY from run_llm_pipeline.sh) ---
if [ "$ENABLE_WHISPER_FILTER" = true ]; then
    echo ""
    echo "--- Step 1.5: Filtering Whisper Results ---"
    echo "Filtering Whisper (large-v3) results from transcript directories..."
    
    # Count Whisper files
    WHISPER_COUNT=0
    echo ""
    echo "Whisper (large-v3) file counts by directory:"
    for dir in "${TRANSCRIPT_DIRS[@]}"; do
        DIR_WHISPER_COUNT=$(find "$dir" -name "*.txt" | grep -c "large-v3_" || echo "0")
        WHISPER_COUNT=$((WHISPER_COUNT + DIR_WHISPER_COUNT))
        echo "  - $dir: $DIR_WHISPER_COUNT Whisper files"
        
        if [ "$VERBOSE" = true ] && [ "$DIR_WHISPER_COUNT" -gt 0 ]; then
            echo "    Whisper files:"
            find "$dir" -name "*large-v3_*.txt" | head -5 | while read file; do
                echo "      * $(basename "$file")"
            done
            if [ "$DIR_WHISPER_COUNT" -gt 5 ]; then
                echo "      ... and $((DIR_WHISPER_COUNT - 5)) more files"
            fi
        fi
    done
    
    echo ""
    if [ "$WHISPER_COUNT" -gt 0 ]; then
        echo "✅ Found $WHISPER_COUNT Whisper (large-v3) files"
    else
        echo "❌ No Whisper (large-v3) files found!"
        echo ""
        echo "Available file patterns in transcript directories:"
        for dir in "${TRANSCRIPT_DIRS[@]}"; do
            echo "  Directory: $dir"
            find "$dir" -name "*.txt" | head -10 | while read file; do
                echo "    - $(basename "$file")"
            done
            TOTAL_IN_DIR=$(find "$dir" -name "*.txt" | wc -l)
            if [ "$TOTAL_IN_DIR" -gt 10 ]; then
                echo "    ... and $((TOTAL_IN_DIR - 10)) more files"
            fi
        done
        
        echo ""
        echo "Hint: Whisper files should contain 'large-v3_' in their filename"
        echo "Examples of expected patterns:"
        echo "  - large-v3_TIMESTAMP_call_N_segment_NNN.txt"
        echo "  - large-v3_202412010133-841696-14744_call_2_segment_001.txt"
    fi
else
    echo ""
    echo "--- Skipping Whisper Filter ---"
    WHISPER_COUNT=$TOTAL_TRANSCRIPTS
fi

# --- File Content Validation ---
echo ""
echo "--- Validating File Contents ---"

READABLE_FILES=0
EMPTY_FILES=0
UNREADABLE_FILES=0

# Sample a few files for content validation
SAMPLE_SIZE=5
echo "Sampling $SAMPLE_SIZE files for content validation..."

if [ "$ENABLE_WHISPER_FILTER" = true ]; then
    # Sample Whisper files
    readarray -t SAMPLE_FILES < <(find "${TRANSCRIPT_DIRS[@]}" -name "*large-v3_*.txt" | head -$SAMPLE_SIZE)
else
    # Sample all txt files
    readarray -t SAMPLE_FILES < <(find "${TRANSCRIPT_DIRS[@]}" -name "*.txt" | head -$SAMPLE_SIZE)
fi

for file in "${SAMPLE_FILES[@]}"; do
    echo ""
    echo "Checking file: $(basename "$file")"
    echo "  Path: $file"
    
    if [ ! -f "$file" ]; then
        echo "  ❌ File does not exist"
        UNREADABLE_FILES=$((UNREADABLE_FILES + 1))
        continue
    fi
    
    FILE_SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
    echo "  Size: $FILE_SIZE bytes"
    
    if [ ! -r "$file" ]; then
        echo "  ❌ File is not readable"
        UNREADABLE_FILES=$((UNREADABLE_FILES + 1))
        continue
    fi
    
    # Check if file is empty
    if [ ! -s "$file" ]; then
        echo "  ⚠️  File is empty"
        EMPTY_FILES=$((EMPTY_FILES + 1))
        continue
    fi
    
    # Try to read first few lines
    echo "  ✅ File is readable"
    READABLE_FILES=$((READABLE_FILES + 1))
    
    if [ "$VERBOSE" = true ]; then
        echo "  Content preview:"
        head -3 "$file" | sed 's/^/    /'
        CHAR_COUNT=$(wc -c < "$file")
        echo "  Total characters: $CHAR_COUNT"
    fi
done

echo ""
echo "=== Content Validation Summary ==="
echo "Sampled files: ${#SAMPLE_FILES[@]}"
echo "Readable files: $READABLE_FILES"
echo "Empty files: $EMPTY_FILES"
echo "Unreadable files: $UNREADABLE_FILES"

# --- Final Summary ---
echo ""
echo "=== Final Summary ==="
echo "ASR Results Directory: $ASR_RESULTS_DIR"
echo "Transcript Directories Found: ${#TRANSCRIPT_DIRS[@]}"
echo "Total Transcript Files: $TOTAL_TRANSCRIPTS"
if [ "$ENABLE_WHISPER_FILTER" = true ]; then
    echo "Whisper (large-v3) Files: $WHISPER_COUNT"
fi
echo "Sample Readable Files: $READABLE_FILES"
echo "Sample Empty Files: $EMPTY_FILES"
echo "Sample Unreadable Files: $UNREADABLE_FILES"

# Determine success
if [ "$TOTAL_TRANSCRIPTS" -eq 0 ]; then
    echo ""
    echo "❌ FAILED: No transcript files found"
    exit 1
elif [ "$ENABLE_WHISPER_FILTER" = true ] && [ "$WHISPER_COUNT" -eq 0 ]; then
    echo ""
    echo "❌ FAILED: No Whisper files found (try --disable_whisper_filter)"
    exit 1
elif [ "$READABLE_FILES" -eq 0 ] && [ ${#SAMPLE_FILES[@]} -gt 0 ]; then
    echo ""
    echo "❌ FAILED: No readable files in sample"
    exit 1
else
    echo ""
    echo "✅ SUCCESS: Transcript discovery completed successfully"
    echo ""
    echo "Next steps:"
    echo "1. Use these transcript directories in your pipeline:"
    for dir in "${TRANSCRIPT_DIRS[@]}"; do
        echo "   - $dir"
    done
    if [ "$ENABLE_WHISPER_FILTER" = true ]; then
        echo "2. Filter for Whisper files using pattern: *large-v3_*.txt"
    fi
    echo "3. Process files using your LLM pipeline"
fi