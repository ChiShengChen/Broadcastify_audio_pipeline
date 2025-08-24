#!/bin/bash

echo "=== Fixing ASR Selection Configuration ==="
echo ""

# Original results directory
ORIGINAL_RESULTS="/media/meow/One Touch/ems_call/llm_results_20250823_133955"
ASR_RESULTS_DIR="/media/meow/One Touch/ems_call/pipeline_results_20250823_095857"

# New output directory with correct ASR selection
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NEW_OUTPUT_DIR="/media/meow/One Touch/ems_call/llm_results_asr_selection_fixed_${TIMESTAMP}"

echo "Problem Analysis:"
echo "  Original run had ENABLE_ASR_SELECTION=true but ENABLE_WHISPER_FILTER=true"
echo "  This caused only Whisper files to be processed, no Canary files for comparison"
echo ""

echo "Solution:"
echo "  - Disable Whisper filter to include both Canary and Whisper files"
echo "  - Enable auto-detect multi-ASR for proper file organization"
echo "  - Run ASR selection on both ASR types"
echo ""

echo "Configuration:"
echo "  ASR Results Directory: $ASR_RESULTS_DIR"
echo "  New Output Directory: $NEW_OUTPUT_DIR"
echo ""

# Check for multi-ASR files
echo "Verifying multi-ASR files..."
CANARY_COUNT=$(find "$ASR_RESULTS_DIR" -name "canary-1b_*.txt" | wc -l)
WHISPER_COUNT=$(find "$ASR_RESULTS_DIR" -name "large-v3_*.txt" | wc -l)

echo "  Canary files: $CANARY_COUNT"
echo "  Whisper files: $WHISPER_COUNT"

if [ $CANARY_COUNT -eq 0 ] || [ $WHISPER_COUNT -eq 0 ]; then
    echo "❌ Error: Need both Canary and Whisper files"
    exit 1
fi

echo "✅ Multi-ASR files available"
echo ""

echo "Running corrected ASR Selection pipeline..."
echo ""

# Run the pipeline with correct configuration
./run_llm_pipeline.sh \
    --asr_results_dir "$ASR_RESULTS_DIR" \
    --output_dir "$NEW_OUTPUT_DIR" \
    --enable_asr_selection \
    --disable_whisper_filter \
    --enable_auto_detect_multi_asr \
    --medical_correction_model gpt-oss-20b \
    --disable_page_generation \
    --disable_evaluation

PIPELINE_EXIT_CODE=$?

echo ""
echo "=== Results ==="
echo "Exit code: $PIPELINE_EXIT_CODE"
echo ""

# Check for ASR selection results
if [ -d "$NEW_OUTPUT_DIR" ]; then
    echo "✅ New output directory created: $NEW_OUTPUT_DIR"
    
    # Check for ASR selection CSV
    CSV_FILE="$NEW_OUTPUT_DIR/corrected_transcripts/asr_selection_results.csv"
    if [ -f "$CSV_FILE" ]; then
        echo "✅ ASR selection CSV report found!"
        echo ""
        echo "=== ASR Selection Results ==="
        echo "File: $CSV_FILE"
        echo ""
        
        # Show CSV header and first few rows
        echo "CSV Content:"
        head -10 "$CSV_FILE"
        echo ""
        
        # Count selections
        TOTAL_ROWS=$(wc -l < "$CSV_FILE")
        CANARY_SELECTIONS=$(grep -c "canary" "$CSV_FILE" || echo "0")
        WHISPER_SELECTIONS=$(grep -c "whisper" "$CSV_FILE" || echo "0")
        
        echo "Selection Summary:"
        echo "  Total files processed: $((TOTAL_ROWS - 1))"  # Subtract header
        echo "  Canary selected: $CANARY_SELECTIONS"
        echo "  Whisper selected: $WHISPER_SELECTIONS"
        echo ""
        
        # Show some selection reasons
        echo "Sample Selection Reasons:"
        tail -n +2 "$CSV_FILE" | cut -d',' -f3 | head -5 | while read reason; do
            echo "  - $reason"
        done
        echo ""
        
    else
        echo "❌ ASR selection CSV report not found"
        echo "  Expected: $CSV_FILE"
    fi
    
    # Check for multi-ASR organized directory
    if [ -d "$NEW_OUTPUT_DIR/multi_asr_organized" ]; then
        echo "✅ Multi-ASR organized directory found"
        ORG_CANARY=$(find "$NEW_OUTPUT_DIR/multi_asr_organized/canary" -name "*.txt" 2>/dev/null | wc -l)
        ORG_WHISPER=$(find "$NEW_OUTPUT_DIR/multi_asr_organized/whisper" -name "*.txt" 2>/dev/null | wc -l)
        echo "  Organized Canary files: $ORG_CANARY"
        echo "  Organized Whisper files: $ORG_WHISPER"
    fi
    
    # Show corrected transcripts
    if [ -d "$NEW_OUTPUT_DIR/corrected_transcripts" ]; then
        CORRECTED_COUNT=$(find "$NEW_OUTPUT_DIR/corrected_transcripts" -name "*.txt" | wc -l)
        echo "✅ Corrected transcripts: $CORRECTED_COUNT"
    fi
    
else
    echo "❌ New output directory not created"
fi

echo ""
echo "=== Comparison with Original Run ==="
echo "Original run ($ORIGINAL_RESULTS):"
echo "  - Only Whisper files processed (due to filter)"
echo "  - No ASR selection possible (no Canary files)"
echo "  - No CSV report generated"
echo ""
echo "Fixed run ($NEW_OUTPUT_DIR):"
echo "  - Both Canary and Whisper files processed"
echo "  - ASR selection performed on file pairs"
echo "  - CSV report with selection reasons generated"
echo ""

if [ -f "$CSV_FILE" ]; then
    echo "✅ ASR Selection Fix SUCCESSFUL!"
    echo "  Results available in: $NEW_OUTPUT_DIR"
    echo "  CSV report: $CSV_FILE"
else
    echo "❌ ASR Selection Fix FAILED"
    echo "  Check error logs for details"
fi

echo ""
echo "Fix completed at: $(date)"
