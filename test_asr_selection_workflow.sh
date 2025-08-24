#!/bin/bash

echo "=== Testing ASR Selection Workflow ==="
echo ""

# Set up test parameters
ASR_RESULTS_DIR="/media/meow/One Touch/ems_call/pipeline_results_20250823_095857"
TEST_OUTPUT_DIR="/media/meow/One Touch/ems_call/llm_results_asr_selection_test"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Test Configuration:"
echo "  ASR Results Directory: $ASR_RESULTS_DIR"
echo "  Test Output Directory: $TEST_OUTPUT_DIR"
echo ""

# Check if ASR results directory exists
if [ ! -d "$ASR_RESULTS_DIR" ]; then
    echo "❌ Error: ASR results directory not found: $ASR_RESULTS_DIR"
    exit 1
fi

# Check for both Canary and Whisper files
echo "Checking for multi-ASR files..."
CANARY_COUNT=$(find "$ASR_RESULTS_DIR" -name "canary-1b_*.txt" | wc -l)
WHISPER_COUNT=$(find "$ASR_RESULTS_DIR" -name "large-v3_*.txt" | wc -l)

echo "  Canary files found: $CANARY_COUNT"
echo "  Whisper files found: $WHISPER_COUNT"

if [ $CANARY_COUNT -eq 0 ] || [ $WHISPER_COUNT -eq 0 ]; then
    echo "❌ Error: Need both Canary and Whisper files for ASR selection"
    echo "  Canary files: $CANARY_COUNT"
    echo "  Whisper files: $WHISPER_COUNT"
    exit 1
fi

echo "✅ Multi-ASR files found - ready for ASR selection"
echo ""

# Clean up previous test output
if [ -d "$TEST_OUTPUT_DIR" ]; then
    echo "Cleaning up previous test output..."
    rm -rf "$TEST_OUTPUT_DIR"
fi

echo "Running ASR Selection Pipeline..."
echo ""

# Run the pipeline with ASR selection enabled
./run_llm_pipeline.sh \
    --asr_results_dir "$ASR_RESULTS_DIR" \
    --output_dir "$TEST_OUTPUT_DIR" \
    --enable_asr_selection \
    --disable_whisper_filter \
    --enable_auto_detect_multi_asr \
    --medical_correction_model gpt-oss-20b \
    --disable_page_generation \
    --disable_evaluation

PIPELINE_EXIT_CODE=$?

echo ""
echo "=== Pipeline Execution Results ==="
echo "Exit code: $PIPELINE_EXIT_CODE"
echo ""

# Check results
if [ -d "$TEST_OUTPUT_DIR" ]; then
    echo "✅ Test output directory created"
    
    # Check for corrected transcripts
    if [ -d "$TEST_OUTPUT_DIR/corrected_transcripts" ]; then
        CORRECTED_COUNT=$(find "$TEST_OUTPUT_DIR/corrected_transcripts" -name "*.txt" | wc -l)
        echo "✅ Corrected transcripts: $CORRECTED_COUNT"
        
        # Check for ASR selection CSV report
        if [ -f "$TEST_OUTPUT_DIR/corrected_transcripts/asr_selection_results.csv" ]; then
            echo "✅ ASR selection CSV report found"
            
            # Show CSV content
            echo ""
            echo "=== ASR Selection Results ==="
            head -5 "$TEST_OUTPUT_DIR/corrected_transcripts/asr_selection_results.csv"
            echo "..."
            echo ""
            
            # Count selections
            CANARY_SELECTIONS=$(grep -c "canary" "$TEST_OUTPUT_DIR/corrected_transcripts/asr_selection_results.csv" || echo "0")
            WHISPER_SELECTIONS=$(grep -c "whisper" "$TEST_OUTPUT_DIR/corrected_transcripts/asr_selection_results.csv" || echo "0")
            
            echo "Selection Summary:"
            echo "  Canary selected: $CANARY_SELECTIONS"
            echo "  Whisper selected: $WHISPER_SELECTIONS"
            
        else
            echo "❌ ASR selection CSV report not found"
        fi
    else
        echo "❌ Corrected transcripts directory not found"
    fi
    
    # Check for multi-ASR organized directory
    if [ -d "$TEST_OUTPUT_DIR/multi_asr_organized" ]; then
        echo "✅ Multi-ASR organized directory found"
        ORG_CANARY=$(find "$TEST_OUTPUT_DIR/multi_asr_organized/canary" -name "*.txt" 2>/dev/null | wc -l)
        ORG_WHISPER=$(find "$TEST_OUTPUT_DIR/multi_asr_organized/whisper" -name "*.txt" 2>/dev/null | wc -l)
        echo "  Organized Canary files: $ORG_CANARY"
        echo "  Organized Whisper files: $ORG_WHISPER"
    else
        echo "❌ Multi-ASR organized directory not found"
    fi
    
    # Show summary
    if [ -f "$TEST_OUTPUT_DIR/llm_enhanced_pipeline_summary.txt" ]; then
        echo ""
        echo "=== Pipeline Summary ==="
        grep -A 20 "Configuration:" "$TEST_OUTPUT_DIR/llm_enhanced_pipeline_summary.txt"
    fi
    
else
    echo "❌ Test output directory not created"
fi

echo ""
echo "=== Test Conclusion ==="
if [ $PIPELINE_EXIT_CODE -eq 0 ] && [ -f "$TEST_OUTPUT_DIR/corrected_transcripts/asr_selection_results.csv" ]; then
    echo "✅ ASR Selection test PASSED"
    echo "  - Pipeline executed successfully"
    echo "  - ASR selection CSV report generated"
    echo "  - Results available in: $TEST_OUTPUT_DIR"
else
    echo "❌ ASR Selection test FAILED"
    echo "  - Check error logs for details"
    echo "  - Verify model availability"
    echo "  - Ensure multi-ASR files are present"
fi

echo ""
echo "Test completed at: $(date)"
