#!/bin/bash

# Test script to verify multi-ASR comparison fix
echo "=== Testing Multi-ASR Comparison Fix ==="
echo ""

# Check if we have existing multi-ASR results
echo "Looking for existing multi-ASR results..."

# Find the most recent pipeline with multi-ASR results
RECENT_PIPELINE=$(find . -name "pipeline_results_*" -type d | sort | tail -1)

if [ -n "$RECENT_PIPELINE" ]; then
    echo "Found pipeline: $RECENT_PIPELINE"
    
    # Check for multiple ASR models
    if [ -d "$RECENT_PIPELINE/merged_segmented_transcripts" ]; then
        CANARY_COUNT=$(ls "$RECENT_PIPELINE/merged_segmented_transcripts/"*canary* 2>/dev/null | wc -l)
        WHISPER_COUNT=$(ls "$RECENT_PIPELINE/merged_segmented_transcripts/"*large-v3* 2>/dev/null | wc -l)
        
        echo "  - Canary files: $CANARY_COUNT"
        echo "  - Whisper files: $WHISPER_COUNT"
        
        if [ $CANARY_COUNT -gt 0 ] && [ $WHISPER_COUNT -gt 0 ]; then
            echo "✅ Found multi-ASR results! Testing the fix..."
            echo ""
            
            # Create a test output directory
            TEST_OUTPUT="./test_multi_asr_fix_$(date +%Y%m%d_%H%M%S)"
            
            echo "Running multi-ASR comparison with fixed model loading..."
            echo "Output directory: $TEST_OUTPUT"
            echo ""
            
            # Run the multi-ASR comparison directly
            python3 multi_asr_comparison.py \
                --input_dir "$RECENT_PIPELINE/merged_segmented_transcripts" \
                --output_dir "$TEST_OUTPUT" \
                --model "gpt-oss-20b" \
                --device "auto" \
                --prompt "Compare these two ASR transcripts and provide the best combined version: CANARY: {canary_transcript} WHISPER: {whisper_transcript} COMBINED:" \
                --batch_size 1
            
            if [ $? -eq 0 ]; then
                echo ""
                echo "✅ Multi-ASR comparison test completed successfully!"
                
                # Check results
                if [ -d "$TEST_OUTPUT" ]; then
                    echo "Results saved to: $TEST_OUTPUT"
                    ls -la "$TEST_OUTPUT"
                    
                    # Count output files
                    OUTPUT_COUNT=$(find "$TEST_OUTPUT" -name "*.txt" | wc -l)
                    echo "Generated $OUTPUT_COUNT combined transcript files"
                fi
            else
                echo "❌ Multi-ASR comparison test failed"
            fi
        else
            echo "❌ Not enough ASR models found for comparison"
            echo "   Need both Canary and Whisper results"
        fi
    else
        echo "❌ No merged transcripts found"
    fi
else
    echo "❌ No pipeline results found"
fi

echo ""
echo "=== Summary ==="
echo "The fix addresses the model loading issue in multi_asr_comparison.py"
echo "by adding proper model name mapping to HuggingFace paths."
echo ""
echo "Key changes:"
echo "  - Added _get_model_path() method to map model names"
echo "  - gpt-oss-20b -> openai/gpt-oss-20b"
echo "  - gpt-oss-120b -> openai/gpt-oss-120b"
echo "  - etc."
echo ""
echo "Now the multi-ASR comparison should work correctly!"
