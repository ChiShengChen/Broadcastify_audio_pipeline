#!/bin/bash

# Complete workflow test script
echo "=== Complete Workflow Test ==="
echo "This script tests the complete workflow:"
echo "1. Run pipeline without ground truth (should work)"
echo "2. Test multi-ASR auto-detection with LLM pipeline"
echo ""

# Step 1: Run pipeline without ground truth
echo "Step 1: Running pipeline without ground truth..."
PIPELINE_OUTPUT="./pipeline_test_$(date +%Y%m%d_%H%M%S)"

echo "Running: bash run_pipeline.sh --input_dir random_samples_1 --output_dir $PIPELINE_OUTPUT"
echo ""

# Run the pipeline
bash run_pipeline.sh --input_dir random_samples_1 --output_dir "$PIPELINE_OUTPUT" 2>&1 | tee pipeline_test.log

PIPELINE_EXIT_CODE=$?

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Step 1 PASSED: Pipeline completed successfully"
    echo "Pipeline output directory: $PIPELINE_OUTPUT"
    
    # Check if ASR results were generated
    if [ -d "$PIPELINE_OUTPUT/asr_transcripts" ] || [ -d "$PIPELINE_OUTPUT/merged_transcripts" ] || [ -d "$PIPELINE_OUTPUT/merged_segmented_transcripts" ]; then
        echo "✅ ASR results found in pipeline output"
        
        # Find transcript directory
        TRANSCRIPT_DIR=""
        if [ -d "$PIPELINE_OUTPUT/merged_segmented_transcripts" ]; then
            TRANSCRIPT_DIR="$PIPELINE_OUTPUT/merged_segmented_transcripts"
        elif [ -d "$PIPELINE_OUTPUT/merged_transcripts" ]; then
            TRANSCRIPT_DIR="$PIPELINE_OUTPUT/merged_transcripts"
        elif [ -d "$PIPELINE_OUTPUT/asr_transcripts" ]; then
            TRANSCRIPT_DIR="$PIPELINE_OUTPUT/asr_transcripts"
        fi
        
        if [ -n "$TRANSCRIPT_DIR" ]; then
            echo "Using transcript directory: $TRANSCRIPT_DIR"
            
            # Count transcript files
            TRANSCRIPT_COUNT=$(find "$TRANSCRIPT_DIR" -name "*.txt" | wc -l)
            echo "Found $TRANSCRIPT_COUNT transcript files"
            
            # Check for different ASR models
            echo "Checking for different ASR models..."
            find "$TRANSCRIPT_DIR" -name "*.txt" | head -10 | while read file; do
                echo "  - $(basename "$file")"
            done
            
            echo ""
            echo "Step 2: Testing multi-ASR auto-detection..."
            echo "Running: ./run_llm_pipeline.sh --asr_results_dir $PIPELINE_OUTPUT --enable_multi_asr_comparison --enable_auto_detect_multi_asr --enable_information_extraction --disable_page_generation"
            echo ""
            
            # Run LLM pipeline with multi-ASR auto-detection
            ./run_llm_pipeline.sh \
                --asr_results_dir "$PIPELINE_OUTPUT" \
                --enable_multi_asr_comparison \
                --enable_auto_detect_multi_asr \
                --enable_information_extraction \
                --disable_page_generation 2>&1 | tee llm_test.log
            
            LLM_EXIT_CODE=$?
            
            if [ $LLM_EXIT_CODE -eq 0 ]; then
                echo ""
                echo "✅ Step 2 PASSED: LLM pipeline with multi-ASR auto-detection completed"
                
                # Check LLM output
                LLM_OUTPUT_DIR=$(find . -name "llm_results_*" -type d | sort | tail -1)
                if [ -n "$LLM_OUTPUT_DIR" ]; then
                    echo "LLM output directory: $LLM_OUTPUT_DIR"
                    
                    # Check for multi-ASR organized results
                    if [ -d "$LLM_OUTPUT_DIR/multi_asr_organized" ]; then
                        echo "✅ Multi-ASR organized results found"
                        ls -la "$LLM_OUTPUT_DIR/multi_asr_organized/"
                        
                        if [ -f "$LLM_OUTPUT_DIR/multi_asr_organized/multi_asr_mapping.json" ]; then
                            echo "✅ Multi-ASR mapping file found"
                            echo "Mapping file contents:"
                            head -20 "$LLM_OUTPUT_DIR/multi_asr_organized/multi_asr_mapping.json"
                        fi
                    fi
                    
                    # Check for corrected transcripts
                    if [ -d "$LLM_OUTPUT_DIR/corrected_transcripts" ]; then
                        CORRECTED_COUNT=$(find "$LLM_OUTPUT_DIR/corrected_transcripts" -name "*.txt" | wc -l)
                        echo "✅ Corrected transcripts found: $CORRECTED_COUNT files"
                    fi
                    
                    # Check for extracted information
                    if [ -d "$LLM_OUTPUT_DIR/extracted_information" ]; then
                        EXTRACTED_COUNT=$(find "$LLM_OUTPUT_DIR/extracted_information" -name "*.txt" | wc -l)
                        echo "✅ Extracted information found: $EXTRACTED_COUNT files"
                    fi
                fi
            else
                echo ""
                echo "❌ Step 2 FAILED: LLM pipeline with multi-ASR auto-detection failed"
                echo "Check llm_test.log for details"
            fi
        else
            echo "❌ No transcript directory found in pipeline output"
        fi
    else
        echo "❌ No ASR results found in pipeline output"
    fi
else
    echo ""
    echo "❌ Step 1 FAILED: Pipeline failed"
    echo "Check pipeline_test.log for details"
fi

echo ""
echo "=== Test Summary ==="
echo "Pipeline output: $PIPELINE_OUTPUT"
echo "Pipeline log: pipeline_test.log"
echo "LLM log: llm_test.log"

if [ -n "$LLM_OUTPUT_DIR" ]; then
    echo "LLM output: $LLM_OUTPUT_DIR"
fi

echo ""
echo "To run the complete workflow manually:"
echo "1. bash run_pipeline.sh --input_dir random_samples_1 --output_dir ./pipeline_results"
echo "2. ./run_llm_pipeline.sh --asr_results_dir ./pipeline_results --enable_multi_asr_comparison --enable_auto_detect_multi_asr --enable_information_extraction --disable_page_generation"
