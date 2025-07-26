#!/usr/bin/env python3
"""
Test script to simulate pipeline success and failure scenarios
"""

import os
import tempfile
import subprocess
from pathlib import Path

def create_test_scenario(scenario_name, has_errors=False, has_warnings=False):
    """Create a test scenario with specific error conditions"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Create output directory structure
        output_dir = temp_dir_path / "test_output"
        output_dir.mkdir()
        
        # Create error log file
        error_log_file = output_dir / "error_analysis.log"
        
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write("=== Error Analysis Log ===\n")
            f.write("Analysis Date: 2025-07-27 07:15:00\n")
            f.write("Pipeline Output Directory: test_output\n\n")
            
            if has_errors:
                f.write("[ERROR] 2025-07-27T07:15:01.000000 - TEST_ERROR: Test error message\n")
                f.write("  Details: This is a test error for scenario testing\n")
                f.write("  File: test_file.txt\n\n")
            
            if has_warnings:
                f.write("[WARNING] 2025-07-27T07:15:02.000000 - TEST_WARNING: Test warning message\n")
                f.write("  Details: This is a test warning for scenario testing\n")
                f.write("  File: test_file.txt\n\n")
            
            f.write("[WARNING] 2025-07-27T07:15:03.000000 - ANALYSIS_COMPLETE: Analysis completed\n")
            f.write("  Details: Test analysis completed\n\n")
        
        # Create evaluation results file (only for success scenarios)
        if not has_errors:
            eval_file = output_dir / "asr_evaluation_results.csv"
            eval_file.write_text("Model,WER,MER,WIL\nlarge-v3,0.5,0.4,0.6\n")
        
        # Create transcript directory (only for success scenarios)
        if not has_errors:
            transcript_dir = output_dir / "merged_transcripts"
            transcript_dir.mkdir()
            (transcript_dir / "test.txt").write_text("test content")
        
        # Create summary file
        summary_file = output_dir / "pipeline_summary.txt"
        summary_file.write_text("Pipeline Summary\n===============\nTest scenario completed.\n")
        
        print(f"Created test scenario: {scenario_name}")
        print(f"  Output directory: {output_dir}")
        print(f"  Has errors: {has_errors}")
        print(f"  Has warnings: {has_warnings}")
        print()

def test_pipeline_status_logic():
    """Test the pipeline status logic"""
    
    print("=== Testing Pipeline Status Logic ===")
    print()
    
    # Test scenarios
    scenarios = [
        ("Success - No Errors, No Warnings", False, False),
        ("Success - No Errors, With Warnings", False, True),
        ("Failure - With Errors, No Warnings", True, False),
        ("Failure - With Errors, With Warnings", True, True),
    ]
    
    for scenario_name, has_errors, has_warnings in scenarios:
        print(f"Testing: {scenario_name}")
        print("-" * 50)
        
        create_test_scenario(scenario_name, has_errors, has_warnings)
        
        # Simulate the status check logic
        error_count = 1 if has_errors else 0
        warning_count = 1 if has_warnings else 0
        
        pipeline_success = error_count == 0
        
        if pipeline_success:
            print("✅ Status: Pipeline Completed Successfully")
            if warning_count > 0:
                print(f"⚠️  Note: {warning_count} warnings were detected during processing.")
        else:
            print("❌ Status: Pipeline Completed with Errors")
            print(f"  - Errors detected: {error_count}")
            if warning_count > 0:
                print(f"  - Warnings detected: {warning_count}")
        
        print()

def test_actual_pipeline_output():
    """Test with actual pipeline output"""
    
    print("=== Testing with Actual Pipeline Output ===")
    print()
    
    # Check if we have actual pipeline results
    pipeline_dir = "pipeline_results_20250727_070443"
    
    if os.path.exists(pipeline_dir):
        print(f"Found actual pipeline output: {pipeline_dir}")
        
        error_log_file = os.path.join(pipeline_dir, "error_analysis.log")
        output_file = os.path.join(pipeline_dir, "asr_evaluation_results.csv")
        transcript_dir = os.path.join(pipeline_dir, "merged_transcripts")
        
        # Check error log
        error_count = 0
        warning_count = 0
        
        if os.path.exists(error_log_file):
            with open(error_log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                error_count = content.count("[ERROR]")
                warning_count = content.count("[WARNING]")
        
        # Determine status
        pipeline_success = True
        
        if error_count > 0:
            pipeline_success = False
        
        if not os.path.exists(output_file):
            pipeline_success = False
        
        if not os.path.exists(transcript_dir):
            pipeline_success = False
        
        # Display results
        print("Actual Pipeline Status:")
        print(f"  Error count: {error_count}")
        print(f"  Warning count: {warning_count}")
        print(f"  Evaluation file exists: {os.path.exists(output_file)}")
        print(f"  Transcript directory exists: {os.path.exists(transcript_dir)}")
        print()
        
        if pipeline_success:
            print("✅ Status: Pipeline Completed Successfully")
            if warning_count > 0:
                print(f"⚠️  Note: {warning_count} warnings were detected during processing.")
        else:
            print("❌ Status: Pipeline Completed with Errors")
            if error_count > 0:
                print(f"  - Errors detected: {error_count}")
            if warning_count > 0:
                print(f"  - Warnings detected: {warning_count}")
            if not os.path.exists(output_file):
                print("  - Missing evaluation results file")
            if not os.path.exists(transcript_dir):
                print("  - Missing transcript directory")
    else:
        print("No actual pipeline output found for testing")

def main():
    """Main test function"""
    test_pipeline_status_logic()
    test_actual_pipeline_output()

if __name__ == '__main__':
    main() 