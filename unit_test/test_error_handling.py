#!/usr/bin/env python3
"""
Test script to verify error handling functionality
"""

import os
import tempfile
import subprocess
import shutil
from pathlib import Path

def create_test_environment():
    """Create a test environment with various error conditions"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Create test audio directory
        audio_dir = temp_dir_path / "test_audio"
        audio_dir.mkdir()
        
        # Create a dummy audio file
        dummy_audio = audio_dir / "test.wav"
        dummy_audio.write_bytes(b"dummy audio content")
        
        # Create ground truth file with some issues
        gt_content = '''Filename,transcript,original_files,duration,group_id
test.wav,"[x] Test transcript with EMS and BLS content. Engine 7 to command. 10-4.",['test.wav'],30.0,1
missing_file.wav,"This file will be missing from transcripts",['missing.wav'],25.0,2
empty_file.wav,"",['empty.wav'],20.0,3'''
        
        gt_file = temp_dir_path / "test_ground_truth.csv"
        gt_file.write_text(gt_content)
        
        # Create output directory
        output_dir = temp_dir_path / "pipeline_output"
        output_dir.mkdir()
        
        # Create transcript directory with various issues
        transcript_dir = output_dir / "merged_transcripts"
        transcript_dir.mkdir(parents=True)
        
        # Create some valid transcript files
        valid_transcript = transcript_dir / "large-v3_test.txt"
        valid_transcript.write_text("This is a valid transcript content")
        
        # Create an empty file
        empty_transcript = transcript_dir / "canary-1b_empty_file.txt"
        empty_transcript.write_text("")
        
        # Create a file with encoding issues (binary content)
        encoding_issue = transcript_dir / "wav2vec-xls-r_encoding_issue.txt"
        encoding_issue.write_bytes(b"Binary content that should cause encoding error")
        
        # Create a file with suspicious content
        suspicious_content = transcript_dir / "parakeet-tdt-0.6b-v2_suspicious.txt"
        suspicious_content.write_text("ERROR: This file contains an error message")
        
        # Create a file that's too short
        short_content = transcript_dir / "large-v3_short.txt"
        short_content.write_text("Hi")
        
        return temp_dir_path, audio_dir, gt_file, output_dir, transcript_dir

def test_error_handling():
    """Test the error handling functionality"""
    
    print("=== Testing Error Handling Functionality ===")
    
    temp_dir_path, audio_dir, gt_file, output_dir, transcript_dir = create_test_environment()
    
    # Test 1: Run enhanced model file analysis
    print("\n1️⃣ Testing Enhanced Model File Analysis")
    print("-" * 50)
    
    error_log_file = output_dir / "error_analysis.log"
    analysis_file = output_dir / "model_analysis.txt"
    
    cmd = [
        "python3", "analyze_model_files_enhanced.py",
        "--transcript_dir", str(transcript_dir),
        "--ground_truth_file", str(gt_file),
        "--output_file", str(analysis_file),
        "--error_log_file", str(error_log_file),
        "--pipeline_output_dir", str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Enhanced model file analysis completed successfully")
        
        # Check if error log was created
        if error_log_file.exists():
            print("✓ Error log file created")
            
            # Read and display error log
            error_content = error_log_file.read_text()
            print(f"Error log content ({len(error_content)} characters):")
            print(error_content[:500] + "..." if len(error_content) > 500 else error_content)
        else:
            print("✗ Error log file not created")
        
        # Check if analysis file was created
        if analysis_file.exists():
            print("✓ Analysis file created")
            
            # Read and display analysis content
            analysis_content = analysis_file.read_text()
            print(f"Analysis content ({len(analysis_content)} characters):")
            print(analysis_content[:500] + "..." if len(analysis_content) > 500 else analysis_content)
        else:
            print("✗ Analysis file not created")
    else:
        print(f"✗ Enhanced model file analysis failed: {result.stderr}")
    
    # Test 2: Test with missing directory
    print("\n2️⃣ Testing with Missing Directory")
    print("-" * 50)
    
    missing_dir = output_dir / "nonexistent"
    error_log_file2 = output_dir / "error_analysis2.log"
    analysis_file2 = output_dir / "model_analysis2.txt"
    
    cmd2 = [
        "python3", "analyze_model_files_enhanced.py",
        "--transcript_dir", str(missing_dir),
        "--ground_truth_file", str(gt_file),
        "--output_file", str(analysis_file2),
        "--error_log_file", str(error_log_file2),
        "--pipeline_output_dir", str(output_dir)
    ]
    
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    if result2.returncode == 1:
        print("✓ Correctly handled missing directory")
        
        if error_log_file2.exists():
            error_content2 = error_log_file2.read_text()
            print("✓ Error log created for missing directory")
            print("Error log content:")
            print(error_content2)
        else:
            print("✗ Error log not created for missing directory")
    else:
        print(f"✗ Unexpected result for missing directory: {result2.returncode}")
    
    # Test 3: Test with invalid ground truth file
    print("\n3️⃣ Testing with Invalid Ground Truth File")
    print("-" * 50)
    
    invalid_gt = output_dir / "invalid_gt.csv"
    invalid_gt.write_text("invalid,csv,content")
    
    error_log_file3 = output_dir / "error_analysis3.log"
    analysis_file3 = output_dir / "model_analysis3.txt"
    
    cmd3 = [
        "python3", "analyze_model_files_enhanced.py",
        "--transcript_dir", str(transcript_dir),
        "--ground_truth_file", str(invalid_gt),
        "--output_file", str(analysis_file3),
        "--error_log_file", str(error_log_file3),
        "--pipeline_output_dir", str(output_dir)
    ]
    
    result3 = subprocess.run(cmd3, capture_output=True, text=True)
    
    if result3.returncode == 1:
        print("✓ Correctly handled invalid ground truth file")
        
        if error_log_file3.exists():
            error_content3 = error_log_file3.read_text()
            print("✓ Error log created for invalid ground truth")
            print("Error log content:")
            print(error_content3)
        else:
            print("✗ Error log not created for invalid ground truth")
    else:
        print(f"✗ Unexpected result for invalid ground truth: {result3.returncode}")
    
    print("\n=== Error Handling Test Summary ===")
    print("All tests completed. Check the generated files for detailed error analysis.")

def main():
    """Main test function"""
    test_error_handling()

if __name__ == '__main__':
    main() 