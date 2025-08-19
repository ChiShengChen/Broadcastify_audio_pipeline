#!/usr/bin/env python3
"""
Test script for LLM-Enhanced Pipeline

This script tests the basic functionality of the LLM pipeline components
without requiring actual LLM API endpoints.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

def create_test_transcripts(output_dir: Path) -> List[Path]:
    """Create test transcript files"""
    test_transcripts = [
        {
            "filename": "test_call_1.txt",
            "content": "Patient reports chest pain and shortness of breath. Blood pressure is 140 over 90. Patient has history of diabetes and hypertension. Need immediate medical attention."
        },
        {
            "filename": "test_call_2.txt", 
            "content": "Motor vehicle accident on highway 101. Multiple vehicles involved. Patient has head injury and bleeding from left arm. Unconscious but breathing."
        },
        {
            "filename": "test_call_3.txt",
            "content": "Patient complaining of severe abdominal pain. Pain started 2 hours ago. Patient is 45 years old female. No known allergies. Taking aspirin for headache."
        }
    ]
    
    created_files = []
    for transcript in test_transcripts:
        file_path = output_dir / transcript["filename"]
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(transcript["content"])
        created_files.append(file_path)
    
    return created_files

def create_test_ground_truth(output_dir: Path) -> Path:
    """Create test ground truth file"""
    ground_truth_data = [
        {
            "Filename": "test_call_1.txt",
            "transcript": "Patient reports chest pain and shortness of breath. Blood pressure is 140 over 90. Patient has history of diabetes and hypertension. Need immediate medical attention."
        },
        {
            "Filename": "test_call_2.txt",
            "transcript": "Motor vehicle accident on highway 101. Multiple vehicles involved. Patient has head injury and bleeding from left arm. Unconscious but breathing."
        },
        {
            "Filename": "test_call_3.txt", 
            "transcript": "Patient complaining of severe abdominal pain. Pain started 2 hours ago. Patient is 45 years old female. No known allergies. Taking aspirin for headache."
        }
    ]
    
    ground_truth_file = output_dir / "test_ground_truth.csv"
    
    # Create CSV content
    csv_content = "Filename,transcript\n"
    for item in ground_truth_data:
        csv_content += f'"{item["Filename"]}","{item["transcript"]}"\n'
    
    with open(ground_truth_file, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    
    return ground_truth_file

def test_llm_client_creation():
    """Test LLM client creation without actual API calls"""
    try:
        # Import the client creation function
        sys.path.append('.')
        from llm_medical_correction import create_llm_client
        
        # Test client creation for different models
        models = ["gpt-oss-20b", "gpt-oss-120b", "BioMistral-7B", "Meditron-7B", "Llama-3-8B-UltraMedica"]
        
        for model in models:
            try:
                client = create_llm_client(model, "http://localhost:8000/v1", "http://localhost:8000/v1")
                print(f"✓ Successfully created client for {model}")
            except Exception as e:
                print(f"✗ Failed to create client for {model}: {e}")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_file_processing_functions():
    """Test file processing functions"""
    try:
        from llm_medical_correction import find_transcript_files, load_transcript, save_corrected_transcript
        
        # Create temporary test directory
        test_dir = Path("test_temp")
        test_dir.mkdir(exist_ok=True)
        
        # Create test files
        test_files = create_test_transcripts(test_dir)
        
        # Test find_transcript_files
        found_files = find_transcript_files([str(test_dir)])
        if len(found_files) == len(test_files):
            print("✓ find_transcript_files works correctly")
        else:
            print(f"✗ find_transcript_files found {len(found_files)} files, expected {len(test_files)}")
            return False
        
        # Test load_transcript
        test_content = load_transcript(test_files[0])
        if test_content:
            print("✓ load_transcript works correctly")
        else:
            print("✗ load_transcript failed")
            return False
        
        # Test save_corrected_transcript
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_output.txt"
        
        if save_corrected_transcript("Test corrected content", output_path):
            print("✓ save_corrected_transcript works correctly")
        else:
            print("✗ save_corrected_transcript failed")
            return False
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        
        return True
    except Exception as e:
        print(f"✗ File processing test failed: {e}")
        return False

def test_emergency_page_template():
    """Test emergency page template creation"""
    try:
        from llm_emergency_page_generator import create_emergency_page_template
        
        template = create_emergency_page_template()
        if template and "EMERGENCY PAGE TEMPLATE" in template:
            print("✓ Emergency page template creation works correctly")
            return True
        else:
            print("✗ Emergency page template creation failed")
            return False
    except Exception as e:
        print(f"✗ Emergency page template test failed: {e}")
        return False

def run_integration_test():
    """Run a basic integration test"""
    print("Running LLM-Enhanced Pipeline Integration Test")
    print("=" * 50)
    
    # Create test directory structure
    test_base_dir = Path("test_llm_pipeline")
    test_base_dir.mkdir(exist_ok=True)
    
    # Create ASR results structure
    asr_results_dir = test_base_dir / "asr_results"
    asr_results_dir.mkdir(exist_ok=True)
    
    # Create test transcripts
    print("Creating test transcripts...")
    test_files = create_test_transcripts(asr_results_dir)
    print(f"Created {len(test_files)} test transcript files")
    
    # Create test ground truth
    print("Creating test ground truth...")
    ground_truth_file = create_test_ground_truth(asr_results_dir)
    print(f"Created ground truth file: {ground_truth_file}")
    
    # Test LLM client creation
    print("\nTesting LLM client creation...")
    client_test_passed = test_llm_client_creation()
    
    # Test file processing functions
    print("\nTesting file processing functions...")
    file_processing_passed = test_file_processing_functions()
    
    # Test emergency page template
    print("\nTesting emergency page template...")
    template_test_passed = test_emergency_page_template()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"LLM Client Creation: {'✓ PASSED' if client_test_passed else '✗ FAILED'}")
    print(f"File Processing: {'✓ PASSED' if file_processing_passed else '✗ FAILED'}")
    print(f"Emergency Page Template: {'✓ PASSED' if template_test_passed else '✗ FAILED'}")
    
    if client_test_passed and file_processing_passed and template_test_passed:
        print("\n✓ All tests passed! The pipeline components are working correctly.")
        print(f"\nTest files created in: {test_base_dir}")
        print(f"ASR results directory: {asr_results_dir}")
        print(f"Ground truth file: {ground_truth_file}")
        print("\nYou can now run the LLM pipeline with:")
        print(f"./run_llm_enhanced_pipeline.sh --asr_results_dir {asr_results_dir}")
        return True
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test LLM-Enhanced Pipeline")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test files")
    parser.add_argument("--test-only", choices=["client", "files", "template"], 
                       help="Run only specific test")
    
    args = parser.parse_args()
    
    if args.cleanup:
        # Clean up test files
        test_dir = Path("test_llm_pipeline")
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
            print("✓ Test files cleaned up")
        else:
            print("No test files to clean up")
        return
    
    if args.test_only:
        # Run specific test
        if args.test_only == "client":
            test_llm_client_creation()
        elif args.test_only == "files":
            test_file_processing_functions()
        elif args.test_only == "template":
            test_emergency_page_template()
        return
    
    # Run full integration test
    success = run_integration_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 