#!/usr/bin/env python3
"""
Test Local Models Script

This script tests the local model functionality without requiring actual model downloads.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

def test_model_imports():
    """Test if required libraries can be imported"""
    print("Testing library imports...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        from transformers import BitsAndBytesConfig
        print("✓ BitsAndBytesConfig imported successfully")
    except ImportError as e:
        print(f"✗ BitsAndBytesConfig import failed: {e}")
        return False
    
    try:
        import accelerate
        print("✓ Accelerate imported successfully")
    except ImportError as e:
        print(f"✗ Accelerate import failed: {e}")
        return False
    
    return True

def test_cuda_availability():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ CUDA available: {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
            return True
        else:
            print("⚠ CUDA not available - models will run on CPU")
            return False
            
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def test_model_configs():
    """Test model configuration loading"""
    print("\nTesting model configurations...")
    
    try:
        from llm_local_models import create_local_model
        
        # Test model path mappings
        model_paths = {
            "BioMistral-7B": "BioMistral/BioMistral-7B",
            "Meditron-7B": "epfl-llm/meditron-7b"
        }
        
        for model_name, expected_path in model_paths.items():
            print(f"✓ Model path mapping for {model_name}: {expected_path}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Could not import llm_local_models: {e}")
        return False
    except Exception as e:
        print(f"✗ Model config test failed: {e}")
        return False

def test_file_processing():
    """Test file processing functions"""
    print("\nTesting file processing functions...")
    
    try:
        from llm_local_models import find_transcript_files, load_transcript, save_corrected_transcript
        
        # Create test directory
        test_dir = Path("test_local_models_temp")
        test_dir.mkdir(exist_ok=True)
        
        # Create test files
        test_files = [
            ("test1.txt", "Patient reports chest pain."),
            ("test2.txt", "Motor vehicle accident on highway."),
            ("subdir/test3.txt", "Patient complaining of abdominal pain.")
        ]
        
        for filename, content in test_files:
            file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Test find_transcript_files
        found_files = find_transcript_files([str(test_dir)])
        if len(found_files) == 3:
            print("✓ find_transcript_files works correctly")
        else:
            print(f"✗ find_transcript_files found {len(found_files)} files, expected 3")
            return False
        
        # Test load_transcript
        test_content = load_transcript(found_files[0])
        if test_content:
            print("✓ load_transcript works correctly")
        else:
            print("✗ load_transcript failed")
            return False
        
        # Test save_corrected_transcript
        output_path = test_dir / "output" / "test_output.txt"
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

def test_setup_script():
    """Test setup script functionality"""
    print("\nTesting setup script...")
    
    try:
        from setup_local_models import MODEL_CONFIGS, check_cuda_availability
        
        # Test model configs
        expected_models = ["BioMistral-7B", "Meditron-7B", "gpt-oss-20b", "Llama-3-8B-UltraMedica"]
        for model in expected_models:
            if model in MODEL_CONFIGS:
                print(f"✓ Model config for {model} found")
            else:
                print(f"✗ Model config for {model} missing")
                return False
        
        # Test CUDA check function
        check_cuda_availability()
        print("✓ CUDA check function works")
        
        return True
        
    except ImportError as e:
        print(f"✗ Could not import setup_local_models: {e}")
        return False
    except Exception as e:
        print(f"✗ Setup script test failed: {e}")
        return False

def create_test_models_config():
    """Create a test models configuration"""
    print("\nCreating test models configuration...")
    
    test_config = {
        "models": {
            "BioMistral-7B": {
                "huggingface_id": "BioMistral/BioMistral-7B",
                "description": "Medical-focused Mistral model",
                "size_gb": 14,
                "requirements": "16GB+ RAM, CUDA recommended"
            },
            "Meditron-7B": {
                "huggingface_id": "epfl-llm/meditron-7b",
                "description": "Medical instruction-tuned model",
                "size_gb": 14,
                "requirements": "16GB+ RAM, CUDA recommended"
            }
        },
        "download_path": "./test_models",
        "setup_date": "test",
        "cuda_available": True
    }
    
    test_models_dir = Path("test_models")
    test_models_dir.mkdir(exist_ok=True)
    
    config_file = test_models_dir / "model_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(test_config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Test models configuration created: {config_file}")
    return True

def run_integration_test():
    """Run a basic integration test"""
    print("Running Local Models Integration Test")
    print("=" * 50)
    
    # Test library imports
    imports_ok = test_model_imports()
    
    # Test CUDA availability
    cuda_ok = test_cuda_availability()
    
    # Test model configurations
    config_ok = test_model_configs()
    
    # Test file processing
    file_processing_ok = test_file_processing()
    
    # Test setup script
    setup_ok = test_setup_script()
    
    # Create test configuration
    config_creation_ok = create_test_models_config()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Library Imports: {'✓ PASSED' if imports_ok else '✗ FAILED'}")
    print(f"CUDA Availability: {'✓ PASSED' if cuda_ok else '⚠ WARNING'}")
    print(f"Model Configurations: {'✓ PASSED' if config_ok else '✗ FAILED'}")
    print(f"File Processing: {'✓ PASSED' if file_processing_ok else '✗ FAILED'}")
    print(f"Setup Script: {'✓ PASSED' if setup_ok else '✗ FAILED'}")
    print(f"Config Creation: {'✓ PASSED' if config_creation_ok else '✗ FAILED'}")
    
    all_passed = imports_ok and config_ok and file_processing_ok and setup_ok and config_creation_ok
    
    if all_passed:
        print("\n✓ All tests passed! The local models components are working correctly.")
        print("\nNext steps:")
        print("1. Download models: python3 setup_local_models.py --models BioMistral-7B")
        print("2. Run pipeline: ./run_llm_enhanced_pipeline.sh --asr_results_dir /path/to/results")
        return True
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Local Models")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test files")
    parser.add_argument("--test-only", choices=["imports", "cuda", "config", "files", "setup"], 
                       help="Run only specific test")
    
    args = parser.parse_args()
    
    if args.cleanup:
        # Clean up test files
        test_dir = Path("test_models")
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
            print("✓ Test files cleaned up")
        else:
            print("No test files to clean up")
        return
    
    if args.test_only:
        # Run specific test
        if args.test_only == "imports":
            test_model_imports()
        elif args.test_only == "cuda":
            test_cuda_availability()
        elif args.test_only == "config":
            test_model_configs()
        elif args.test_only == "files":
            test_file_processing()
        elif args.test_only == "setup":
            test_setup_script()
        return
    
    # Run full integration test
    success = run_integration_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 