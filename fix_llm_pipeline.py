#!/usr/bin/env python3
"""
Fix LLM Pipeline Issues

This script identifies and fixes common issues with the LLM pipeline.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        'torch',
        'transformers', 
        'accelerate',
        'bitsandbytes'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    return missing_packages

def fix_bash_script_args():
    """Fix argument passing issues in the bash script"""
    script_path = Path("run_llm_pipeline.sh")
    
    if not script_path.exists():
        print("run_llm_pipeline.sh not found")
        return False
    
    # Read the script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Fix the input_dirs argument passing
    # Replace: --input_dirs "${TRANSCRIPT_DIRS[@]}"
    # With: --input_dirs $(printf '%s ' "${TRANSCRIPT_DIRS[@]}")
    
    fixes = [
        {
            'old': '--input_dirs "${TRANSCRIPT_DIRS[@]}"',
            'new': '--input_dirs $(printf "%s " "${TRANSCRIPT_DIRS[@]}")'
        }
    ]
    
    modified = False
    for fix in fixes:
        if fix['old'] in content:
            content = content.replace(fix['old'], fix['new'])
            modified = True
            print(f"✓ Fixed argument passing: {fix['old']}")
    
    if modified:
        # Create backup
        backup_path = script_path.with_suffix('.sh.backup')
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Write fixed version
        with open(script_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed script saved, backup at {backup_path}")
        return True
    else:
        print("No fixes needed in bash script")
        return False

def create_lightweight_model_config():
    """Create a configuration for lightweight models"""
    config = {
        "models": {
            "medical_correction": {
                "name": "microsoft/DialoGPT-small",
                "description": "Lightweight model for CPU usage",
                "requirements": {
                    "memory": "2GB",
                    "device": "cpu"
                }
            },
            "emergency_page": {
                "name": "microsoft/DialoGPT-small", 
                "description": "Lightweight model for CPU usage",
                "requirements": {
                    "memory": "2GB",
                    "device": "cpu"
                }
            }
        },
        "prompts": {
            "medical_correction": "Please correct any medical terms in this transcript: {transcript}\nCorrected:",
            "emergency_page": "Generate an emergency page from this medical transcript: {transcript}\nEmergency Page:"
        }
    }
    
    config_path = Path("lightweight_llm_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created lightweight model config: {config_path}")
    return config_path

def test_simple_pipeline():
    """Test the simple pipeline with a sample file"""
    # Find a sample Whisper file
    asr_dir = Path("/media/meow/One Touch/ems_call/pipeline_results_20250729_033902")
    if not asr_dir.exists():
        print("ASR results directory not found, cannot test")
        return False
    
    sample_files = list(asr_dir.rglob("*large-v3*.txt"))
    if not sample_files:
        print("No Whisper files found for testing")
        return False
    
    sample_file = sample_files[0]
    print(f"Testing with sample file: {sample_file}")
    
    # Create test output directory
    test_output = Path("test_simple_llm_output")
    test_output.mkdir(exist_ok=True)
    
    # Test medical correction
    try:
        cmd = [
            "python3", "simple_llm_pipeline.py",
            "--mode", "medical_correction",
            "--input_dirs", str(sample_file.parent),
            "--output_dir", str(test_output / "corrected"),
            "--model", "microsoft/DialoGPT-small",
            "--batch_size", "1"
        ]
        
        print("Running test medical correction...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Medical correction test passed")
            return True
        else:
            print(f"✗ Medical correction test failed:")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Test timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False

def main():
    print("=== LLM Pipeline Fix Script ===")
    print()
    
    # Check Python packages
    print("1. Checking Python packages...")
    missing = check_python_packages()
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        print()
    
    # Fix bash script
    print("2. Checking bash script...")
    fix_bash_script_args()
    print()
    
    # Create lightweight config
    print("3. Creating lightweight model config...")
    create_lightweight_model_config()
    print()
    
    # Test simple pipeline
    if not missing:  # Only test if packages are available
        print("4. Testing simple pipeline...")
        test_simple_pipeline()
        print()
    else:
        print("4. Skipping pipeline test (missing packages)")
        print()
    
    print("=== Recommendations ===")
    print()
    
    if missing:
        print("CRITICAL: Install missing Python packages first")
        print(f"  pip install {' '.join(missing)}")
        print()
    
    print("OPTION 1: Use the simple pipeline (recommended for testing)")
    print("  ./run_simple_llm_pipeline.sh")
    print()
    
    print("OPTION 2: Fix the main pipeline")
    print("  1. Install missing packages")
    print("  2. Use smaller models or CPU mode")
    print("  3. Check model accessibility")
    print("  ./debug_llm_pipeline.sh  # Run debug script first")
    print()
    
    print("OPTION 3: Use the original pipeline with fixes")
    print("  1. Ensure all packages are installed")
    print("  2. Use --load_in_8bit for memory efficiency")
    print("  3. Use --device cpu if no GPU")
    print("  ./run_llm_pipeline.sh --asr_results_dir <path> --load_in_8bit --device cpu")

if __name__ == "__main__":
    main()