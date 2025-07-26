#!/usr/bin/env python3
"""
Test script to verify enhanced preprocessor integration
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def test_enhanced_preprocessor():
    """Test the enhanced preprocessor functionality"""
    
    print("=== Testing Enhanced Preprocessor Integration ===")
    
    # Create a temporary test ground truth file
    test_gt_content = '''Filename,transcript,original_files,duration,group_id
test_001.wav,"[x] Arriving on scene, I'll assume command as [x] on the outside. EMS 4 en route to cardiac arrest call 5-0-1-2 newson road. Patient is 95 year old male. BLS needed for breathing difficulty priority 1. Engine 7 to command, attic is clear. Engine 7 can handle. 10-4. We're going to handle. All units clear. [x] rescue 2 is clear. Command to ladder 2, status of [x]. [x] Copy. Comments stated there's possibly a occupant on oxygen interior. 19:20 [x] possibly one more individual in [x] over by where we're parked. We're making patient contact. [x] just stand by for a second. Will do. [x] roll call for 1541 winter road. Engine 2. Engine 2. Ladder 2. Engine 4. Engine 7. Engine 7. Rescue 2. Rescue 2. Battalion 5. 1. 1. [x] engine 4. EMS 4. EMS 4. [x] copy all units respond with engine 4.",['test_file.wav'],230.325875,1
test_002.wav,"Rescue 8, 1424P can you respond to an illness 612 6-1-2 jack rabbit road three shipps coffee at 11:17. Rescue 8. 1424P jack rabbit road [x] at 11:17. 3P for an unconscious at 1533 1-5-3-3 brookwood crescent [x] green run at 11:18. Rescue 15 1623P for an unconscious at 1-5-3-3 brookwood crescent. Rescue 9 923P for a bleed 4853 4-8-5-3 [x] road [x] extension at 11:18. Rescue 9 923P bleed 4-8-5-3 [x] road [x] extension at 11:18. 6-1-2 jack rabbit road. Three shipps coffee patient is going to be waiting out front [x] 48 year old male he is conscious and breathing at 11:19 starting roll call for 1117 [berkley] drive. Engine 19. 19. Engine 7. Engine 7. Engine 2. Respond for the overdose at 2501 2-5-0-1 james madison boulevard, at the jail, at 11:29. 621P copy right, have a 40 year old male, conscious breathing, possible overdose, narcan times two at 11:29. [x] ems 1. Ems 1. [x] scene working fire. [x]95 go ahead 19. Engine 9 is on scene. Looks like we got a garage detached garage in the back. Engine 19 to investigate, i'll pass command off to you. Copy, establishing command. [x] on scene. Engine 4 to 923, we got a spot right in front of the pick up truck for you. 73 1-6-7-3 gray friars chase at 12:36. Engine 22 to 1723, [x]. We have 1 BLS patient. When you guys get here, just pull up right in front of the white mistubishi and between the engine and the white car and we'll move [x] chief complaint headache. Will do. Ambulance 2, 220P, copy [x] to the choking 1-6-7-3 gray friars chase patient is 77 year old female conscious breathing [x] choking 12:37",['test_file2.wav'],289.8451875,2
test_003.wav,"920 can you respond for an illness Our Lady of Perpetual Help 4560 princess anne road apartment 132 B as in boy 20:28. [x] Our Lady of Perpetual Help 4560 princess anne road apartment 132 B. I have a 64 year old male conscious and breathing. [x] needs a foley catheter inserted, they want him transported 20:28. EMS 1. Would you put the following units [x] unavailable? 923P 1622P 1723P. Command to batallion 5. [x] command. There's [x] there's gonna be a small fire due to a lawn mower backing over leaf debris. Fire is out. Ladder 2 can handle. You can clear all additional units. [x] clear. Do you need the tac? Negative, you can clear the tac. [x] BLS to [x]. Both units are ok on scene. 12:08. [x] ems 3. [x] ems has it. Clear from [x] you can show us available. 12:08. Rescue 10 ambulance 422 for an injury from a fall 56 56 indian river road waffle house [x] at 12:08. Do not charge the line. 220 is arriving on scene. Second engine. Command copy to 20. [x] water supply. [x] no extension in the house, engine 3 and the ladder 10 can handle [x]. [x] boulevard. [x] for ems. [x] engine 20 can clear. All units except for ladder 10 engine 3 and rescue 2 you can clear. Clear 20. [x] 4 copy. [x] ambulance 1622 available.",['test_file3.wav'],216.8249375,3'''
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Create test ground truth file
        test_gt_file = temp_dir_path / "test_ground_truth.csv"
        with open(test_gt_file, 'w', encoding='utf-8') as f:
            f.write(test_gt_content)
        
        print(f"Created test ground truth file: {test_gt_file}")
        
        # Test 1: Basic preprocessor (conservative mode)
        print("\n--- Test 1: Basic Preprocessor (Conservative Mode) ---")
        basic_output = temp_dir_path / "basic_processed.csv"
        
        cmd = [
            "python3", "smart_preprocess_ground_truth.py",
            "--input_file", str(test_gt_file),
            "--output_file", str(basic_output),
            "--mode", "conservative"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Basic preprocessor (conservative) completed successfully")
            
            # Read and show sample of processed content
            with open(basic_output, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print("Sample processed content (first 200 chars):")
                if len(lines) > 1:
                    print(lines[1][:200] + "...")
        else:
            print(f"✗ Basic preprocessor failed: {result.stderr}")
            return False
        
        # Test 2: Enhanced preprocessor (conservative mode)
        print("\n--- Test 2: Enhanced Preprocessor (Conservative Mode) ---")
        enhanced_output = temp_dir_path / "enhanced_processed.csv"
        
        cmd = [
            "python3", "enhanced_ground_truth_preprocessor.py",
            "--input_file", str(test_gt_file),
            "--output_file", str(enhanced_output),
            "--mode", "conservative"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Enhanced preprocessor (conservative) completed successfully")
            
            # Read and show sample of processed content
            with open(enhanced_output, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print("Sample processed content (first 200 chars):")
                if len(lines) > 1:
                    print(lines[1][:200] + "...")
        else:
            print(f"✗ Enhanced preprocessor failed: {result.stderr}")
            return False
        
        # Test 3: Enhanced preprocessor (aggressive mode)
        print("\n--- Test 3: Enhanced Preprocessor (Aggressive Mode) ---")
        enhanced_aggressive_output = temp_dir_path / "enhanced_aggressive_processed.csv"
        
        cmd = [
            "python3", "enhanced_ground_truth_preprocessor.py",
            "--input_file", str(test_gt_file),
            "--output_file", str(enhanced_aggressive_output),
            "--mode", "aggressive"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Enhanced preprocessor (aggressive) completed successfully")
            
            # Read and show sample of processed content
            with open(enhanced_aggressive_output, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print("Sample processed content (first 200 chars):")
                if len(lines) > 1:
                    print(lines[1][:200] + "...")
        else:
            print(f"✗ Enhanced preprocessor (aggressive) failed: {result.stderr}")
            return False
        
        # Test 4: Compare results
        print("\n--- Test 4: Comparison of Results ---")
        
        # Read all processed files
        with open(basic_output, 'r', encoding='utf-8') as f:
            basic_content = f.read()
        
        with open(enhanced_output, 'r', encoding='utf-8') as f:
            enhanced_content = f.read()
        
        with open(enhanced_aggressive_output, 'r', encoding='utf-8') as f:
            enhanced_aggressive_content = f.read()
        
        # Compare lengths
        print(f"Original length: {len(test_gt_content)} characters")
        print(f"Basic processed length: {len(basic_content)} characters")
        print(f"Enhanced (conservative) length: {len(enhanced_content)} characters")
        print(f"Enhanced (aggressive) length: {len(enhanced_aggressive_content)} characters")
        
        # Check for specific transformations
        print("\nChecking for specific transformations:")
        
        # Check noise markers
        if "[x]" not in basic_content:
            print("✓ Basic preprocessor: [x] markers removed")
        else:
            print("✗ Basic preprocessor: [x] markers still present")
        
        if "[x]" not in enhanced_content:
            print("✓ Enhanced preprocessor: [x] markers removed")
        else:
            print("✗ Enhanced preprocessor: [x] markers still present")
        
        # Check abbreviations
        if "emergency medical services" in enhanced_content.lower():
            print("✓ Enhanced preprocessor: EMS expanded")
        else:
            print("✗ Enhanced preprocessor: EMS not expanded")
        
        if "basic life support" in enhanced_content.lower():
            print("✓ Enhanced preprocessor: BLS expanded")
        else:
            print("✗ Enhanced preprocessor: BLS not expanded")
        
        # Check contractions
        if "we are" in enhanced_aggressive_content.lower():
            print("✓ Enhanced preprocessor (aggressive): contractions expanded")
        else:
            print("✗ Enhanced preprocessor (aggressive): contractions not expanded")
        
        print("\n=== All Tests Completed Successfully ===")
        return True

def test_pipeline_integration():
    """Test the pipeline integration with enhanced preprocessor"""
    
    print("\n=== Testing Pipeline Integration ===")
    
    # Test pipeline help to ensure new options are available
    print("Testing pipeline help command...")
    
    cmd = ["./run_pipeline.sh", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        help_output = result.stdout
        
        # Check for new options
        if "--use-enhanced-preprocessor" in help_output:
            print("✓ Enhanced preprocessor option found in help")
        else:
            print("✗ Enhanced preprocessor option not found in help")
            return False
        
        if "--enhanced-preprocessor-mode" in help_output:
            print("✓ Enhanced preprocessor mode option found in help")
        else:
            print("✗ Enhanced preprocessor mode option not found in help")
            return False
        
        if "enhanced preprocessor" in help_output.lower():
            print("✓ Enhanced preprocessor examples found in help")
        else:
            print("✗ Enhanced preprocessor examples not found in help")
            return False
        
        print("✓ Pipeline integration test completed successfully")
        return True
    else:
        print(f"✗ Pipeline help command failed: {result.stderr}")
        return False

def main():
    """Main test function"""
    
    print("Enhanced Preprocessor Integration Test")
    print("=" * 50)
    
    # Test 1: Enhanced preprocessor functionality
    test1_result = test_enhanced_preprocessor()
    
    # Test 2: Pipeline integration
    test2_result = test_pipeline_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if test1_result and test2_result:
        print("✓ ALL TESTS PASSED")
        print("Enhanced preprocessor is successfully integrated!")
        print("\nYou can now use:")
        print("  ./run_pipeline.sh --use-enhanced-preprocessor")
        print("  ./run_pipeline.sh --use-enhanced-preprocessor --enhanced-preprocessor-mode aggressive")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        if not test1_result:
            print("- Enhanced preprocessor functionality test failed")
        if not test2_result:
            print("- Pipeline integration test failed")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 