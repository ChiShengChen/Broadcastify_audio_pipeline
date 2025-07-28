#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Comprehensive Tests
=======================

Complete test suite for ASR pipeline:
1. Create test dataset
2. Run unit tests
3. Diagnose limitations
4. Generate comprehensive report
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully ({duration:.2f}s)")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ {description} failed ({duration:.2f}s)")
            print("Error:")
            print(result.stderr)
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def create_test_dataset():
    """Create test dataset"""
    return run_command(
        [sys.executable, "unit_test/create_test_dataset.py"],
        "Create Test Dataset"
    )

def run_unit_tests():
    """Run unit tests"""
    return run_command(
        [sys.executable, "-m", "pytest", "unit_test/test_pipeline_components.py", "-v"],
        "Run Unit Tests"
    )

def diagnose_limitations():
    """Diagnose pipeline limitations"""
    return run_command(
        [sys.executable, "unit_test/diagnose_pipeline_limitations.py"],
        "Diagnose Pipeline Limitations"
    )

def run_vad_diagnosis():
    """Run VAD diagnosis"""
    return run_command(
        [sys.executable, "unit_test/vad_diagnosis.py"],
        "VAD Diagnosis"
    )

def run_simple_asr_test():
    """Run simple ASR test"""
    return run_command(
        [sys.executable, "unit_test/simple_asr_test.py"],
        "Simple ASR Test"
    )

def generate_summary_report():
    """Generate summary report"""
    print(f"\n{'='*50}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*50}")
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tests_run': [],
        'results': {},
        'recommendations': []
    }
    
    # Check for test results
    test_files = [
        "test_dataset",
        "pipeline_limitations_report.json",
        "fixed_vad_config.json"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            if os.path.isdir(test_file):
                file_count = len([f for f in os.listdir(test_file) if f.endswith('.wav')])
                summary['results'][test_file] = f"Directory with {file_count} test files"
            else:
                summary['results'][test_file] = "File exists"
        else:
            summary['results'][test_file] = "Not found"
    
    # Generate recommendations based on results
    if "test_dataset" in summary['results']:
        summary['recommendations'].append("Test dataset created successfully")
    
    if "pipeline_limitations_report.json" in summary['results']:
        summary['recommendations'].append("Pipeline limitations analyzed")
    
    if "fixed_vad_config.json" in summary['results']:
        summary['recommendations'].append("VAD configuration optimized")
    
    # Save summary
    summary_file = "comprehensive_test_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary report saved to: {summary_file}")
    
    # Print summary
    print(f"\nTest Summary:")
    for test_name, result in summary['results'].items():
        print(f"  {test_name}: {result}")
    
    print(f"\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  - {rec}")
    
    return summary

def main():
    """Main function to run comprehensive tests"""
    print("Run Comprehensive Tests")
    print("=" * 50)
    
    # Track test results
    test_results = {}
    
    try:
        # Step 1: Create test dataset
        test_results['create_dataset'] = create_test_dataset()
        
        # Step 2: Run unit tests
        test_results['unit_tests'] = run_unit_tests()
        
        # Step 3: Run VAD diagnosis
        test_results['vad_diagnosis'] = run_vad_diagnosis()
        
        # Step 4: Run simple ASR test
        test_results['simple_asr_test'] = run_simple_asr_test()
        
        # Step 5: Diagnose limitations
        test_results['limitations_diagnosis'] = diagnose_limitations()
        
        # Step 6: Generate summary
        summary = generate_summary_report()
        
        # Final results
        successful_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        
        print(f"\n{'='*50}")
        print("FINAL RESULTS")
        print(f"{'='*50}")
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        print(f"\nOverall: {successful_tests}/{total_tests} tests passed")
        
        if successful_tests == total_tests:
            print("🎉 All tests completed successfully!")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")
        
        # Save test results
        results_file = "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_results': test_results,
                'summary': summary,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 