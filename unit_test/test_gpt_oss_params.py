#!/usr/bin/env python3
"""
测试 gpt-oss-20b 参数传递
"""

import sys
import subprocess

def test_parameter_passing():
    """测试参数传递"""
    print("Testing gpt-oss-20b parameter passing...")
    
    # 测试参数传递
    cmd = [
        "python", "llm_gpt_oss_20b.py",
        "/media/meow/One Touch/ems_call/llm_results_20250819_220855/whisper_filtered",
        "/media/meow/One Touch/ems_call/test_params",
        "Test prompt",
        "0.3",  # temperature
        "64"    # max_new_tokens
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print("Command executed successfully")
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print("Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        if result.stderr:
            print("Error:", result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
    except subprocess.TimeoutExpired:
        print("Command timed out (expected for model loading)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_parameter_passing() 