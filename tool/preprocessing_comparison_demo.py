#!/usr/bin/env python3
"""
演示不同預處理設定的效果差異
"""

import tempfile
import subprocess
import os
from pathlib import Path

def create_test_data():
    """創建測試數據"""
    test_content = '''Filename,transcript,original_files,duration,group_id
test_001.wav,"[x] Arriving on scene, I'll assume command as [x] on the outside. EMS 4 en route to cardiac arrest call 5-0-1-2 newson road. Patient is 95 year old male. BLS needed for breathing difficulty priority 1. Engine 7 to command, attic is clear. Engine 7 can handle. 10-4. We're going to handle. All units clear. [x] rescue 2 is clear. Command to ladder 2, status of [x]. [x] Copy. Comments stated there's possibly a occupant on oxygen interior. 19:20 [x] possibly one more individual in [x] over by where we're parked. We're making patient contact. [x] just stand by for a second. Will do. [x] roll call for 1541 winter road. Engine 2. Engine 2. Ladder 2. Engine 4. Engine 7. Engine 7. Rescue 2. Rescue 2. Battalion 5. 1. 1. [x] engine 4. EMS 4. EMS 4. [x] copy all units respond with engine 4.",['test_file.wav'],230.325875,1
test_002.wav,"Rescue 8, 1424P can you respond to an illness 612 6-1-2 jack rabbit road three shipps coffee at 11:17. Rescue 8. 1424P jack rabbit road [x] at 11:17. 3P for an unconscious at 1533 1-5-3-3 brookwood crescent [x] green run at 11:18. Rescue 15 1623P for an unconscious at 1-5-3-3 brookwood crescent. Rescue 9 923P for a bleed 4853 4-8-5-3 [x] road [x] extension at 11:18. Rescue 9 923P bleed 4-8-5-3 [x] road [x] extension at 11:18. 6-1-2 jack rabbit road. Three shipps coffee patient is going to be waiting out front [x] 48 year old male he is conscious and breathing at 11:19 starting roll call for 1117 [berkley] drive. Engine 19. 19. Engine 7. Engine 7. Engine 2. Respond for the overdose at 2501 2-5-0-1 james madison boulevard, at the jail, at 11:29. 621P copy right, have a 40 year old male, conscious breathing, possible overdose, narcan times two at 11:29. [x] ems 1. Ems 1. [x] scene working fire. [x]95 go ahead 19. Engine 9 is on scene. Looks like we got a garage detached garage in the back. Engine 19 to investigate, i'll pass command off to you. Copy, establishing command. [x] on scene. Engine 4 to 923, we got a spot right in front of the pick up truck for you. 73 1-6-7-3 gray friars chase at 12:36. Engine 22 to 1723, [x]. We have 1 BLS patient. When you guys get here, just pull up right in front of the white mistubishi and between the engine and the white car and we'll move [x] chief complaint headache. Will do. Ambulance 2, 220P, copy [x] to the choking 1-6-7-3 gray friars chase patient is 77 year old female conscious breathing [x] choking 12:37",['test_file2.wav'],289.8451875,2'''
    
    return test_content

def run_preprocessor(script_name, input_file, output_file, mode):
    """運行預處理器"""
    cmd = [
        "python3", script_name,
        "--input_file", input_file,
        "--output_file", output_file,
        "--mode", mode
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def read_processed_content(file_path):
    """讀取處理後的內容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) > 1:
            return lines[1].split(',', 1)[1].strip('"')
    return ""

def compare_preprocessing_methods():
    """比較不同的預處理方法"""
    
    print("=== 預處理方法對比演示 ===")
    print()
    
    # 創建測試數據
    test_content = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # 創建測試文件
        test_file = temp_dir_path / "test_ground_truth.csv"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print("原始文本示例：")
        print("-" * 50)
        original_text = test_content.split('\n')[1].split(',', 1)[1].strip('"')
        print(original_text[:200] + "...")
        print()
        
        # 測試 1: 基本預處理器（保守模式）
        print("1️⃣ 基本預處理器（保守模式）")
        print("-" * 50)
        basic_output = temp_dir_path / "basic_conservative.csv"
        
        if run_preprocessor("smart_preprocess_ground_truth.py", str(test_file), str(basic_output), "conservative"):
            basic_content = read_processed_content(basic_output)
            print("處理後文本：")
            print(basic_content[:200] + "...")
            print(f"字符數變化：{len(original_text)} → {len(basic_content)} (+{len(basic_content) - len(original_text)})")
        else:
            print("❌ 基本預處理器執行失敗")
        print()
        
        # 測試 2: 增強版預處理器（保守模式）
        print("2️⃣ 增強版預處理器（保守模式）")
        print("-" * 50)
        enhanced_conservative_output = temp_dir_path / "enhanced_conservative.csv"
        
        if run_preprocessor("enhanced_ground_truth_preprocessor.py", str(test_file), str(enhanced_conservative_output), "conservative"):
            enhanced_conservative_content = read_processed_content(enhanced_conservative_output)
            print("處理後文本：")
            print(enhanced_conservative_content[:200] + "...")
            print(f"字符數變化：{len(original_text)} → {len(enhanced_conservative_content)} (+{len(enhanced_conservative_content) - len(original_text)})")
        else:
            print("❌ 增強版預處理器（保守）執行失敗")
        print()
        
        # 測試 3: 增強版預處理器（激進模式）
        print("3️⃣ 增強版預處理器（激進模式）")
        print("-" * 50)
        enhanced_aggressive_output = temp_dir_path / "enhanced_aggressive.csv"
        
        if run_preprocessor("enhanced_ground_truth_preprocessor.py", str(test_file), str(enhanced_aggressive_output), "aggressive"):
            enhanced_aggressive_content = read_processed_content(enhanced_aggressive_output)
            print("處理後文本：")
            print(enhanced_aggressive_content[:200] + "...")
            print(f"字符數變化：{len(original_text)} → {len(enhanced_aggressive_content)} (+{len(enhanced_aggressive_content) - len(original_text)})")
        else:
            print("❌ 增強版預處理器（激進）執行失敗")
        print()
        
        # 詳細對比
        print("📊 詳細對比分析")
        print("=" * 60)
        
        # 檢查特定變化的差異
        print("特定變化檢查：")
        
        # 檢查 [x] 標記
        print(f"  [x] 標記處理：")
        print(f"    原始：{original_text.count('[x]')} 個")
        print(f"    基本預處理器：{basic_content.count('[x]')} 個")
        print(f"    增強版（保守）：{enhanced_conservative_content.count('[x]')} 個")
        print(f"    增強版（激進）：{enhanced_aggressive_content.count('[x]')} 個")
        
        # 檢查縮寫
        print(f"  EMS 縮寫處理：")
        print(f"    原始：{'EMS' in original_text}")
        print(f"    基本預處理器：{'emergency medical services' in basic_content.lower()}")
        print(f"    增強版（保守）：{'emergency medical services' in enhanced_conservative_content.lower()}")
        print(f"    增強版（激進）：{'emergency medical services' in enhanced_aggressive_content.lower()}")
        
        # 檢查縮寫形式
        print(f"  縮寫形式處理：")
        print(f"    原始：{'I\\'ll' in original_text}")
        print(f"    基本預處理器：{'i will' in basic_content.lower()}")
        print(f"    增強版（保守）：{'i will' in enhanced_conservative_content.lower()}")
        print(f"    增強版（激進）：{'i will' in enhanced_aggressive_content.lower()}")
        
        # 檢查數字
        print(f"  數字處理：")
        print(f"    原始：{'5-0-1-2' in original_text}")
        print(f"    基本預處理器：{'5 0 1 2' in basic_content}")
        print(f"    增強版（保守）：{'5 0 1 2' in enhanced_conservative_content}")
        print(f"    增強版（激進）：{'five zero one two' in enhanced_aggressive_content.lower()}")
        
        print()
        print("🎯 使用建議：")
        print("  • 基本預處理器：適合簡單的文本清理")
        print("  • 增強版（保守）：適合需要保持專業術語準確性的場景")
        print("  • 增強版（激進）：適合需要最大化 ASR 匹配率的場景")

def main():
    """主函數"""
    compare_preprocessing_methods()

if __name__ == '__main__':
    main() 