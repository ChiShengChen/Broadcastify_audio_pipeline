#!/usr/bin/env python3
"""
簡單的英文轉錄執行腳本
"""

import sys
import os
from pathlib import Path

# 添加當前目錄到系統路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 導入轉錄模組
from canary_transcribe import main

if __name__ == "__main__":
    print("="*60)
    print("NVIDIA Canary-1B 英文音頻轉錄工具")
    print("="*60)
    print()
    
    # 檢查是否有GPU
    import torch
    if torch.cuda.is_available():
        print(f"✓ 檢測到GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠ 未檢測到GPU，將使用CPU（處理速度較慢）")
    
    print()
    print("開始英文轉錄處理...")
    print("注意：首次運行時會下載模型文件，請耐心等待")
    print("語言設置：英文 (English)")
    print()
    
    # 執行主要轉錄程序
    main()
    
    print()
    print("轉錄程序執行完成！")
    print("轉錄結果已保存在各自的源文件夾中") 