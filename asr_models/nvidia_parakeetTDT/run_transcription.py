#!/usr/bin/env python3
"""
簡單的 NVIDIA Parakeet TDT 英文轉錄執行腳本
"""

import sys
import os
from pathlib import Path

# 添加當前目錄到系統路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 導入轉錄模組
from parakeet_transcribe import main

if __name__ == "__main__":
    print("="*60)
    print("NVIDIA Parakeet TDT 0.6B v2 英文音頻轉錄工具")
    print("="*60)
    print()
    
    # 檢查是否有GPU
    import torch
    if torch.cuda.is_available():
        print(f"✓ 檢測到GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("⚠ 未檢測到GPU，將使用CPU（處理速度較慢）")
    
    print()
    print("模型資訊:")
    print("  - 使用模型: NVIDIA Parakeet TDT 0.6B v2")
    print("  - 模型類型: CTC (Connectionist Temporal Classification)")
    print("  - 語言支援: 英文")
    print("  - 模型大小: ~600M 參數")
    print("  - 專業領域: 通用語音識別")
    print()
    
    print("開始英文轉錄處理...")
    print("注意：首次運行時會下載模型文件，請耐心等待")
    print("語言設置：英文 (English)")
    print("音頻文件會被分割成30秒片段進行處理")
    print()
    
    # 執行主要轉錄程序
    main()
    
    print()
    print("轉錄程序執行完成！")
    print("轉錄結果已保存在各自的源文件夾中")
    print("文件名格式: parakeet-tdt-0.6b-v2_{原始文件名}.txt") 