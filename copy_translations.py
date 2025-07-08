import os
import shutil
import re

# --- 設定 ---
# 來源資料夾，存放原始音檔和翻譯 .txt 檔
SOURCE_DIR = '/media/meow/One Touch/ems_call/long_calls_filtered'

# 目標資料夾列表
TARGET_DIRS = [
    '/media/meow/One Touch/ems_call/random_samples_1',
    '/media/meow/One Touch/ems_call/random_samples_2'
]
# --- 結束設定 ---

def copy_translation_files():
    """
    遍歷目標資料夾中的 .wav 檔案，
    從來源資料夾的對應子資料夾中，找到所有相關的 .txt 翻譯檔案並複製過來。
    """
    print("開始複製翻譯檔案...")

    if not os.path.isdir(SOURCE_DIR):
        print(f"錯誤：來源資料夾不存在，請檢查路徑：{SOURCE_DIR}")
        return

    total_copied_count = 0

    for target_dir in TARGET_DIRS:
        if not os.path.isdir(target_dir):
            print(f"警告：目標資料夾不存在，略過：{target_dir}")
            continue

        print(f"\n正在處理資料夾：{target_dir}")
        
        try:
            target_filenames = os.listdir(target_dir)
        except OSError as e:
            print(f"錯誤：無法讀取資料夾 {target_dir} 的內容：{e}")
            continue
        
        copied_in_dir = 0
        for filename in target_filenames:
            # 只處理 .wav 檔案
            if not filename.lower().endswith('.wav'):
                continue

            # 從 .wav 檔案名稱中解析出基礎名稱和對應的來源子資料夾名稱
            # e.g., '202412010133-841696-14744_call_2.wav'
            base_name = os.path.splitext(filename)[0] # '202412010133-841696-14744_call_2'
            
            # The part like '202412010133-841696-14744' is the subdirectory name
            try:
                subdir_name = base_name.split('_call_')[0]
            except IndexError:
                # 如果檔名不符合預期格式，就略過
                continue

            source_subdir_path = os.path.join(SOURCE_DIR, subdir_name)

            if not os.path.isdir(source_subdir_path):
                # 找不到對應的來源子資料夾，略過
                continue

            # 現在遍歷來源子資料夾，找到所有相關的 .txt 檔案
            try:
                source_files = os.listdir(source_subdir_path)
            except OSError as e:
                print(f"  錯誤：無法讀取來源子資料夾 {source_subdir_path}：{e}")
                continue
            
            for source_filename in source_files:
                # 檢查 .txt 檔案是否與 .wav 檔案的基礎名稱相關
                # e.g., source: 'canary-1b_202412010133-841696-14744_call_2.txt'
                #       base_name: '202412010133-841696-14744_call_2'
                if base_name in source_filename and source_filename.lower().endswith('.txt'):
                    source_file_path = os.path.join(source_subdir_path, source_filename)
                    dest_file_path = os.path.join(target_dir, source_filename)
                    
                    try:
                        # print(f"  正在複製 {source_filename}...")
                        shutil.copy2(source_file_path, dest_file_path)
                        copied_in_dir += 1
                        total_copied_count += 1
                    except shutil.Error as e:
                        print(f"  錯誤：複製檔案 {source_filename} 時發生錯誤：{e}")
                    except IOError as e:
                        print(f"  錯誤：檔案 I/O 錯誤：{e}")

        if copied_in_dir == 0:
            print("  此資料夾中沒有找到或複製任何翻譯檔案。")
        else:
            print(f"  成功複製 {copied_in_dir} 個檔案到此資料夾。")


    print(f"\n所有操作完成！總共複製了 {total_copied_count} 個檔案。")

if __name__ == "__main__":
    copy_translation_files() 