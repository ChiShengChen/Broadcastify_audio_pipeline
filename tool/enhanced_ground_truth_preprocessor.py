#!/usr/bin/env python3
"""
Enhanced Ground Truth Preprocessor
涵蓋所有重要文字預處理項目的增強版預處理腳本
"""

import pandas as pd
import re
import argparse
import os
from typing import Dict, Any, List

class EnhancedTextPreprocessor:
    def __init__(self, mode: str = 'conservative'):
        self.mode = mode
        self.setup_patterns()
    
    def setup_patterns(self):
        """設置所有預處理模式"""
        
        # 1. 縮寫 (Abbreviations) - 高優先級
        self.abbreviations = {
            'PD': 'police department',
            'EMS': 'emergency medical services',
            'BLS': 'basic life support',
            'ALS': 'advanced life support',
            'CPR': 'cardiopulmonary resuscitation',
            'O2': 'oxygen',
            'CO': 'carbon monoxide',
            'CO2': 'carbon dioxide',
            'N/A': 'not applicable',
            'ETA': 'estimated time of arrival',
            'DOA': 'dead on arrival',
            'V/S': 'vital signs',
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'RR': 'respiratory rate',
            'IV': 'intravenous',
            'IM': 'intramuscular',
            'PO': 'by mouth',
            'PRN': 'as needed',
            'STAT': 'immediately',
            'ASAP': 'as soon as possible',
            'NPO': 'nothing by mouth',
            'C/O': 'complains of',
            'H/O': 'history of',
            'P/O': 'post operative',
            'S/P': 'status post',
            'W/': 'with',
            'W/O': 'without',
            'QID': 'four times daily',
            'TID': 'three times daily',
            'BID': 'twice daily',
            'QD': 'once daily',
            'MVA': 'motor vehicle accident',
        }
        
        # 2. 醫療術語標準化 (Medical Terms) - 中優先級
        self.medical_terms = {
            'conscious': 'conscious',
            'unconscious': 'unconscious',
            'breathing': 'breathing',
            'not breathing': 'not breathing',
            'pulse': 'pulse',
            'no pulse': 'no pulse',
            'cardiac arrest': 'cardiac arrest',
            'heart attack': 'heart attack',
            'stroke': 'stroke',
            'seizure': 'seizure',
            'diabetic': 'diabetic',
            'overdose': 'overdose',
            'trauma': 'trauma',
            'fall': 'fall',
            'motor vehicle accident': 'motor vehicle accident',
            'car accident': 'car accident',
            'bleeding': 'bleeding',
            'chest pain': 'chest pain',
            'shortness of breath': 'shortness of breath',
            'difficulty breathing': 'difficulty breathing',
            'pregnant': 'pregnant',
            'labor': 'labor',
            'delivery': 'delivery',
            'baby': 'baby',
            'child': 'child',
            'infant': 'infant',
            'elderly': 'elderly',
            'geriatric': 'geriatric',
            'pediatric': 'pediatric',
            'adult': 'adult',
            'male': 'male',
            'female': 'female',
            'unknown': 'unknown',
            'unidentified': 'unidentified',
            'patient': 'patient',
            'victim': 'victim',
            'casualty': 'casualty',
        }
        
        # 3. 單位識別符標準化 (Unit Identifiers) - 中優先級
        self.unit_identifiers = {
            'engine': 'engine',
            'ladder': 'ladder',
            'rescue': 'rescue',
            'ambulance': 'ambulance',
            'battalion': 'battalion',
            'command': 'command',
            'scene': 'scene',
            'available': 'available',
            'unavailable': 'unavailable',
            'busy': 'busy',
            'en route': 'en route',
            'on scene': 'on scene',
        }
        
        # 4. 位置術語標準化 (Location Terms) - 中優先級
        self.location_terms = {
            'road': 'road',
            'street': 'street',
            'drive': 'drive',
            'boulevard': 'boulevard',
            'avenue': 'avenue',
            'way': 'way',
            'circle': 'circle',
            'court': 'court',
            'apartment': 'apartment',
            'unit': 'unit',
            'room': 'room',
            'floor': 'floor',
            'building': 'building',
            'structure': 'structure',
            'north': 'north',
            'south': 'south',
            'east': 'east',
            'west': 'west',
            'northeast': 'northeast',
            'northwest': 'northwest',
            'southeast': 'southeast',
            'southwest': 'southwest',
        }
        
        # 5. 緊急代碼標準化 (Emergency Codes) - 高優先級
        self.emergency_codes = {
            '10-4': 'ten four',
            '10-20': 'ten twenty',
            '10-8': 'ten eight',
            '10-7': 'ten seven',
            '10-23': 'ten twenty three',
            '10-24': 'ten twenty four',
            '10-97': 'ten ninety seven',
            '10-98': 'ten ninety eight',
        }
        
        # 6. 技術術語標準化 (Technical Terms) - 低優先級
        self.technical_terms = {
            'water supply': 'water supply',
            'hose line': 'hose line',
            'ladder truck': 'ladder truck',
            'engine company': 'engine company',
            'rescue squad': 'rescue squad',
            'fire suppression': 'fire suppression',
            'ventilation': 'ventilation',
            'overhaul': 'overhaul',
            'salvage': 'salvage',
            'primary search': 'primary search',
            'secondary search': 'secondary search',
            'vent entry': 'vent entry',
            'roof operations': 'roof operations',
            'incident command': 'incident command',
            'command post': 'command post',
            'staging area': 'staging area',
            'rehab area': 'rehab area',
        }
        
        # 7. 音標拼寫標準化 (Phonetic Spellings) - 低優先級
        self.phonetic_spellings = {
            'alpha': 'alpha',
            'bravo': 'bravo',
            'charlie': 'charlie',
            'delta': 'delta',
            'echo': 'echo',
            'foxtrot': 'foxtrot',
            'golf': 'golf',
            'hotel': 'hotel',
            'india': 'india',
            'juliet': 'juliet',
            'kilo': 'kilo',
            'lima': 'lima',
            'mike': 'mike',
            'november': 'november',
            'oscar': 'oscar',
            'papa': 'papa',
            'quebec': 'quebec',
            'romeo': 'romeo',
            'sierra': 'sierra',
            'tango': 'tango',
            'uniform': 'uniform',
            'victor': 'victor',
            'whiskey': 'whiskey',
            'xray': 'xray',
            'yankee': 'yankee',
            'zulu': 'zulu',
        }
    
    def normalize_abbreviations(self, text: str) -> str:
        """標準化縮寫"""
        for abbr, full in self.abbreviations.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)
        return text
    
    def normalize_medical_terms(self, text: str) -> str:
        """標準化醫療術語"""
        for term, normalized in self.medical_terms.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
        return text
    
    def normalize_unit_identifiers(self, text: str) -> str:
        """標準化單位識別符"""
        for term, normalized in self.unit_identifiers.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
        return text
    
    def normalize_location_terms(self, text: str) -> str:
        """標準化位置術語"""
        for term, normalized in self.location_terms.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
        return text
    
    def normalize_emergency_codes(self, text: str) -> str:
        """標準化緊急代碼"""
        for code, normalized in self.emergency_codes.items():
            pattern = r'\b' + re.escape(code) + r'\b'
            text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
        return text
    
    def normalize_technical_terms(self, text: str) -> str:
        """標準化技術術語"""
        for term, normalized in self.technical_terms.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
        return text
    
    def normalize_phonetic_spellings(self, text: str) -> str:
        """標準化音標拼寫"""
        for spelling, normalized in self.phonetic_spellings.items():
            pattern = r'\b' + re.escape(spelling) + r'\b'
            text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
        return text
    
    def normalize_numbers_conservative(self, text: str) -> str:
        """保守的數字標準化"""
        # 處理時間格式
        text = re.sub(r'(\d{1,2}):(\d{2})', lambda m: f"{m.group(1)} {m.group(2)}", text)
        
        # 處理電話代碼
        text = re.sub(r'(\d)-(\d)-(\d)', lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}", text)
        
        # 只在特定上下文中轉換單個數字
        text = re.sub(r'\b(\d)\b(?=\s+(year|month|week|day|hour|minute|second|time|priority|code|unit|engine|ladder|rescue|ambulance|battalion))', 
                      lambda m: {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                               '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}[m.group(1)], text)
        
        return text
    
    def normalize_numbers_aggressive(self, text: str) -> str:
        """激進的數字標準化"""
        number_mapping = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        
        # 處理時間格式
        text = re.sub(r'(\d{1,2}):(\d{2})', lambda m: f"{m.group(1)} {m.group(2)}", text)
        
        # 處理電話代碼
        text = re.sub(r'(\d)-(\d)-(\d)', lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}", text)
        
        # 處理地址號碼
        text = re.sub(r'(\d{3,4})', lambda m: ' '.join([number_mapping[d] for d in m.group(1)]), text)
        
        # 處理單個數字
        text = re.sub(r'\b(\d)\b', lambda m: number_mapping[m.group(1)], text)
        
        return text
    
    def normalize_special_characters(self, text: str) -> str:
        """標準化特殊字符"""
        replacements = {
            '%': ' percent',
            '&': ' and',
            '[': '',
            ']': '',
            '(': '',
            ')': '',
            '{': '',
            '}': '',
            '<': ' less than ',
            '>': ' greater than ',
            '=': ' equals ',
            '+': ' plus ',
            '#': ' number ',
            '@': ' at ',
            '$': ' dollars ',
            '^': ' to the power of ',
            '|': ' or ',
            '\\': ' ',
            '/': ' slash ',
            '?': ' question mark ',
            '!': ' exclamation mark ',
            '~': ' approximately ',
            '`': ' ',
            '"': ' ',
            "'": ' ',
            ';': ' ',
            ':': ' ',
            ',': ' ',
            '.': ' ',
            '-': ' ',
            '_': ' ',
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def normalize_contractions(self, text: str) -> str:
        """標準化縮寫形式"""
        contractions = {
            "I'll": "I will",
            "I'm": "I am",
            "I've": "I have",
            "I'd": "I would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have",
            "you'd": "you would",
            "he'll": "he will",
            "he's": "he is",
            "he'd": "he would",
            "she'll": "she will",
            "she's": "she is",
            "she'd": "she would",
            "it'll": "it will",
            "it's": "it is",
            "it'd": "it would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "we'd": "we would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "they'd": "they would",
            "can't": "cannot",
            "couldn't": "could not",
            "won't": "will not",
            "wouldn't": "would not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
        }
        
        for contraction, full in contractions.items():
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)
        
        return text
    
    def remove_filler_words(self, text: str) -> str:
        """移除填充詞"""
        filler_words = [
            'um', 'uh', 'er', 'ah', 'hmm', 'well', 'like', 'you know', 'i mean',
            'so', 'basically', 'actually', 'literally', 'obviously'
        ]
        
        for word in filler_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """主要的文本清理函數"""
        if not isinstance(text, str):
            return ""
        
        # 轉換為小寫
        text = text.lower()
        
        # 高優先級處理
        text = self.normalize_abbreviations(text)
        text = self.normalize_emergency_codes(text)
        text = self.normalize_special_characters(text)
        
        # 中優先級處理
        text = self.normalize_medical_terms(text)
        text = self.normalize_unit_identifiers(text)
        text = self.normalize_location_terms(text)
        text = self.normalize_technical_terms(text)
        text = self.normalize_phonetic_spellings(text)
        
        # 數字處理
        if self.mode == 'aggressive':
            text = self.normalize_numbers_aggressive(text)
        else:
            text = self.normalize_numbers_conservative(text)
        
        # 低優先級處理
        if self.mode == 'aggressive':
            text = self.normalize_contractions(text)
            text = self.remove_filler_words(text)
        
        # 清理多餘空格
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

def preprocess_ground_truth(input_file: str, output_file: str, mode: str = 'conservative', backup_original: bool = True) -> Dict[str, Any]:
    """預處理 ground truth CSV 文件"""
    print(f"載入 ground truth 文件: {input_file}")
    print(f"預處理模式: {mode}")
    
    try:
        # 載入 CSV 文件
        df = pd.read_csv(input_file)
        
        # 檢查必要欄位
        if 'Filename' not in df.columns or 'transcript' not in df.columns:
            raise ValueError("CSV 必須包含 'Filename' 和 'transcript' 欄位")
        
        # 創建備份
        if backup_original:
            backup_file = input_file + '.backup'
            df.to_csv(backup_file, index=False)
            print(f"創建備份: {backup_file}")
        
        # 保存原始轉錄
        df['original_transcript'] = df['transcript'].copy()
        
        # 初始化預處理器
        preprocessor = EnhancedTextPreprocessor(mode)
        
        # 預處理轉錄
        print("預處理轉錄...")
        processed_count = 0
        total_count = len(df)
        
        for idx, row in df.iterrows():
            original_text = str(row['transcript'])
            processed_text = preprocessor.clean_text(original_text)
            
            if processed_text != original_text:
                processed_count += 1
                print(f"  處理 {row['Filename']}: {len(original_text)} -> {len(processed_text)} 字符")
            
            df.at[idx, 'transcript'] = processed_text
        
        # 保存處理後的文件
        df.to_csv(output_file, index=False)
        
        print(f"預處理完成:")
        print(f"  總文件數: {total_count}")
        print(f"  處理文件數: {processed_count}")
        print(f"  輸出文件: {output_file}")
        print(f"  模式: {mode}")
        
        return {
            'total_files': total_count,
            'processed_files': processed_count,
            'output_file': output_file,
            'backup_file': backup_file if backup_original else None,
            'mode': mode
        }
        
    except Exception as e:
        print(f"預處理 ground truth 時發生錯誤: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="增強版 ground truth 預處理器")
    parser.add_argument("--input_file", required=True, help="輸入 ground truth CSV 文件")
    parser.add_argument("--output_file", required=True, help="輸出處理後 CSV 文件")
    parser.add_argument("--mode", choices=['conservative', 'aggressive'], default='conservative', 
                       help="預處理模式: conservative (最小變更) 或 aggressive (全面變更)")
    parser.add_argument("--no_backup", action="store_true", help="不創建原始文件備份")
    parser.add_argument("--preview", action="store_true", help="預覽變更而不保存")
    
    args = parser.parse_args()
    
    if args.preview:
        # 預覽模式
        print("=== 預覽模式 ===")
        print(f"模式: {args.mode}")
        df = pd.read_csv(args.input_file)
        
        preprocessor = EnhancedTextPreprocessor(args.mode)
        
        print(f"前3個文件的變更預覽:")
        for idx, row in df.head(3).iterrows():
            original = str(row['transcript'])
            processed = preprocessor.clean_text(original)
            
            print(f"\n{row['Filename']}:")
            print(f"  原始: {original[:100]}...")
            print(f"  處理後: {processed[:100]}...")
            if original != processed:
                print(f"  ✓ 已變更")
            else:
                print(f"  - 無變更")
    else:
        # 處理文件
        result = preprocess_ground_truth(
            args.input_file, 
            args.output_file, 
            mode=args.mode,
            backup_original=not args.no_backup
        )
        print("預處理成功完成!")

if __name__ == '__main__':
    main() 