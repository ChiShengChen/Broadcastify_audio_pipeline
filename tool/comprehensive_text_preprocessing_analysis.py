#!/usr/bin/env python3
"""
Comprehensive Text Preprocessing Analysis
分析除了數字和特殊符號之外，還有哪些重要的文字預處理項目
"""

import os
import re
import pandas as pd
from collections import defaultdict, Counter
import argparse
from typing import Dict, List, Set, Tuple

def analyze_text_patterns(text: str) -> Dict[str, List[str]]:
    """
    分析文本中的各種模式
    """
    patterns = {
        'abbreviations': [],
        'medical_terms': [],
        'emergency_codes': [],
        'unit_identifiers': [],
        'location_terms': [],
        'time_expressions': [],
        'measurements': [],
        'contractions': [],
        'filler_words': [],
        'repetitions': [],
        'incomplete_sentences': [],
        'noise_markers': [],
        'phonetic_spellings': [],
        'technical_terms': [],
        'slang_terms': []
    }
    
    # 1. 縮寫 (Abbreviations)
    abbreviation_patterns = [
        r'\b[A-Z]{2,}\b',  # 大寫縮寫如 EMS, BLS, ALS
        r'\b[A-Z]\d+\b',   # 字母數字組合如 1424P, 923P
        r'\b[A-Z]+\d+[A-Z]+\b',  # 複雜縮寫
    ]
    
    for pattern in abbreviation_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            patterns['abbreviations'].append(match.group())
    
    # 2. 醫療術語 (Medical Terms)
    medical_patterns = [
        r'\b(conscious|unconscious|breathing|not breathing|pulse|no pulse)\b',
        r'\b(cardiac arrest|heart attack|stroke|seizure|diabetic|overdose)\b',
        r'\b(trauma|fall|motor vehicle accident|MVA|car accident)\b',
        r'\b(bleeding|chest pain|shortness of breath|difficulty breathing)\b',
        r'\b(pregnant|labor|delivery|baby|child|infant|elderly|geriatric)\b',
        r'\b(male|female|unknown|unidentified|patient|victim|casualty)\b',
    ]
    
    for pattern in medical_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['medical_terms'].append(match.group().lower())
    
    # 3. 緊急代碼 (Emergency Codes)
    code_patterns = [
        r'\b\d+-\d+-\d+\b',  # 電話代碼如 6-1-2
        r'\b\d{3,4}\b',      # 地址號碼
        r'\b\d{1,2}:\d{2}\b',  # 時間格式
    ]
    
    for pattern in code_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            patterns['emergency_codes'].append(match.group())
    
    # 4. 單位識別符 (Unit Identifiers)
    unit_patterns = [
        r'\b(engine|ladder|rescue|ambulance|battalion|command)\b',
        r'\b(engine \d+|ladder \d+|rescue \d+|ambulance \d+)\b',
        r'\b(battalion \d+|command \d+)\b',
    ]
    
    for pattern in unit_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['unit_identifiers'].append(match.group().lower())
    
    # 5. 位置術語 (Location Terms)
    location_patterns = [
        r'\b(scene|on scene|en route|available|unavailable|busy)\b',
        r'\b(road|street|drive|boulevard|avenue|way|circle|court)\b',
        r'\b(apartment|unit|room|floor|building|structure)\b',
        r'\b(north|south|east|west|northeast|northwest|southeast|southwest)\b',
    ]
    
    for pattern in location_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['location_terms'].append(match.group().lower())
    
    # 6. 時間表達 (Time Expressions)
    time_patterns = [
        r'\b(at \d{1,2}:\d{2})\b',
        r'\b(\d{1,2}:\d{2})\b',
        r'\b(now|currently|immediately|asap|stat)\b',
    ]
    
    for pattern in time_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['time_expressions'].append(match.group().lower())
    
    # 7. 測量單位 (Measurements)
    measurement_patterns = [
        r'\b(\d+ year old|\d+ years old)\b',
        r'\b(\d+ week|\d+ weeks)\b',
        r'\b(\d+ month|\d+ months)\b',
        r'\b(\d+ day|\d+ days)\b',
        r'\b(\d+ hour|\d+ hours)\b',
        r'\b(\d+ minute|\d+ minutes)\b',
    ]
    
    for pattern in measurement_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['measurements'].append(match.group().lower())
    
    # 8. 縮寫形式 (Contractions)
    contraction_patterns = [
        r'\b(I\'ll|I\'m|I\'ve|I\'d|you\'ll|you\'re|you\'ve|you\'d)\b',
        r'\b(he\'ll|he\'s|he\'d|she\'ll|she\'s|she\'d|it\'ll|it\'s|it\'d)\b',
        r'\b(we\'ll|we\'re|we\'ve|we\'d|they\'ll|they\'re|they\'ve|they\'d)\b',
        r'\b(can\'t|couldn\'t|won\'t|wouldn\'t|don\'t|doesn\'t|didn\'t)\b',
        r'\b(isn\'t|aren\'t|wasn\'t|weren\'t|hasn\'t|haven\'t|hadn\'t)\b',
    ]
    
    for pattern in contraction_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['contractions'].append(match.group().lower())
    
    # 9. 填充詞 (Filler Words)
    filler_patterns = [
        r'\b(um|uh|er|ah|hmm|well|like|you know|i mean)\b',
        r'\b(so|basically|actually|literally|obviously)\b',
    ]
    
    for pattern in filler_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['filler_words'].append(match.group().lower())
    
    # 10. 重複詞 (Repetitions)
    words = text.lower().split()
    word_counts = Counter(words)
    repetitions = [word for word, count in word_counts.items() if count > 2 and len(word) > 2]
    patterns['repetitions'] = repetitions[:10]  # 只取前10個
    
    # 11. 不完整句子 (Incomplete Sentences)
    incomplete_patterns = [
        r'\b(go ahead|copy|received|roger|affirmative|negative|over|out)\b',
        r'\b(standby|clear|available|unavailable|busy|en route|on scene)\b',
    ]
    
    for pattern in incomplete_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['incomplete_sentences'].append(match.group().lower())
    
    # 12. 噪音標記 (Noise Markers)
    noise_patterns = [
        r'\[x\]',  # [x] 標記
        r'\b(static|noise|interference|unclear|inaudible)\b',
    ]
    
    for pattern in noise_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['noise_markers'].append(match.group().lower())
    
    # 13. 音標拼寫 (Phonetic Spellings)
    phonetic_patterns = [
        r'\b(alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel)\b',
        r'\b(india|juliet|kilo|lima|mike|november|oscar|papa)\b',
        r'\b(quebec|romeo|sierra|tango|uniform|victor|whiskey|xray)\b',
        r'\b(yankee|zulu)\b',
    ]
    
    for pattern in phonetic_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['phonetic_spellings'].append(match.group().lower())
    
    # 14. 技術術語 (Technical Terms)
    technical_patterns = [
        r'\b(water supply|hose line|ladder truck|engine company|rescue squad)\b',
        r'\b(fire suppression|ventilation|overhaul|salvage|overhaul)\b',
        r'\b(primary search|secondary search|vent entry|roof operations)\b',
        r'\b(incident command|command post|staging area|rehab area)\b',
    ]
    
    for pattern in technical_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['technical_terms'].append(match.group().lower())
    
    # 15. 俚語術語 (Slang Terms)
    slang_patterns = [
        r'\b(10-4|10-20|10-8|10-7|10-23|10-24|10-97|10-98)\b',
        r'\b(copy|roger|affirmative|negative|over|out|standby)\b',
        r'\b(clear|available|unavailable|busy|en route|on scene)\b',
    ]
    
    for pattern in slang_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns['slang_terms'].append(match.group().lower())
    
    return patterns

def analyze_ground_truth_file(ground_truth_file: str) -> Dict[str, Dict]:
    """
    分析 ground truth 文件中的所有文本模式
    """
    print(f"分析 ground truth 文件: {ground_truth_file}")
    
    # 讀取 CSV 文件
    df = pd.read_csv(ground_truth_file)
    
    # 統計所有模式
    all_patterns = defaultdict(list)
    file_patterns = {}
    
    for idx, row in df.iterrows():
        filename = row['Filename']
        transcript = str(row['transcript'])
        
        # 分析單個文件的模式
        patterns = analyze_text_patterns(transcript)
        file_patterns[filename] = patterns
        
        # 收集所有模式
        for pattern_type, items in patterns.items():
            all_patterns[pattern_type].extend(items)
    
    # 統計頻率
    pattern_stats = {}
    for pattern_type, items in all_patterns.items():
        counter = Counter(items)
        pattern_stats[pattern_type] = {
            'total_occurrences': len(items),
            'unique_items': len(counter),
            'most_common': counter.most_common(10)
        }
    
    return {
        'file_patterns': file_patterns,
        'pattern_stats': pattern_stats,
        'total_files': len(df)
    }

def generate_preprocessing_recommendations(pattern_stats: Dict) -> Dict[str, List[str]]:
    """
    基於分析結果生成預處理建議
    """
    recommendations = {
        'high_priority': [],
        'medium_priority': [],
        'low_priority': [],
        'considerations': []
    }
    
    # 高優先級項目
    if pattern_stats['abbreviations']['total_occurrences'] > 0:
        recommendations['high_priority'].append(
            "縮寫標準化：處理 EMS、BLS、ALS、PD 等專業縮寫"
        )
    
    if pattern_stats['emergency_codes']['total_occurrences'] > 0:
        recommendations['high_priority'].append(
            "緊急代碼標準化：處理電話代碼、地址號碼、時間格式"
        )
    
    if pattern_stats['noise_markers']['total_occurrences'] > 0:
        recommendations['high_priority'].append(
            "噪音標記處理：移除或標準化 [x] 等標記"
        )
    
    # 中優先級項目
    if pattern_stats['contractions']['total_occurrences'] > 0:
        recommendations['medium_priority'].append(
            "縮寫形式標準化：處理 I'll、don't、can't 等"
        )
    
    if pattern_stats['unit_identifiers']['total_occurrences'] > 0:
        recommendations['medium_priority'].append(
            "單位識別符標準化：處理 engine、ladder、rescue 等"
        )
    
    if pattern_stats['medical_terms']['total_occurrences'] > 0:
        recommendations['medium_priority'].append(
            "醫療術語標準化：確保醫療術語的一致性"
        )
    
    # 低優先級項目
    if pattern_stats['filler_words']['total_occurrences'] > 0:
        recommendations['low_priority'].append(
            "填充詞處理：移除 um、uh、well 等填充詞"
        )
    
    if pattern_stats['repetitions']['total_occurrences'] > 0:
        recommendations['low_priority'].append(
            "重複詞處理：處理過度重複的詞彙"
        )
    
    # 考慮事項
    recommendations['considerations'].extend([
        "保持專業術語的準確性",
        "考慮上下文的重要性",
        "平衡標準化與語義保持",
        "根據 ASR 模型特性調整預處理策略"
    ])
    
    return recommendations

def print_analysis_report(analysis_results: Dict, output_file: str = None):
    """
    打印分析報告
    """
    report_lines = []
    
    report_lines.append("=" * 60)
    report_lines.append("全面文字預處理分析報告")
    report_lines.append("=" * 60)
    report_lines.append(f"分析文件數量: {analysis_results['total_files']}")
    report_lines.append("")
    
    # 模式統計
    report_lines.append("模式統計:")
    report_lines.append("-" * 30)
    
    for pattern_type, stats in analysis_results['pattern_stats'].items():
        if stats['total_occurrences'] > 0:
            report_lines.append(f"\n{pattern_type.upper()}:")
            report_lines.append(f"  總出現次數: {stats['total_occurrences']}")
            report_lines.append(f"  唯一項目數: {stats['unique_items']}")
            report_lines.append(f"  最常見項目:")
            for item, count in stats['most_common'][:5]:
                report_lines.append(f"    - {item}: {count} 次")
    
    # 生成建議
    recommendations = generate_preprocessing_recommendations(analysis_results['pattern_stats'])
    
    report_lines.append("\n" + "=" * 60)
    report_lines.append("預處理建議")
    report_lines.append("=" * 60)
    
    report_lines.append("\n高優先級項目:")
    for rec in recommendations['high_priority']:
        report_lines.append(f"  ✓ {rec}")
    
    report_lines.append("\n中優先級項目:")
    for rec in recommendations['medium_priority']:
        report_lines.append(f"  ○ {rec}")
    
    report_lines.append("\n低優先級項目:")
    for rec in recommendations['low_priority']:
        report_lines.append(f"  - {rec}")
    
    report_lines.append("\n考慮事項:")
    for rec in recommendations['considerations']:
        report_lines.append(f"  • {rec}")
    
    # 詳細示例
    report_lines.append("\n" + "=" * 60)
    report_lines.append("詳細示例")
    report_lines.append("=" * 60)
    
    # 顯示前3個文件的詳細分析
    for i, (filename, patterns) in enumerate(list(analysis_results['file_patterns'].items())[:3]):
        report_lines.append(f"\n文件 {i+1}: {filename}")
        report_lines.append("-" * 40)
        
        for pattern_type, items in patterns.items():
            if items:
                report_lines.append(f"  {pattern_type}: {', '.join(set(items[:5]))}")
                if len(items) > 5:
                    report_lines.append(f"    ... 還有 {len(items) - 5} 個項目")
    
    # 輸出報告
    report_text = '\n'.join(report_lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"報告已保存到: {output_file}")
    else:
        print(report_text)
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description="全面文字預處理分析")
    parser.add_argument("--ground_truth_file", required=True, help="Ground truth CSV 文件路徑")
    parser.add_argument("--output_file", help="輸出報告文件路徑")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ground_truth_file):
        print(f"錯誤: 找不到文件 {args.ground_truth_file}")
        return
    
    # 執行分析
    analysis_results = analyze_ground_truth_file(args.ground_truth_file)
    
    # 生成報告
    print_analysis_report(analysis_results, args.output_file)

if __name__ == '__main__':
    main() 