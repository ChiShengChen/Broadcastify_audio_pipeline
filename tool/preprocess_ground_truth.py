#!/usr/bin/env python3
"""
Ground Truth Preprocessing Script
Converts special characters and numbers to ASR-friendly format
"""

import pandas as pd
import re
import argparse
import os
from typing import Dict, Any

def normalize_numbers(text: str) -> str:
    """
    Convert numbers to word format for better ASR matching
    """
    # Convert digits to words
    number_mapping = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    
    # Handle time formats like 11:17, 20:28
    text = re.sub(r'(\d{1,2}):(\d{2})', lambda m: f"{m.group(1)} {m.group(2)}", text)
    
    # Handle phone numbers and codes (e.g., 612, 6-1-2)
    text = re.sub(r'(\d)-(\d)-(\d)', lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}", text)
    
    # Handle addresses with numbers (e.g., 4560, 132)
    text = re.sub(r'(\d{3,4})', lambda m: ' '.join([number_mapping[d] for d in m.group(1)]), text)
    
    # Handle single digits in context (but not in addresses)
    text = re.sub(r'\b(\d)\b', lambda m: number_mapping[m.group(1)], text)
    
    return text

def normalize_special_characters(text: str) -> str:
    """
    Convert special characters to ASR-friendly format
    """
    # Replace special characters with words or remove them
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

def normalize_common_abbreviations(text: str) -> str:
    """
    Convert common abbreviations to full words
    """
    abbreviations = {
        'PD': 'police department',
        'EMS': 'emergency medical services',
        'BLS': 'basic life support',
        'ALS': 'advanced life support',
        'CPR': 'cardiopulmonary resuscitation',
        'O2': 'oxygen',
        'CO': 'carbon monoxide',
        'CPR': 'cardiopulmonary resuscitation',
        'N/A': 'not applicable',
        'ETA': 'estimated time of arrival',
        'DOA': 'dead on arrival',
        'V/S': 'vital signs',
        'BP': 'blood pressure',
        'HR': 'heart rate',
        'RR': 'respiratory rate',
        'O2': 'oxygen',
        'CO2': 'carbon dioxide',
        'IV': 'intravenous',
        'IM': 'intramuscular',
        'PO': 'by mouth',
        'PRN': 'as needed',
        'QID': 'four times daily',
        'TID': 'three times daily',
        'BID': 'twice daily',
        'QD': 'once daily',
        'STAT': 'immediately',
        'ASAP': 'as soon as possible',
        'PRN': 'as needed',
        'NPO': 'nothing by mouth',
        'C/O': 'complains of',
        'H/O': 'history of',
        'P/O': 'post operative',
        'S/P': 'status post',
        'W/': 'with',
        'W/O': 'without',
        'H/O': 'history of',
        'P/O': 'post operative',
        'S/P': 'status post',
        'C/O': 'complains of',
        'NPO': 'nothing by mouth',
        'PRN': 'as needed',
        'STAT': 'immediately',
        'ASAP': 'as soon as possible',
        'ETA': 'estimated time of arrival',
        'DOA': 'dead on arrival',
        'V/S': 'vital signs',
        'BP': 'blood pressure',
        'HR': 'heart rate',
        'RR': 'respiratory rate',
        'O2': 'oxygen',
        'CO2': 'carbon dioxide',
        'IV': 'intravenous',
        'IM': 'intramuscular',
        'PO': 'by mouth',
        'PRN': 'as needed',
        'QID': 'four times daily',
        'TID': 'three times daily',
        'BID': 'twice daily',
        'QD': 'once daily',
        'STAT': 'immediately',
        'ASAP': 'as soon as possible',
        'PRN': 'as needed',
        'NPO': 'nothing by mouth',
        'C/O': 'complains of',
        'H/O': 'history of',
        'P/O': 'post operative',
        'S/P': 'status post',
        'W/': 'with',
        'W/O': 'without',
    }
    
    # Apply abbreviations (case insensitive)
    for abbr, full in abbreviations.items():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)
    
    return text

def normalize_emergency_terms(text: str) -> str:
    """
    Normalize emergency service specific terms
    """
    emergency_terms = {
        'engine': 'engine',
        'ladder': 'ladder',
        'rescue': 'rescue',
        'ambulance': 'ambulance',
        'battalion': 'battalion',
        'command': 'command',
        'scene': 'scene',
        'patient': 'patient',
        'victim': 'victim',
        'casualty': 'casualty',
        'injury': 'injury',
        'illness': 'illness',
        'emergency': 'emergency',
        'fire': 'fire',
        'smoke': 'smoke',
        'alarm': 'alarm',
        'dispatch': 'dispatch',
        'respond': 'respond',
        'arrive': 'arrive',
        'clear': 'clear',
        'copy': 'copy',
        'received': 'received',
        'roger': 'roger',
        'affirmative': 'affirmative',
        'negative': 'negative',
        'over': 'over',
        'out': 'out',
        'standby': 'standby',
        'en route': 'en route',
        'on scene': 'on scene',
        'available': 'available',
        'unavailable': 'unavailable',
        'busy': 'busy',
        'code': 'code',
        'priority': 'priority',
        'urgent': 'urgent',
        'critical': 'critical',
        'stable': 'stable',
        'unstable': 'unstable',
        'conscious': 'conscious',
        'unconscious': 'unconscious',
        'breathing': 'breathing',
        'not breathing': 'not breathing',
        'pulse': 'pulse',
        'no pulse': 'no pulse',
        'bleeding': 'bleeding',
        'chest pain': 'chest pain',
        'shortness of breath': 'shortness of breath',
        'difficulty breathing': 'difficulty breathing',
        'cardiac arrest': 'cardiac arrest',
        'heart attack': 'heart attack',
        'stroke': 'stroke',
        'seizure': 'seizure',
        'diabetic': 'diabetic',
        'overdose': 'overdose',
        'trauma': 'trauma',
        'fall': 'fall',
        'motor vehicle accident': 'motor vehicle accident',
        'MVA': 'motor vehicle accident',
        'car accident': 'car accident',
        'pedestrian': 'pedestrian',
        'gunshot': 'gunshot',
        'stabbing': 'stabbing',
        'assault': 'assault',
        'domestic': 'domestic',
        'suicide': 'suicide',
        'homicide': 'homicide',
        'overdose': 'overdose',
        'drug': 'drug',
        'alcohol': 'alcohol',
        'intoxicated': 'intoxicated',
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
    }
    
    # Apply emergency terms (case insensitive)
    for term, normalized in emergency_terms.items():
        pattern = r'\b' + re.escape(term) + r'\b'
        text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
    
    return text

def clean_text(text: str) -> str:
    """
    Clean and normalize text for ASR evaluation
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Apply all normalizations
    text = normalize_special_characters(text)
    text = normalize_numbers(text)
    text = normalize_common_abbreviations(text)
    text = normalize_emergency_terms(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def preprocess_ground_truth(input_file: str, output_file: str, backup_original: bool = True) -> Dict[str, Any]:
    """
    Preprocess ground truth CSV file
    """
    print(f"Loading ground truth file: {input_file}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)
        
        # Check required columns
        if 'Filename' not in df.columns or 'transcript' not in df.columns:
            raise ValueError("CSV must contain 'Filename' and 'transcript' columns")
        
        # Create backup if requested
        if backup_original:
            backup_file = input_file + '.backup'
            df.to_csv(backup_file, index=False)
            print(f"Created backup: {backup_file}")
        
        # Store original transcript for comparison
        df['original_transcript'] = df['transcript'].copy()
        
        # Preprocess transcripts
        print("Preprocessing transcripts...")
        processed_count = 0
        total_count = len(df)
        
        for idx, row in df.iterrows():
            original_text = str(row['transcript'])
            processed_text = clean_text(original_text)
            
            if processed_text != original_text:
                processed_count += 1
                print(f"  Processed {row['Filename']}: {len(original_text)} -> {len(processed_text)} chars")
            
            df.at[idx, 'transcript'] = processed_text
        
        # Save processed file
        df.to_csv(output_file, index=False)
        
        print(f"Preprocessing completed:")
        print(f"  Total files: {total_count}")
        print(f"  Processed files: {processed_count}")
        print(f"  Output file: {output_file}")
        
        return {
            'total_files': total_count,
            'processed_files': processed_count,
            'output_file': output_file,
            'backup_file': backup_file if backup_original else None
        }
        
    except Exception as e:
        print(f"Error preprocessing ground truth: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Preprocess ground truth CSV for ASR evaluation")
    parser.add_argument("--input_file", required=True, help="Input ground truth CSV file")
    parser.add_argument("--output_file", required=True, help="Output processed CSV file")
    parser.add_argument("--no_backup", action="store_true", help="Don't create backup of original file")
    parser.add_argument("--preview", action="store_true", help="Show preview of changes without saving")
    
    args = parser.parse_args()
    
    if args.preview:
        # Preview mode - show changes without saving
        print("=== PREVIEW MODE ===")
        df = pd.read_csv(args.input_file)
        
        print(f"Preview of changes for first 3 files:")
        for idx, row in df.head(3).iterrows():
            original = str(row['transcript'])
            processed = clean_text(original)
            
            print(f"\n{row['Filename']}:")
            print(f"  Original: {original[:100]}...")
            print(f"  Processed: {processed[:100]}...")
            if original != processed:
                print(f"  ✓ Changed")
            else:
                print(f"  - No changes")
    else:
        # Process the file
        result = preprocess_ground_truth(
            args.input_file, 
            args.output_file, 
            backup_original=not args.no_backup
        )
        print("Preprocessing completed successfully!")

if __name__ == '__main__':
    main() 