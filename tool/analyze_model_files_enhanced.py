#!/usr/bin/env python3
"""
Enhanced Model File Analysis with Error Logging
Analyzes ASR model output files and logs any errors encountered
"""

import os
import glob
import pandas as pd
import argparse
import traceback
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class ErrorLogger:
    """Error logging utility"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.errors = []
        self.warnings = []
        
    def log_error(self, error_type: str, message: str, details: str = "", file_path: str = ""):
        """Log an error"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': message,
            'details': details,
            'file_path': file_path
        }
        self.errors.append(error_entry)
        
        # Write to log file immediately
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[ERROR] {error_entry['timestamp']} - {error_type}: {message}\n")
            if details:
                f.write(f"  Details: {details}\n")
            if file_path:
                f.write(f"  File: {file_path}\n")
            f.write("\n")
    
    def log_warning(self, warning_type: str, message: str, details: str = "", file_path: str = ""):
        """Log a warning"""
        warning_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': warning_type,
            'message': message,
            'details': details,
            'file_path': file_path
        }
        self.warnings.append(warning_entry)
        
        # Write to log file immediately
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[WARNING] {warning_entry['timestamp']} - {warning_type}: {message}\n")
            if details:
                f.write(f"  Details: {details}\n")
            if file_path:
                f.write(f"  File: {file_path}\n")
            f.write("\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'error_types': {},
            'warning_types': {}
        }

class EnhancedModelFileAnalyzer:
    """Enhanced model file analyzer with error handling"""
    
    def __init__(self, transcript_dir: str, ground_truth_file: str, error_logger: ErrorLogger):
        self.transcript_dir = transcript_dir
        self.ground_truth_file = ground_truth_file
        self.error_logger = error_logger
        self.ground_truth_data = None
        self.analysis_results = {}
        
    def load_ground_truth(self) -> bool:
        """Load ground truth data with error handling"""
        try:
            if not os.path.exists(self.ground_truth_file):
                self.error_logger.log_error(
                    "FILE_NOT_FOUND",
                    f"Ground truth file not found: {self.ground_truth_file}",
                    "The ground truth file is required for analysis"
                )
                return False
            
            self.ground_truth_data = pd.read_csv(self.ground_truth_file)
            
            # Validate required columns
            required_columns = ['Filename', 'transcript']
            missing_columns = [col for col in required_columns if col not in self.ground_truth_data.columns]
            
            if missing_columns:
                self.error_logger.log_error(
                    "INVALID_FORMAT",
                    f"Missing required columns in ground truth file: {missing_columns}",
                    f"Found columns: {list(self.ground_truth_data.columns)}"
                )
                return False
            
            # Check for empty or invalid data
            if self.ground_truth_data.empty:
                self.error_logger.log_error(
                    "EMPTY_DATA",
                    "Ground truth file is empty",
                    "No data found in the CSV file"
                )
                return False
            
            # Check for missing values
            missing_filenames = self.ground_truth_data['Filename'].isna().sum()
            missing_transcripts = self.ground_truth_data['transcript'].isna().sum()
            
            if missing_filenames > 0:
                self.error_logger.log_warning(
                    "MISSING_DATA",
                    f"Found {missing_filenames} missing filenames in ground truth",
                    "These entries will be skipped during analysis"
                )
            
            if missing_transcripts > 0:
                self.error_logger.log_warning(
                    "MISSING_DATA",
                    f"Found {missing_transcripts} missing transcripts in ground truth",
                    "These entries will be skipped during analysis"
                )
            
            # Remove rows with missing data
            self.ground_truth_data = self.ground_truth_data.dropna(subset=['Filename', 'transcript'])
            
            return True
            
        except Exception as e:
            self.error_logger.log_error(
                "LOAD_ERROR",
                f"Failed to load ground truth file: {str(e)}",
                traceback.format_exc(),
                self.ground_truth_file
            )
            return False
    
    def analyze_transcript_file(self, file_path: str, model_name: str) -> Dict[str, Any]:
        """Analyze a single transcript file with error handling"""
        result = {
            'file_path': file_path,
            'model_name': model_name,
            'status': 'unknown',
            'file_size': 0,
            'content_length': 0,
            'encoding': 'unknown',
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result['status'] = 'missing'
                result['errors'].append('File not found')
                self.error_logger.log_error(
                    "FILE_NOT_FOUND",
                    f"Transcript file not found: {file_path}",
                    f"Model: {model_name}"
                )
                return result
            
            # Get file size
            file_size = os.path.getsize(file_path)
            result['file_size'] = file_size
            
            if file_size == 0:
                result['status'] = 'empty'
                result['warnings'].append('File is empty')
                self.error_logger.log_warning(
                    "EMPTY_FILE",
                    f"Empty transcript file: {file_path}",
                    f"Model: {model_name}, Size: {file_size} bytes"
                )
                return result
            
            # Try to read file with different encodings
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            used_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        used_encoding = encoding
                        break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                result['status'] = 'encoding_error'
                result['errors'].append('Failed to decode file with any encoding')
                self.error_logger.log_error(
                    "ENCODING_ERROR",
                    f"Failed to decode transcript file: {file_path}",
                    f"Model: {model_name}, Tried encodings: {encodings_to_try}",
                    file_path
                )
                return result
            
            result['encoding'] = used_encoding
            result['content_length'] = len(content)
            
            # Check for empty content after reading
            if not content.strip():
                result['status'] = 'empty_content'
                result['warnings'].append('File contains only whitespace')
                self.error_logger.log_warning(
                    "EMPTY_CONTENT",
                    f"Transcript file contains only whitespace: {file_path}",
                    f"Model: {model_name}, Encoding: {used_encoding}"
                )
                return result
            
            # Check for suspicious content patterns
            if len(content) < 10:
                result['warnings'].append('Content seems too short')
                self.error_logger.log_warning(
                    "SHORT_CONTENT",
                    f"Transcript content seems too short: {file_path}",
                    f"Model: {model_name}, Length: {len(content)} characters"
                )
            
            # Check for common error patterns in content
            # Skip this check for summary files
            filename = os.path.basename(file_path)
            if filename not in ['merging_summary.txt', 'summary.txt', 'processing_summary.txt']:
                error_patterns = [
                    ('ERROR', 'Contains "ERROR" in content'),
                    ('Exception', 'Contains "Exception" in content'),
                    ('Traceback', 'Contains "Traceback" in content'),
                    ('Failed', 'Contains "Failed" in content'),
                    ('null', 'Contains "null" in content'),
                    ('undefined', 'Contains "undefined" in content'),
                    ('<empty>', 'Contains "<empty>" in content'),
                    ('<error>', 'Contains "<error>" in content')
                ]
                
                for pattern, description in error_patterns:
                    if pattern.lower() in content.lower():
                        result['warnings'].append(description)
                        self.error_logger.log_warning(
                            "SUSPICIOUS_CONTENT",
                            f"Transcript contains suspicious pattern: {file_path}",
                            f"Model: {model_name}, Pattern: {pattern}"
                        )
            
            result['status'] = 'success'
            
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
            self.error_logger.log_error(
                "ANALYSIS_ERROR",
                f"Error analyzing transcript file: {file_path}",
                traceback.format_exc(),
                file_path
            )
        
        return result
    
    def find_model_files(self) -> Dict[str, List[str]]:
        """Find all model transcript files with error handling"""
        model_files = {}
        
        try:
            if not os.path.exists(self.transcript_dir):
                self.error_logger.log_error(
                    "DIRECTORY_NOT_FOUND",
                    f"Transcript directory not found: {self.transcript_dir}",
                    "Cannot proceed with analysis"
                )
                return model_files
            
            # Find all .txt files recursively (including subdirectories)
            txt_files = []
            for root, dirs, files in os.walk(self.transcript_dir):
                for file in files:
                    if file.endswith('.txt'):
                        txt_files.append(os.path.join(root, file))
            
            if not txt_files:
                self.error_logger.log_warning(
                    "NO_FILES_FOUND",
                    f"No .txt files found in transcript directory: {self.transcript_dir}",
                    "Check if ASR processing completed successfully"
                )
                return model_files
            
            # Group files by model
            for file_path in txt_files:
                filename = os.path.basename(file_path)
                
                # Skip summary files and other non-transcript files
                if filename in ['merging_summary.txt', 'summary.txt', 'processing_summary.txt']:
                    continue
                
                # Extract model name from filename
                # Expected format: model_name_original_filename.txt
                parts = filename.split('_')
                if len(parts) >= 2:
                    model_name = parts[0]
                    if model_name not in model_files:
                        model_files[model_name] = []
                    model_files[model_name].append(file_path)
                else:
                    self.error_logger.log_warning(
                        "INVALID_FILENAME",
                        f"Invalid filename format: {filename}",
                        f"Expected format: model_name_original_filename.txt",
                        file_path
                    )
            
        except Exception as e:
            self.error_logger.log_error(
                "FILE_DISCOVERY_ERROR",
                f"Error discovering model files: {str(e)}",
                traceback.format_exc()
            )
        
        return model_files
    
    def analyze_all_models(self) -> Dict[str, Any]:
        """Analyze all model files"""
        print("Discovering model files...")
        model_files = self.find_model_files()
        
        if not model_files:
            self.error_logger.log_error(
                "NO_MODELS_FOUND",
                "No model files found for analysis",
                "Check transcript directory and file naming conventions"
            )
            return {}
        
        print(f"Found {len(model_files)} models: {list(model_files.keys())}")
        
        all_results = {}
        
        for model_name, file_paths in model_files.items():
            print(f"Analyzing {model_name} ({len(file_paths)} files)...")
            
            model_results = {
                'model_name': model_name,
                'total_files': len(file_paths),
                'successful_files': 0,
                'empty_files': 0,
                'error_files': 0,
                'missing_ground_truth': 0,
                'file_details': []
            }
            
            for file_path in file_paths:
                file_result = self.analyze_transcript_file(file_path, model_name)
                model_results['file_details'].append(file_result)
                
                # Count by status
                if file_result['status'] == 'success':
                    model_results['successful_files'] += 1
                elif file_result['status'] in ['empty', 'empty_content']:
                    model_results['empty_files'] += 1
                else:
                    model_results['error_files'] += 1
            
            all_results[model_name] = model_results
        
        return all_results
    
    def check_ground_truth_matching(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if transcript files have corresponding ground truth entries"""
        if self.ground_truth_data is None:
            return analysis_results
        
        ground_truth_filenames = set(self.ground_truth_data['Filename'].tolist())
        
        for model_name, model_data in analysis_results.items():
            missing_gt_count = 0
            
            for file_detail in model_data['file_details']:
                if file_detail['status'] == 'success':
                    # Extract original filename from transcript filename
                    filename = os.path.basename(file_detail['file_path'])
                    parts = filename.split('_')
                    
                    if len(parts) >= 2:
                        # Reconstruct original filename
                        original_filename = '_'.join(parts[1:]).replace('.txt', '.wav')
                        
                        if original_filename not in ground_truth_filenames:
                            missing_gt_count += 1
                            file_detail['warnings'].append('No corresponding ground truth entry')
                            self.error_logger.log_warning(
                                "MISSING_GROUND_TRUTH",
                                f"No ground truth entry for: {original_filename}",
                                f"Model: {model_name}, Transcript: {filename}"
                            )
            
            model_data['missing_ground_truth'] = missing_gt_count
        
        return analysis_results

def generate_analysis_report(analysis_results: Dict[str, Any], error_logger: ErrorLogger, 
                           output_file: str, error_log_file: str) -> None:
    """Generate comprehensive analysis report"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Model File Processing Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Error Log File: {error_log_file}\n")
        f.write("\n")
        
        # Summary table
        f.write("Summary Table:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<20} {'Total':<8} {'Success':<8} {'Empty':<8} {'Errors':<8} {'Missing GT':<10}\n")
        f.write("-" * 80 + "\n")
        
        for model_name, model_data in analysis_results.items():
            f.write(f"{model_name:<20} {model_data['total_files']:<8} {model_data['successful_files']:<8} "
                   f"{model_data['empty_files']:<8} {model_data['error_files']:<8} {model_data['missing_ground_truth']:<10}\n")
        
        f.write("\n")
        
        # Detailed analysis for each model
        for model_name, model_data in analysis_results.items():
            f.write(f"=== {model_name} ===\n")
            f.write(f"Total files found: {model_data['total_files']}\n")
            f.write(f"Successful files: {model_data['successful_files']}\n")
            f.write(f"Empty files: {model_data['empty_files']}\n")
            f.write(f"Error files: {model_data['error_files']}\n")
            f.write(f"Missing ground truth: {model_data['missing_ground_truth']}\n")
            
            # List successful files
            successful_files = [fd for fd in model_data['file_details'] if fd['status'] == 'success']
            if successful_files:
                f.write("Successful files:\n")
                for file_detail in successful_files:
                    filename = os.path.basename(file_detail['file_path'])
                    f.write(f"  ✓ {filename}\n")
            
            # List files with issues
            problematic_files = [fd for fd in model_data['file_details'] if fd['status'] != 'success']
            if problematic_files:
                f.write("Files with issues:\n")
                for file_detail in problematic_files:
                    filename = os.path.basename(file_detail['file_path'])
                    f.write(f"  ✗ {filename} ({file_detail['status']})\n")
                    if file_detail['errors']:
                        for error in file_detail['errors']:
                            f.write(f"    - Error: {error}\n")
                    if file_detail['warnings']:
                        for warning in file_detail['warnings']:
                            f.write(f"    - Warning: {warning}\n")
            
            f.write("\n")
        
        # Error summary
        error_summary = error_logger.get_summary()
        if error_summary['total_errors'] > 0 or error_summary['total_warnings'] > 0:
            f.write("=== Error Summary ===\n")
            f.write(f"Total errors: {error_summary['total_errors']}\n")
            f.write(f"Total warnings: {error_summary['total_warnings']}\n")
            f.write("See detailed error log: " + error_log_file + "\n")
            f.write("\n")
        
        # Model comparison
        f.write("=== Model Comparison ===\n")
        if analysis_results:
            max_files = max(model_data['total_files'] for model_data in analysis_results.values())
            f.write(f"Expected files per model: {max_files}\n")
            f.write(f"Maximum successful files: {max_files}\n\n")
            
            for model_name, model_data in analysis_results.items():
                if model_data['successful_files'] == max_files:
                    f.write(f"{model_name}: ✓ All files processed\n")
                else:
                    f.write(f"{model_name}: ⚠ {model_data['successful_files']}/{max_files} files processed\n")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Model File Analysis with Error Logging")
    parser.add_argument("--transcript_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--ground_truth_file", required=True, help="Ground truth CSV file")
    parser.add_argument("--output_file", required=True, help="Output analysis file")
    parser.add_argument("--error_log_file", required=True, help="Error log file")
    parser.add_argument("--pipeline_output_dir", help="Pipeline output directory for additional context")
    
    args = parser.parse_args()
    
    # Initialize error logger
    error_logger = ErrorLogger(args.error_log_file)
    
    # Ensure error log directory exists
    error_log_dir = os.path.dirname(args.error_log_file)
    if error_log_dir:  # Only create directory if path is not empty
        os.makedirs(error_log_dir, exist_ok=True)
    
    # Log analysis start
    error_logger.log_warning(
        "ANALYSIS_START",
        f"Starting enhanced model file analysis",
        f"Transcript dir: {args.transcript_dir}, Ground truth: {args.ground_truth_file}"
    )
    
    # Initialize analyzer
    analyzer = EnhancedModelFileAnalyzer(args.transcript_dir, args.ground_truth_file, error_logger)
    
    # Load ground truth
    if not analyzer.load_ground_truth():
        print("Failed to load ground truth data. Check error log for details.")
        return 1
    
    # Analyze all models
    analysis_results = analyzer.analyze_all_models()
    
    if not analysis_results:
        print("No model files found for analysis. Check error log for details.")
        return 1
    
    # Check ground truth matching
    analysis_results = analyzer.check_ground_truth_matching(analysis_results)
    
    # Generate report
    generate_analysis_report(analysis_results, error_logger, args.output_file, args.error_log_file)
    
    # Log analysis completion
    error_logger.log_warning(
        "ANALYSIS_COMPLETE",
        f"Enhanced model file analysis completed",
        f"Results saved to: {args.output_file}"
    )
    
    print(f"Analysis completed. Results saved to: {args.output_file}")
    print(f"Error log saved to: {args.error_log_file}")
    
    return 0

if __name__ == '__main__':
    exit(main()) 