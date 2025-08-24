#!/usr/bin/env python3
"""
ASR Selection Script
Compares Canary and Whisper ASR results and selects the better one.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import csv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASRSelector:
    def __init__(self, model_name: str, device: str = "auto", load_in_8bit: bool = False, load_in_4bit: bool = False):
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.model = None

    def _get_model_path(self, model_name: str) -> str:
        """Map simplified model names to full HuggingFace paths"""
        model_mapping = {
            "gpt-oss-20b": "openai/gpt-oss-20b",
            "gpt-oss-120b": "openai/gpt-oss-120b",
            "BioMistral-7B": "BioMistral/BioMistral-7B",
            "Meditron-7B": "epfl-llm/meditron-7b",
            "Llama-3-8B-UltraMedica": "/path/to/llama-3-8b-ultramedica"
        }
        return model_mapping.get(model_name, model_name)

    def load_model(self):
        """Load the LLM model for ASR selection"""
        try:
            # Get the full model path
            full_model_path = self._get_model_path(self.model_name)
            logger.info(f"Loading model: {self.model_name} -> {full_model_path}")
            
            # Set device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(full_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization if specified
            model_kwargs = {}
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            if self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                full_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None,
                **model_kwargs
            )
            
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def select_better_asr(self, canary_text: str, whisper_text: str, prompt: str) -> Dict:
        """Compare two ASR transcripts and select the better one"""
        try:
            # Format the prompt with actual transcripts
            formatted_prompt = prompt.replace("{canary_transcript}", canary_text)
            formatted_prompt = formatted_prompt.replace("{whisper_transcript}", whisper_text)

            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)

            # Move to device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (after the prompt)
            generated_text = response[len(formatted_prompt):].strip()

            # Parse JSON response
            try:
                # Find JSON object in the response
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    # Fallback: create basic result
                    logger.warning("Could not parse JSON response, using fallback")
                    return {
                        "selected_asr": "whisper" if len(whisper_text) > len(canary_text) else "canary",
                        "reason": "Fallback selection based on length",
                        "accuracy_score": 5,
                        "completeness_score": 5,
                        "medical_terminology_score": 5
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Fallback: create basic result
                return {
                    "selected_asr": "whisper" if len(whisper_text) > len(canary_text) else "canary",
                    "reason": "Fallback selection due to JSON parsing error",
                    "accuracy_score": 5,
                    "completeness_score": 5,
                    "medical_terminology_score": 5
                }

        except Exception as e:
            logger.error(f"Error during ASR selection: {e}")
            # Fallback: return basic result
            return {
                "selected_asr": "whisper" if len(whisper_text) > len(canary_text) else "canary",
                "reason": "Fallback selection due to processing error",
                "accuracy_score": 5,
                "completeness_score": 5,
                "medical_terminology_score": 5
            }

    def process_file_pair(self, canary_file: str, whisper_file: str, output_dir: str, prompt: str) -> Tuple[bool, Dict]:
        """Process a pair of Canary and Whisper files"""
        try:
            # Read transcripts
            with open(canary_file, 'r', encoding='utf-8') as f:
                canary_text = f.read().strip()

            with open(whisper_file, 'r', encoding='utf-8') as f:
                whisper_text = f.read().strip()

            if not canary_text and not whisper_text:
                logger.warning(f"Both transcripts are empty for {canary_file}")
                return False, {}

            if not canary_text:
                logger.info(f"Canary transcript empty, using Whisper")
                selected_text = whisper_text
                selection_result = {
                    "selected_asr": "whisper",
                    "reason": "Canary transcript was empty",
                    "accuracy_score": 0,
                    "completeness_score": 0,
                    "medical_terminology_score": 0
                }
            elif not whisper_text:
                logger.info(f"Whisper transcript empty, using Canary")
                selected_text = canary_text
                selection_result = {
                    "selected_asr": "canary",
                    "reason": "Whisper transcript was empty",
                    "accuracy_score": 0,
                    "completeness_score": 0,
                    "medical_terminology_score": 0
                }
            else:
                # Compare and select better ASR
                logger.info(f"Comparing ASR results for {os.path.basename(canary_file)}")
                selection_result = self.select_better_asr(canary_text, whisper_text, prompt)
                
                # Get the selected text
                if selection_result["selected_asr"] == "canary":
                    selected_text = canary_text
                else:
                    selected_text = whisper_text

            # Save selected result
            base_name = os.path.basename(canary_file)
            if base_name.startswith("canary-1b_"):
                base_name = base_name[10:]  # Remove "canary-1b_" prefix
            
            output_file = os.path.join(output_dir, f"{base_name}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(selected_text)

            logger.info(f"Saved selected transcript: {output_file}")
            return True, selection_result

        except Exception as e:
            logger.error(f"Error processing file pair {canary_file} + {whisper_file}: {e}")
            return False, {}

def create_asr_mapping(input_dir: str) -> Dict[str, Dict[str, str]]:
    """Create mapping between Canary and Whisper files for ASR selection"""
    mapping = {}
    
    # Find all transcript files
    transcript_files = []
    for file_path in Path(input_dir).glob("*.txt"):
        transcript_files.append(file_path)
    
    # Separate files by model
    canary_files = {}
    whisper_files = {}
    
    for file_path in transcript_files:
        filename = file_path.name
        
        # Extract base name (remove model prefix)
        if filename.startswith("canary-1b_"):
            base_name = filename[10:]  # Remove "canary-1b_" prefix
            canary_files[base_name] = str(file_path)
        elif filename.startswith("large-v3_"):
            base_name = filename[9:]   # Remove "large-v3_" prefix
            whisper_files[base_name] = str(file_path)
    
    # Create mapping for files that exist in both
    for base_name in set(canary_files.keys()) & set(whisper_files.keys()):
        mapping[base_name] = {
            "canary": canary_files[base_name],
            "whisper": whisper_files[base_name]
        }
    
    logger.info(f"Created mapping for {len(mapping)} file pairs")
    return mapping

def save_selection_results_csv(results: Dict[str, Dict], output_file: str):
    """Save ASR selection results to CSV"""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'selected_asr', 'reason', 'accuracy_score', 'completeness_score', 'medical_terminology_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for filename, result in results.items():
                row = {
                    'filename': filename,
                    'selected_asr': result.get('selected_asr', ''),
                    'reason': result.get('reason', ''),
                    'accuracy_score': result.get('accuracy_score', 0),
                    'completeness_score': result.get('completeness_score', 0),
                    'medical_terminology_score': result.get('medical_terminology_score', 0)
                }
                writer.writerow(row)
        
        logger.info(f"Selection results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save CSV results: {e}")

def main():
    parser = argparse.ArgumentParser(description="ASR Selection Tool")
    parser.add_argument("--input_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--output_dir", required=True, help="Output directory for selected transcripts")
    parser.add_argument("--model", default="gpt-oss-20b", help="LLM model to use for selection")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--prompt", required=True, help="Prompt template for ASR selection")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize selector
    selector = ASRSelector(
        model_name=args.model,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # Load model
    selector.load_model()
    
    # Create ASR mapping
    logger.info(f"Creating ASR mapping from: {args.input_dir}")
    mapping = create_asr_mapping(args.input_dir)
    
    if not mapping:
        logger.error("No valid mapping could be created")
        logger.error("Make sure you have both canary-1b_* and large-v3_* files in the input directory")
        return 1
    
    # Process each file pair
    success_count = 0
    total_count = len(mapping)
    selection_results = {}
    
    for base_name, file_paths in mapping.items():
        canary_file = file_paths.get("canary")
        whisper_file = file_paths.get("whisper")
        
        if not canary_file or not whisper_file:
            logger.warning(f"Incomplete mapping for {base_name}")
            continue
        
        success, result = selector.process_file_pair(canary_file, whisper_file, args.output_dir, args.prompt)
        
        if success:
            success_count += 1
            selection_results[base_name] = result
    
    # Save selection results to CSV
    csv_output_file = os.path.join(args.output_dir, "asr_selection_results.csv")
    save_selection_results_csv(selection_results, csv_output_file)
    
    logger.info(f"Processing completed: {success_count}/{total_count} files successful")
    logger.info(f"Selection results saved to: {csv_output_file}")
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    exit(main())
