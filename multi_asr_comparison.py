#!/usr/bin/env python3
"""
Multi-ASR Comparison Script
Compares Canary and Whisper ASR results and generates the best combined version.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiASRComparator:
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
        """Load the LLM model for comparison"""
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
    
    def compare_transcripts(self, canary_text: str, whisper_text: str, prompt: str) -> str:
        """Compare two ASR transcripts and generate the best combined version"""
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
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during transcript comparison: {e}")
            # Fallback: return the longer transcript
            return canary_text if len(canary_text) > len(whisper_text) else whisper_text
    
    def process_file_pair(self, canary_file: str, whisper_file: str, output_file: str, prompt: str) -> bool:
        """Process a pair of Canary and Whisper files"""
        try:
            # Read transcripts
            with open(canary_file, 'r', encoding='utf-8') as f:
                canary_text = f.read().strip()
            
            with open(whisper_file, 'r', encoding='utf-8') as f:
                whisper_text = f.read().strip()
            
            if not canary_text and not whisper_text:
                logger.warning(f"Both transcripts are empty for {output_file}")
                return False
            
            if not canary_text:
                logger.info(f"Canary transcript empty, using Whisper for {output_file}")
                combined_text = whisper_text
            elif not whisper_text:
                logger.info(f"Whisper transcript empty, using Canary for {output_file}")
                combined_text = canary_text
            else:
                # Compare and combine transcripts
                logger.info(f"Comparing transcripts for {output_file}")
                combined_text = self.compare_transcripts(canary_text, whisper_text, prompt)
            
            # Save combined result
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            
            logger.info(f"Saved combined transcript: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file pair {canary_file} + {whisper_file}: {e}")
            return False

def load_multi_asr_mapping(mapping_file: str) -> Dict[str, Dict[str, str]]:
    """Load the multi-ASR mapping file"""
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        logger.info(f"Loaded mapping for {len(mapping)} file pairs")
        return mapping
    except Exception as e:
        logger.error(f"Failed to load mapping file: {e}")
        return {}

def create_multi_asr_mapping(input_dir: str) -> Dict[str, Dict[str, str]]:
    """Create mapping between Canary and Whisper files for multi-ASR comparison"""
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

def main():
    parser = argparse.ArgumentParser(description="Multi-ASR Comparison Tool")
    parser.add_argument("--input_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--output_dir", required=True, help="Output directory for combined transcripts")
    parser.add_argument("--model", default="gpt-oss-20b", help="LLM model to use for comparison")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--prompt", required=True, help="Prompt template for comparison")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = MultiASRComparator(
        model_name=args.model,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # Load model
    comparator.load_model()
    
    # Create mapping file automatically
    logger.info(f"Creating multi-ASR mapping from: {args.input_dir}")
    mapping = create_multi_asr_mapping(args.input_dir)
    
    if not mapping:
        logger.error("No valid mapping could be created")
        logger.error("Make sure you have both canary-1b_* and large-v3_* files in the input directory")
        return 1
    
    # Process each file pair
    success_count = 0
    total_count = len(mapping)
    
    for base_name, file_paths in mapping.items():
        canary_file = file_paths.get("canary")
        whisper_file = file_paths.get("whisper")
        
        if not canary_file or not whisper_file:
            logger.warning(f"Incomplete mapping for {base_name}")
            continue
        
        output_file = os.path.join(args.output_dir, f"{base_name}.txt")
        
        if comparator.process_file_pair(canary_file, whisper_file, output_file, args.prompt):
            success_count += 1
    
    logger.info(f"Processing completed: {success_count}/{total_count} files successful")
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    exit(main())
