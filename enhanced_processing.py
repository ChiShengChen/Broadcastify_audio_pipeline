#!/usr/bin/env python3
"""
Enhanced Processing Script

This script takes JSON extraction results and processes them further using an LLM
to generate enhanced outputs (e.g., structured emergency pages from JSON data).
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedProcessor:
    """Enhanced processing using LLM to further refine extracted JSON data."""
    
    def __init__(self, model_name: str, device: str = "cuda", load_in_8bit: bool = False, load_in_4bit: bool = False):
        """Initialize the enhanced processor with the specified model."""
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Initializing enhanced processor with model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model and tokenizer."""
        try:
            # Determine model path
            model_path = self._get_model_path(self.model_name)
            
            # Configure quantization
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info(f"Loading model from: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cuda" and self.model.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_model_path(self, model_name: str) -> str:
        """Get the full model path for the given model name."""
        model_paths = {
            "BioMistral-7B": "BioMistral/BioMistral-7B",
            "gpt-oss-20b": "microsoft/DialoGPT-medium",  # Placeholder
            "gpt-oss-120b": "microsoft/DialoGPT-medium",  # Placeholder
        }
        
        if model_name in model_paths:
            return model_paths[model_name]
        else:
            # Assume it's already a full path
            return model_name
    
    def process_json_data(self, json_data: Dict[str, Any], prompt: str) -> str:
        """Process JSON data using the LLM with the given prompt."""
        try:
            # Format the input with JSON data
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            input_text = f"{prompt}\n\nJSON Data:\n{json_str}\n\nEnhanced Output:"
            
            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the input)
            if "Enhanced Output:" in generated_text:
                enhanced_part = generated_text.split("Enhanced Output:")[-1].strip()
                return enhanced_part
            else:
                return generated_text.strip()
                
        except Exception as e:
            logger.error(f"Error processing JSON data: {e}")
            return f"Error processing data: {str(e)}"
    
    def process_file(self, input_file: Path, output_file: Path, prompt: str):
        """Process a single JSON file and save the enhanced output."""
        try:
            logger.info(f"Processing file: {input_file}")
            
            # Read JSON data
            with open(input_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Process the data
            enhanced_output = self.process_json_data(json_data, prompt)
            
            # Save enhanced output
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_output)
            
            logger.info(f"Enhanced output saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}")
            # Save error message
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Error processing file: {str(e)}")

def main():
    """Main function to run enhanced processing."""
    parser = argparse.ArgumentParser(description="Enhanced Processing Script")
    parser.add_argument("--input_dir", required=True, help="Input directory containing JSON files")
    parser.add_argument("--output_dir", required=True, help="Output directory for enhanced results")
    parser.add_argument("--model", required=True, help="Model name for enhanced processing")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--prompt", required=True, help="Prompt for enhanced processing")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = EnhancedProcessor(
        model_name=args.model,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # Find JSON files
    input_dir = Path(args.input_dir)
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    for json_file in json_files:
        # Create output filename
        output_filename = json_file.stem + "_enhanced.txt"
        output_file = output_dir / output_filename
        
        # Process the file
        processor.process_file(json_file, output_file, args.prompt)
    
    logger.info(f"Enhanced processing completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
