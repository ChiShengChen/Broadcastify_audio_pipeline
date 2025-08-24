#!/usr/bin/env python3
"""
Information Extraction Script for EMS Transcripts
Extracts structured information from transcripts using LLM
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InformationExtractor:
    def __init__(self, model_name: str, device: str = "auto", load_in_8bit: bool = False, load_in_4bit: bool = False):
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Map model names to full paths
        self.model_path = self._get_model_path(model_name)
        
        # Initialize model and tokenizer
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
        """Load the LLM model and tokenizer"""
        logger.info(f"Loading model: {self.model_name} -> {self.model_path}")
        
        # Configure quantization
        quantization_config = None
        if self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device == "auto" else self.device
        )
        
        logger.info("Model loaded successfully")
    
    def extract_information(self, transcript: str, prompt: str) -> str:
        """Extract structured information from transcript"""
        try:
            # Format the prompt with the transcript
            formatted_prompt = prompt.replace("{transcript}", transcript)
            
            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
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
            
            # Extract the generated part (after the prompt)
            generated_text = response[len(formatted_prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during information extraction: {e}")
            return "{}"  # Return empty JSON as fallback
    
    def process_file(self, input_file: Path, output_file: Path, prompt: str) -> bool:
        """Process a single transcript file"""
        try:
            # Read transcript
            with open(input_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            
            if not transcript:
                logger.warning(f"Empty transcript file: {input_file}")
                return False
            
            # Extract information
            extracted_info = self.extract_information(transcript, prompt)
            
            # Save result
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_info)
            
            logger.info(f"Processed: {input_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
            return False
    
    def process_directory(self, input_dir: str, output_dir: str, prompt: str, batch_size: int = 1):
        """Process all transcript files in directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all transcript files
        transcript_files = list(input_path.glob("*.txt"))
        logger.info(f"Found {len(transcript_files)} transcript files")
        
        success_count = 0
        total_count = len(transcript_files)
        
        for i, input_file in enumerate(transcript_files):
            # Create output filename
            output_file = output_path / f"{input_file.stem}_extracted.json"
            
            # Process file
            if self.process_file(input_file, output_file, prompt):
                success_count += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{total_count}")
        
        logger.info(f"Processing completed: {success_count}/{total_count} files successful")
        return success_count, total_count

def main():
    parser = argparse.ArgumentParser(description="Information Extraction for EMS Transcripts")
    parser.add_argument("--input_dir", required=True, help="Input directory containing transcript files")
    parser.add_argument("--output_dir", required=True, help="Output directory for extracted information")
    parser.add_argument("--model", default="BioMistral-7B", help="Model to use for extraction")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--prompt", required=True, help="Prompt template for information extraction")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = InformationExtractor(
        model_name=args.model,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # Load model
    extractor.load_model()
    
    # Process files
    success_count, total_count = extractor.process_directory(
        args.input_dir,
        args.output_dir,
        args.prompt,
        args.batch_size
    )
    
    if success_count > 0:
        logger.info(f"Information extraction completed successfully: {success_count}/{total_count} files")
        return 0
    else:
        logger.error("No files were processed successfully")
        return 1

if __name__ == "__main__":
    exit(main())
