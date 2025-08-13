#!/usr/bin/env python3
"""
Setup Local Models Script

This script downloads and sets up local Hugging Face models for the LLM pipeline.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import required libraries
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    print(f"Error: Required libraries not installed. Please install: {e}")
    print("Run: pip install torch transformers")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_local_models.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "BioMistral-7B": {
        "huggingface_id": "BioMistral/BioMistral-7B",
        "description": "Medical-focused Mistral model",
        "size_gb": 14,
        "requirements": "16GB+ RAM, CUDA recommended"
    },
    "Meditron-7B": {
        "huggingface_id": "epfl-llm/meditron-7b",
        "description": "Medical instruction-tuned model",
        "size_gb": 14,
        "requirements": "16GB+ RAM, CUDA recommended"
    },
    "gpt-oss-20b": {
        "huggingface_id": "microsoft/DialoGPT-medium",  # Placeholder
        "description": "Open source GPT model (placeholder)",
        "size_gb": 40,
        "requirements": "32GB+ RAM, CUDA required"
    },
    "Llama-3-8B-UltraMedica": {
        "huggingface_id": "meta-llama/Llama-3-8B",  # Placeholder
        "description": "Medical-focused Llama model (placeholder)",
        "size_gb": 16,
        "requirements": "16GB+ RAM, CUDA recommended"
    }
}

def check_disk_space(required_gb: float, download_path: Path) -> bool:
    """Check if there's enough disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(download_path)
        free_gb = free / (1024**3)
        
        logger.info(f"Available disk space: {free_gb:.1f}GB")
        logger.info(f"Required disk space: {required_gb:.1f}GB")
        
        if free_gb < required_gb:
            logger.warning(f"Insufficient disk space. Need {required_gb:.1f}GB, have {free_gb:.1f}GB")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check

def check_cuda_availability() -> bool:
    """Check if CUDA is available"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = []
        for i in range(gpu_count):
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_memory.append(memory)
        
        logger.info(f"CUDA available: {gpu_count} GPU(s)")
        for i, memory in enumerate(gpu_memory):
            logger.info(f"  GPU {i}: {memory:.1f}GB")
        return True
    else:
        logger.warning("CUDA not available. Models will run on CPU (slower)")
        return False

def download_model(model_name: str, download_path: Path, force_download: bool = False) -> bool:
    """Download a model from Hugging Face"""
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}")
        return False
    
    config = MODEL_CONFIGS[model_name]
    huggingface_id = config["huggingface_id"]
    size_gb = config["size_gb"]
    
    logger.info(f"Setting up model: {model_name}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Size: {size_gb}GB")
    logger.info(f"Requirements: {config['requirements']}")
    
    # Check disk space
    if not check_disk_space(size_gb, download_path):
        return False
    
    # Create model directory
    model_dir = download_path / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if model_dir.exists() and not force_download:
        logger.info(f"Model {model_name} already exists at {model_dir}")
        return True
    
    try:
        logger.info(f"Downloading model {model_name} from {huggingface_id}...")
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            huggingface_id,
            cache_dir=model_dir,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(model_dir)
        
        # Download model
        logger.info("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            huggingface_id,
            cache_dir=model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.save_pretrained(model_dir)
        
        logger.info(f"Model {model_name} downloaded successfully to {model_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        return False

def list_available_models():
    """List all available models"""
    print("Available Models:")
    print("=" * 50)
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"Model: {model_name}")
        print(f"  Description: {config['description']}")
        print(f"  Size: {config['size_gb']}GB")
        print(f"  Requirements: {config['requirements']}")
        print(f"  HuggingFace ID: {config['huggingface_id']}")
        print()

def create_model_config_file(download_path: Path):
    """Create a model configuration file"""
    config_file = download_path / "model_config.json"
    
    config = {
        "models": MODEL_CONFIGS,
        "download_path": str(download_path),
        "setup_date": str(Path().cwd()),
        "cuda_available": torch.cuda.is_available()
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Model configuration saved to: {config_file}")

def main():
    parser = argparse.ArgumentParser(description="Setup Local Models")
    parser.add_argument("--models", nargs="+", 
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Models to download")
    parser.add_argument("--download_path", default="./models",
                       help="Path to download models")
    parser.add_argument("--list", action="store_true",
                       help="List available models")
    parser.add_argument("--check_system", action="store_true",
                       help="Check system requirements")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download existing models")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if args.check_system:
        print("System Requirements Check:")
        print("=" * 30)
        
        # Check CUDA
        cuda_available = check_cuda_availability()
        
        # Check disk space
        download_path = Path(args.download_path)
        download_path.mkdir(parents=True, exist_ok=True)
        
        total_size = sum(config["size_gb"] for config in MODEL_CONFIGS.values())
        print(f"\nTotal disk space needed for all models: {total_size:.1f}GB")
        
        if check_disk_space(total_size, download_path):
            print("✓ Sufficient disk space")
        else:
            print("✗ Insufficient disk space")
        
        return
    
    if not args.models:
        print("Error: Please specify models to download with --models")
        print("Use --list to see available models")
        return
    
    # Create download directory
    download_path = Path(args.download_path)
    download_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Download path: {download_path}")
    
    # Check system requirements
    check_cuda_availability()
    
    # Download models
    success_count = 0
    for model_name in args.models:
        if download_model(model_name, download_path, args.force):
            success_count += 1
        else:
            logger.error(f"Failed to download {model_name}")
    
    # Create configuration file
    create_model_config_file(download_path)
    
    # Summary
    logger.info(f"Setup completed: {success_count}/{len(args.models)} models downloaded successfully")
    
    if success_count == len(args.models):
        logger.info("All models downloaded successfully!")
        logger.info(f"Models are available at: {download_path}")
        logger.info("You can now use the LLM pipeline with local models.")
    else:
        logger.warning("Some models failed to download. Check the logs for details.")

if __name__ == "__main__":
    main() 