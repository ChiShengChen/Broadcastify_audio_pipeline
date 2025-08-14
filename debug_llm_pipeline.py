#!/usr/bin/env python3
"""
Debug LLM Pipeline Script

This script helps debug issues with the LLM pipeline, specifically model loading and device conflicts.
It tests each component step by step to isolate problems.
"""

import argparse
import logging
import sys
import torch
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check the environment and dependencies"""
    logger.info("=== Environment Check ===")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    except ImportError:
        logger.error("PyTorch not found!")
        return False
    
    # Check transformers
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("Transformers not found!")
        return False
    
    # Check accelerate
    try:
        import accelerate
        logger.info(f"Accelerate version: {accelerate.__version__}")
    except ImportError:
        logger.warning("Accelerate not found - may cause issues with large models")
    
    # Check bitsandbytes
    try:
        import bitsandbytes
        logger.info(f"BitsAndBytes version: {bitsandbytes.__version__}")
    except ImportError:
        logger.warning("BitsAndBytes not found - quantization not available")
    
    return True

def test_model_loading(model_name="BioMistral-7B", device="auto", load_in_8bit=False, load_in_4bit=False):
    """Test model loading with different configurations"""
    logger.info(f"=== Testing Model Loading: {model_name} ===")
    logger.info(f"Device: {device}, 8-bit: {load_in_8bit}, 4-bit: {load_in_4bit}")
    
    try:
        from llm_local_models import LocalLLMModel
        
        model = LocalLLMModel(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        
        logger.info("✅ Model loaded successfully!")
        
        # Check model properties
        if hasattr(model.model, 'hf_device_map'):
            logger.info(f"Model device map: {model.model.hf_device_map}")
        if hasattr(model.model, 'device'):
            logger.info(f"Model device: {model.model.device}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def test_generation(model, test_prompt="Patient presents with chest pain."):
    """Test text generation"""
    logger.info("=== Testing Text Generation ===")
    
    try:
        result = model.generate(test_prompt)
        if result:
            logger.info("✅ Generation successful!")
            logger.info(f"Input: {test_prompt}")
            logger.info(f"Output preview: {result[:200]}...")
            return True
        else:
            logger.error("❌ Generation returned None")
            return False
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_transcript_processing(input_dir, model):
    """Test processing actual transcript files"""
    logger.info("=== Testing Transcript Processing ===")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return False
    
    # Find transcript files
    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        logger.error(f"No .txt files found in {input_dir}")
        return False
    
    logger.info(f"Found {len(txt_files)} transcript files")
    
    # Test with first file
    test_file = txt_files[0]
    logger.info(f"Testing with file: {test_file.name}")
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            logger.warning(f"File {test_file.name} is empty")
            return False
        
        logger.info(f"File content preview: {content[:100]}...")
        
        # Test generation with transcript content
        prompt = f"You are a medical transcription specialist. Please correct any medical terms in the following transcript: {content}"
        
        result = model.generate(prompt)
        if result:
            logger.info("✅ Transcript processing successful!")
            logger.info(f"Corrected transcript preview: {result[:200]}...")
            return True
        else:
            logger.error("❌ Transcript processing returned None")
            return False
            
    except Exception as e:
        logger.error(f"❌ Transcript processing failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def suggest_fixes():
    """Suggest fixes for common issues"""
    logger.info("=== Suggested Fixes ===")
    
    logger.info("1. Memory Issues:")
    logger.info("   - Try with 8-bit quantization: --load_in_8bit")
    logger.info("   - Try with 4-bit quantization: --load_in_4bit")
    logger.info("   - Reduce batch size in pipeline")
    
    logger.info("2. Device Issues:")
    logger.info("   - Use device='auto' for automatic device mapping")
    logger.info("   - Ensure CUDA is available if using GPU")
    logger.info("   - Try device='cpu' if GPU memory is insufficient")
    
    logger.info("3. Model Loading Issues:")
    logger.info("   - Check internet connection for model download")
    logger.info("   - Verify model name and path")
    logger.info("   - Clear Hugging Face cache if corrupted")
    
    logger.info("4. Pipeline Issues:")
    logger.info("   - Update transformers: pip install --upgrade transformers")
    logger.info("   - Install missing dependencies: pip install accelerate bitsandbytes")

def main():
    parser = argparse.ArgumentParser(description="Debug LLM Pipeline")
    parser.add_argument("--model", default="BioMistral-7B", help="Model name to test")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--test_input_dir", help="Directory with transcript files to test")
    parser.add_argument("--skip_model_test", action="store_true", help="Skip model loading test")
    parser.add_argument("--skip_generation_test", action="store_true", help="Skip generation test")
    
    args = parser.parse_args()
    
    logger.info("Starting LLM Pipeline Debug")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed")
        sys.exit(1)
    
    # Test model loading
    model = None
    if not args.skip_model_test:
        model = test_model_loading(args.model, args.device, args.load_in_8bit, args.load_in_4bit)
        if not model:
            logger.error("Model loading test failed")
            suggest_fixes()
            sys.exit(1)
    
    # Test generation
    if model and not args.skip_generation_test:
        if not test_generation(model):
            logger.error("Generation test failed")
            suggest_fixes()
            sys.exit(1)
    
    # Test transcript processing
    if model and args.test_input_dir:
        if not test_transcript_processing(args.test_input_dir, model):
            logger.error("Transcript processing test failed")
            suggest_fixes()
            sys.exit(1)
    
    logger.info("✅ All tests passed!")
    logger.info("LLM pipeline should work correctly")

if __name__ == "__main__":
    main()