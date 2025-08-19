#!/usr/bin/env python3
"""
简化的LLM模型测试脚本
用于调试模型加载问题
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """测试模型加载"""
    # 使用一个更小的模型来测试
    model_name = "microsoft/DialoGPT-medium"  # 约345M参数
    
    try:
        logger.info(f"Testing model loading: {model_name}")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.warning("CUDA not available, using CPU")
        
        # 暂时跳过量化，直接加载模型
        logger.info("Loading model without quantization...")
        
        # 加载tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型（不使用量化）
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("Model loaded successfully!")
        
        # 测试推理
        logger.info("Testing inference...")
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # 确保输入在正确的设备上
        device = next(model.parameters()).device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else None
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test response: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("✅ Model loading test passed!")
        sys.exit(0)
    else:
        print("❌ Model loading test failed!")
        sys.exit(1) 