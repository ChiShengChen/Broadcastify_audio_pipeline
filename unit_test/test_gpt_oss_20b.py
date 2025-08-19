#!/usr/bin/env python3
"""
专门测试 gpt-oss-20b 模型的脚本
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpt_oss_20b():
    """测试 gpt-oss-20b 模型加载"""
    model_name = "openai/gpt-oss-20b"
    
    try:
        logger.info(f"Testing gpt-oss-20b model loading: {model_name}")
        
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared PyTorch CUDA cache")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        else:
            logger.warning("CUDA not available, using CPU")
        
        # 加载tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 尝试不同的加载方法
        logger.info("Loading model...")
        
        # 方法1：使用最基本的加载方式
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # 使用bfloat16
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("Method 1 (basic) succeeded!")
        except Exception as e:
            logger.warning(f"Method 1 failed: {e}")
            
            # 方法2：不使用 device_map
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # 使用bfloat16
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                logger.info("Method 2 (no device_map) succeeded!")
            except Exception as e2:
                logger.warning(f"Method 2 failed: {e2}")
                
                # 方法3：使用 CPU
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        device_map=None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    logger.info("Method 3 (CPU) succeeded!")
                except Exception as e3:
                    logger.error(f"All methods failed. Last error: {e3}")
                    return False
        
        logger.info("Model loaded successfully!")
        
        # 测试推理
        logger.info("Testing inference...")
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # 确保输入在正确的设备上，并匹配模型的数据类型
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        logger.info(f"Model device: {device}, dtype: {dtype}")
        
        input_ids = inputs.input_ids.to(device=device, dtype=torch.long)  # token IDs should be long
        attention_mask = inputs.attention_mask.to(device=device, dtype=torch.long) if 'attention_mask' in inputs else None
        
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
    success = test_gpt_oss_20b()
    if success:
        print("✅ gpt-oss-20b model loading test passed!")
        sys.exit(0)
    else:
        print("❌ gpt-oss-20b model loading test failed!")
        sys.exit(1) 