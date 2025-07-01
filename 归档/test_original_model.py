import os
import torch
import json
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)
from sklearn.metrics import accuracy_score, classification_report
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(data_dir):
    """加载测试数据"""
    test_path = os.path.join(data_dir, "ohsumed_test.json")
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 验证数据格式
            if not isinstance(data, list):
                print("错误：数据不是列表格式")
                return []
            return data
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        return []

def preprocess_data(texts, tokenizer):
    """预处理数据"""
    prompts = [
        f"### Instruction: Classify the following medical text into one of the 23 categories.\n"
        f"### Text: {text}\n"
        f"### Categories: C1-C23\n"
        f"### Response: The text belongs to category " 
        for text in texts
    ]
    
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=400,  # 改为400
        return_tensors="pt"
    )
    return tokenized

def main():
    # 设置路径
    model_path = "/root/autodl-tmp/miggs/llama3.1_3.18/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"  # 原始模型路径
    data_dir = "/root/autodl-tmp/miggs/ohsumed_converted"  # 测试数据路径
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU运行")
        device = "cpu"
    else:
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
        # 设置CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型配置
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.num_labels = 23
    config.problem_type = "single_label_classification"
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用半精度
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config={
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_has_fp16_weight": False,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    )
    
    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    
    # 加载测试数据
    print("加载测试数据...")
    test_data = load_test_data(data_dir)
    texts = [item["input"] for item in test_data]
    true_labels = [int(item["category"][1:]) - 1 for item in test_data]  # 将C1-C23转换为0-22
    
    # 分批处理数据
    print("开始预测...")
    model.eval()
    predictions = []
    batch_size = 1  # 使用较小的批次大小
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 预处理当前批次
            tokenized_data = preprocess_data(batch_texts, tokenizer)
            input_ids = tokenized_data["input_ids"].to(device)
            attention_mask = tokenized_data["attention_mask"].to(device)
            
            # 预测当前批次
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)
            
            # 清理GPU内存
            del input_ids
            del attention_mask
            del outputs
            del tokenized_data
            torch.cuda.empty_cache()
            
            # 打印进度
            if (i + batch_size) % 10 == 0:
                print(f"已处理 {i + batch_size}/{len(texts)} 个样本")
                # 定期清理缓存
                torch.cuda.empty_cache()
    
    # 计算性能指标
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\n准确率: {accuracy:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, predictions))
    
    # 打印每个类别的样本数量
    print("\n每个类别的样本数量:")
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label+1}: {count} 个样本")

if __name__ == "__main__":
    main() 