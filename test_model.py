import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from datasets import load_dataset
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os
from peft import PeftModel, PeftConfig

def load_test_data(data_dir):
    """加载测试数据"""
    test_file = os.path.join(data_dir, "ohsumed_test.json")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    return test_data

def preprocess_data(texts, tokenizer):
    """预处理数据"""
    prompts = [
        f"### Instruction: Classify the following medical text into one of the 23 categories.\n"
        f"### Text: {text}\n"
        f"### Categories: C1-C23\n"
        f"### Response: The text belongs to category " 
        for text in texts
    ]
    
    # 打印第一个prompt样例
    print("\n=== Prompt样例 ===")
    print(prompts[0])
    print("===================\n")
    
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=400,
        return_tensors="pt"
    )
    return tokenized

def main():
    # 设置路径
    base_model_path = "llama3.1_3.18/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    checkpoint_path = "checkpoint"
    data_dir = "ohsumed_converted"
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU运行")
        device = "cpu"
    else:
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
        # 设置批次大小
        batch_size = 1  # 使用较小的批次大小
        # 启用梯度检查点
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载基础模型配置
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    config.num_labels = 23
    config.problem_type = "single_label_classification"
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用半精度
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # 启用梯度检查点
    base_model.gradient_checkpointing_enable()
    
    # 加载PEFT配置
    print("加载PEFT配置...")
    peft_config = PeftConfig.from_pretrained(checkpoint_path)
    
    # 加载微调后的模型
    print("加载微调后的模型...")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 清理基础模型
    del base_model
    torch.cuda.empty_cache()
    
    # 加载测试数据
    print("加载测试数据...")
    test_data = load_test_data(data_dir)
    texts = [item["input"] for item in test_data]
    true_labels = [int(item["category"][1:]) - 1 for item in test_data]  # 将C1-C23转换为0-22
    
    # 分批处理数据
    print("开始预测...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = true_labels[i:i+batch_size]
            
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