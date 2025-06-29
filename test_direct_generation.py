import os
import torch
import json
import numpy as np
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_ohsumed_test_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['input'] for item in data]
    labels = [int(item['category'][1:]) - 1 for item in data]  # C1-C23 -> 0-22
    return Dataset.from_dict({"text": texts, "label": labels})
# ---- 3. 准备类别Token ----
class_labels = [f"C{str(i).zfill(2)}" for i in range(1, 24)]
def build_prompt(text):
    return (
        "你是一个医疗文本分类专家。你的任务是将给定的医疗文本分类到23个类别（1-23）中的一个。\n\n"
        f"文本: {text}\n\n"
        "请直接给出最终分类结果，不要解释。"
        # "让我们一步步分析这个文本：\n"
        # "1. 首先理解文本的主要内容\n"
        # "2. 分析文本的关键特征\n"
        # "3. 根据特征判断最合适的类别\n"
        # "4. 给出最终分类结果\n"
    )

def extract_category(output):
    """从推理过程中提取类别编号"""
    # 首先尝试匹配最终分类结果
    match = re.search(r"最终分类结果：C(\d{2})", output)
    if match:
        return int(match.group(1)) - 1
    
    # 如果没有找到最终结果，尝试匹配推理过程中的类别号
    match = re.search(r"类别(\d+)", output)
    if match:
        return int(match.group(1)) - 1
    
    # 最后尝试匹配任何两位数字
    match = re.search(r"(\d{2})", output)
    if match:
        return int(match.group(1)) - 1
    
    return -1

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def analyze_errors(texts, labels, preds, outputs, output_dir):
    """分析错误样本"""
    errors = []
    for i, (text, label, pred, output) in enumerate(zip(texts, labels, preds, outputs)):
        if label != pred:
            errors.append({
                "index": i,
                "text": text,
                "true_label": f"C{label+1:02d}",
                "predicted_label": f"C{pred+1:02d}" if pred != -1 else "未分类",
                "model_output": output
            })
    
    # 保存错误分析结果
    with open(os.path.join(output_dir, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    
    return errors

def main():
    base_model_path = "llama3.1_3.18/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    adapter_path = "outputs/ohsumed_causal_lm_classification_cot/checkpoint-1956"
    data_path = "ohsumed_converted/ohsumed_test.json"
    output_dir = "./eval_output_cot"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("加载模型...")
    # 直接使用float16加载模型，禁用量化
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用float16来减少显存使用
        low_cpu_mem_usage=True
    )
    
    print("加载PEFT适配器...")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map="auto",
        torch_dtype=torch.float16  # 确保PEFT模型也使用float16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # 加载和预处理测试集
    print("加载测试数据...")
    test_dataset = load_ohsumed_test_dataset(data_path)
    texts = test_dataset["text"]
    labels = test_dataset["label"]

    print("开始推理...")
    preds = []
    outputs = []
    for text in tqdm(texts, desc="处理样本"):
        prompt = build_prompt(text)
        # 构建对话格式
        messages = [{"role": "user", "content": prompt}]
        chat_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(chat_input, return_tensors="pt", max_length=2048, truncation=True).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        output = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(output)
        pred = extract_category(output)
        preds.append(pred)

    # 计算评估指标
    print("计算评估指标...")
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    plot_confusion_matrix(labels, preds, output_dir)
    
    # 分析错误样本
    print("分析错误样本...")
    errors = analyze_errors(texts, labels, preds, outputs, output_dir)
    
    # 保存评估结果
    results = {
        "accuracy": acc,
        "report": report,
        "error_count": len(errors),
        "total_samples": len(texts),
        "outputs": outputs
    }
    
    with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印主要结果
    print("\n评估结果：")
    print(f"准确率：{acc:.4f}")
    print(f"错误样本数：{len(errors)}")
    print(f"总样本数：{len(texts)}")
    print("\n分类报告：")
    print(json.dumps(report, ensure_ascii=False, indent=2))

def test_show_outputs():
    """展示一些示例输出，包括正确和错误的样本"""
    base_model_path = "llama3.1_3.18/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    adapter_path = "outputs/ohsumed_causal_lm_classification_cot/checkpoint-1956"
    data_path = "ohsumed_converted/ohsumed_test.json"

    print("加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 直接使用float16加载模型，禁用量化
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用float16来减少显存使用
        low_cpu_mem_usage=True
    )
    
    print("加载PEFT适配器...")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map="auto",
        torch_dtype=torch.float16  # 确保PEFT模型也使用float16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # 加载测试集
    print("加载测试数据...")
    test_dataset = load_ohsumed_test_dataset(data_path)
    texts = test_dataset["text"]
    labels = test_dataset["label"]
    
    # 随机选择一些样本进行展示
    indices = np.random.choice(len(texts), min(5, len(texts)), replace=False)
    
    print("\n示例输出：")
    for i, idx in enumerate(indices):
        text = texts[idx]
        label = f"C{labels[idx]+1:02d}"
        prompt = build_prompt(text)
        
        # 构建对话格式
        messages = [{"role": "user", "content": prompt}]
        chat_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(chat_input, return_tensors="pt", max_length=2048, truncation=True).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        output = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_category(output)
        pred_label = f"C{pred+1:02d}" if pred != -1 else "未分类"
        
        print(f"\n样本{i+1}：")
        print(f"文本: {text[:200]}...")  # 只显示前200个字符
        print(f"真实类别: {label}")
        print(f"预测类别: {pred_label}")
        print(f"模型推理过程:\n{output}")
        print("-"*80)

if __name__ == "__main__":
    main()  # 运行完整评估
    # test_show_outputs()  # 运行示例输出展示 