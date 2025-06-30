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

# ---- 统一配置参数 ----
MAX_NEW_TOKENS = 256  # 统一设置最大生成token数
TEMPERATURE = 0.4    # 统一设置温度参数
TOP_P = 0.9          # 统一设置top_p参数
DO_SAMPLE = False    # 统一设置是否采样
BATCH_SIZE = 512      # 批处理大小，根据显存调整

def load_ohsumed_test_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['input'] for item in data]
    labels = [int(item['output'][1:]) - 1 for item in data]  # C1-C23 -> 0-22
    return Dataset.from_dict({"text": texts, "label": labels})
# ---- 3. 准备类别Token ----
class_labels = [f"C{str(i).zfill(2)}" for i in range(1, 24)]
def build_prompt(text):
    return (
        "You are a medical text classification expert. Your task is to classify the given medical text into one of 23 categories.\n\n"
        "Category mapping:\n"
        "C01 - Bacterial Infections and Mycoses\n"
        "C02 - Virus Diseases\n"
        "C03 - Parasitic Diseases\n"
        "C04 - Neoplasms\n"
        "C05 - Musculoskeletal Diseases\n"
        "C06 - Digestive System Diseases\n"
        "C07 - Stomatognathic Diseases\n"
        "C08 - Respiratory Tract Diseases\n"
        "C09 - Otorhinolaryngologic Diseases\n"
        "C10 - Nervous System Diseases\n"
        "C11 - Eye Diseases\n"
        "C12 - Urologic and Male Genital Diseases\n"
        "C13 - Female Genital Diseases and Pregnancy Complications\n"
        "C14 - Cardiovascular Diseases\n"
        "C15 - Hemic and Lymphatic Diseases\n"
        "C16 - Neonatal Diseases and Abnormalities\n"
        "C17 - Skin and Connective Tissue Diseases\n"
        "C18 - Nutritional and Metabolic Diseases\n"
        "C19 - Endocrine Diseases\n"
        "C20 - Immunologic Diseases\n"
        "C21 - Disorders of Environmental Origin\n"
        "C22 - Animal Diseases\n"
        "C23 - Pathological Conditions, Signs and Symptoms\n\n"
        f"Text: {text}\n\n"
        "For example, if the text is related to neoplasms, it belongs to C04. The output must be one of C01-C23 categories. If you output a non-existent category, you will be penalized. Let's classify step by step:"
    )

def extract_category(output):
    """从推理过程中提取类别编号 (推荐的最终版本)"""
    # 1. 优先匹配您期望的最终格式
    match = re.search(r"最终分类结果：C(\d{2})", output)
    if match:
        category_num = int(match.group(1))
        if 1 <= category_num <= 23:
            return category_num - 1

    # 2. 其次，匹配任何地方出现的 CXX 格式
    match = re.search(r"C(\d{2})", output)
    if match:
        category_num = int(match.group(1))
        if 1 <= category_num <= 23:
            return category_num - 1

    # 3. 如果都找不到，则判定为无法分类
    return -1

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
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
    # JUMP HERE
    base_model_path = "/mnt/data1/TC/TextClassDemo/llama3.1-8b"
    adapter_path = "/mnt/data1/TC/TextClassDemo/LLaMA-Factory/llama3.1-8b_ohsumed_direct_lora"
    data_path = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Test_alpaca_noCoT.json"
    
    # 控制是否使用LoRA适配器
    use_lora = False  # 设置为False则不使用LoRA，直接使用base model
    
    # 根据use_lora设置不同的输出目录
    if use_lora:
        output_dir = "./outputs/ohsumed_lora_model"
    else:
        output_dir = "./outputs/ohsumed_base_model"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # 不需要手动设置chat template，LLaMA-3.1-Instruct已经有内置的chat template
    print(f"Chat template: {tokenizer.chat_template}")

    print("加载模型...")
    # 直接使用float16加载模型，禁用量化
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用float16来减少显存使用
        low_cpu_mem_usage=True
    )
    
    if use_lora:
        print("加载PEFT适配器...")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map="auto",
            torch_dtype=torch.float16  # 确保PEFT模型也使用float16
        )
    else:
        print("使用基础模型，不加载LoRA适配器")
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # 加载和预处理测试集
    print("加载测试数据...")
    test_dataset = load_ohsumed_test_dataset(data_path)
    texts = test_dataset["text"]
    labels = test_dataset["label"]

    print("开始批处理推理...")
    preds = []
    outputs = []
    
    # 批处理推理
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="处理批次"):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_prompts = [build_prompt(text) for text in batch_texts]
        
        # 构建批处理的对话格式
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        batch_chat_inputs = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) for messages in batch_messages
        ]
        
        # 批处理tokenization
        batch_inputs = tokenizer(
            batch_chat_inputs, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True  # 启用padding以支持批处理
        ).to(model.device)
        
        with torch.no_grad():
            batch_generated_ids = model.generate(
                **batch_inputs,
                max_new_tokens=MAX_NEW_TOKENS,  # 使用统一配置
                temperature=TEMPERATURE,         # 使用统一配置
                top_p=TOP_P,                    # 使用统一配置
                do_sample=DO_SAMPLE,            # 使用统一配置
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码每个样本的输出
        for j, (input_ids, generated_ids_sample) in enumerate(zip(batch_inputs["input_ids"], batch_generated_ids)):
            # 提取新生成的部分
            new_tokens = generated_ids_sample[input_ids.shape[0]:]
            output = tokenizer.decode(new_tokens, skip_special_tokens=True)
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
        "outputs": outputs,
        "use_lora": use_lora,  # 记录是否使用了LoRA
        "model_type": "LoRA" if use_lora else "Base Model"  # 记录模型类型
    }
    
    with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印主要结果
    print("\n评估结果：")
    print(f"模型类型: {'LoRA' if use_lora else 'Base Model'}")
    print(f"准确率：{acc:.4f}")
    print(f"错误样本数：{len(errors)}")
    print(f"总样本数：{len(texts)}")
    print(f"结果保存到: {output_dir}")
    print("\n分类报告：")
    print(json.dumps(report, ensure_ascii=False, indent=2))

def test_show_outputs():
    base_model_path = "/mnt/data1/TC/TextClassDemo/llama3.1-8b"
    adapter_path = "/mnt/data1/TC/TextClassDemo/LLaMA-Factory/llama3.1-8b_ohsumed_direct_lora"
    data_path = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Test_alpaca_noCoT.json"
    
    # 控制是否使用LoRA适配器
    use_lora = True  # 设置为False则不使用LoRA，直接使用base model
    
    # 根据use_lora设置不同的输出目录
    if use_lora:
        output_dir = "./outputs/ohsumed_lora_model"
    else:
        output_dir = "./outputs/ohsumed_base_model"

    print("加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 不需要手动设置chat template，LLaMA-3.1-Instruct已经有内置的chat template
    print(f"Chat template: {tokenizer.chat_template}")

    # 直接使用float16加载模型，禁用量化
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用float16来减少显存使用
        low_cpu_mem_usage=True
    )
    
    if use_lora:
        print("加载PEFT适配器...")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map="auto",
            torch_dtype=torch.float16  # 确保PEFT模型也使用float16
        )
    else:
        print("使用基础模型，不加载LoRA适配器")
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # 加载测试集
    print("加载测试数据...")
    test_dataset = load_ohsumed_test_dataset(data_path)
    texts = test_dataset["text"]
    labels = test_dataset["label"]
    
    # 随机选择一些样本进行展示
    indices = np.random.choice(len(texts), min(5, len(texts)), replace=False)
    
    print(f"\n使用模型: {'LoRA' if use_lora else 'Base Model'}")
    print("示例输出：")
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
                max_new_tokens=MAX_NEW_TOKENS,  # 使用统一配置
                temperature=TEMPERATURE,         # 使用统一配置
                top_p=TOP_P,                    # 使用统一配置
                do_sample=DO_SAMPLE,            # 使用统一配置
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