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
    labels = [int(item['output'][1:]) - 1 for item in data]  # C1-C23 -> 0-22
    return Dataset.from_dict({"text": texts, "label": labels})
# ---- 3. 准备类别Token ----
class_labels = [f"C{str(i).zfill(2)}" for i in range(1, 24)]
def build_prompt(text):
    return (
        "你是一个医疗文本分类专家。你的任务是将给定的医疗文本分类到23个类别中的一个。\n\n"
        "类别对应关系：\n"
        "C01 - Bacterial Infections and Mycoses (细菌感染和真菌病)\n"
        "C02 - Virus Diseases (病毒疾病)\n"
        "C03 - Parasitic Diseases (寄生虫疾病)\n"
        "C04 - Neoplasms (肿瘤)\n"
        "C05 - Musculoskeletal Diseases (肌肉骨骼疾病)\n"
        "C06 - Digestive System Diseases (消化系统疾病)\n"
        "C07 - Stomatognathic Diseases (口腔颌面疾病)\n"
        "C08 - Respiratory Tract Diseases (呼吸道疾病)\n"
        "C09 - Otorhinolaryngologic Diseases (耳鼻喉疾病)\n"
        "C10 - Nervous System Diseases (神经系统疾病)\n"
        "C11 - Eye Diseases (眼部疾病)\n"
        "C12 - Urologic and Male Genital Diseases (泌尿和男性生殖系统疾病)\n"
        "C13 - Female Genital Diseases and Pregnancy Complications (女性生殖系统疾病和妊娠并发症)\n"
        "C14 - Cardiovascular Diseases (心血管疾病)\n"
        "C15 - Hemic and Lymphatic Diseases (血液和淋巴系统疾病)\n"
        "C16 - Neonatal Diseases and Abnormalities (新生儿疾病和异常)\n"
        "C17 - Skin and Connective Tissue Diseases (皮肤和结缔组织疾病)\n"
        "C18 - Nutritional and Metabolic Diseases (营养和代谢疾病)\n"
        "C19 - Endocrine Diseases (内分泌疾病)\n"
        "C20 - Immunologic Diseases (免疫系统疾病)\n"
        "C21 - Disorders of Environmental Origin (环境源性疾病)\n"
        "C22 - Animal Diseases (动物疾病)\n"
        "C23 - Pathological Conditions, Signs and Symptoms (病理状况、体征和症状)\n\n"
        f"文本: {text}\n\n"
        "请直接给出最终分类结果。例如，如果文本和肿瘤相关，则输出C04。"
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
        inputs = tokenizer(chat_input, return_tensors="pt", max_length=512, truncation=True).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.4,
                # top_p=0.9,
                do_sample=False,
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
                max_new_tokens=5,
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
    # main()  # 运行完整评估
    test_show_outputs()  # 运行示例输出展示 