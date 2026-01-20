import os
import random
import json
import re
from typing import Dict, Tuple, List

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


TARGET_DATASET_NAME = "CR"
NUM_SAMPLES = 100
RANDOM_SEED = None

DATASET_CONFIGS: Dict[str, Dict[str, Dict[int, str]]] = {
    "CR": {
        "path": "data/CR/CR_Test.json",
        "labels": {0: "positive", 1: "negative"},
    },
    "TREC": {
        "path": "data/TREC/TREC_Test.json",
        "labels": {
            0: "abbreviation",
            1: "description",
            2: "entity",
            3: "human",
            4: "location",
            5: "numeric",
        },
    },
    "DBLP": {
        "path": "data/dblp/dblp_Test.json",
        "labels": {
            0: "Database (DB)",
            1: "Artificial Intelligence (AI)",
            2: "Software Engineering (SE/CA)",
            3: "Computer Networks (NET)",
            4: "Data Mining (DM)",
            5: "Security (SEC)",
        },
    },
    "Biomedical": {
        "path": "data/Biomedical/Biomedical_Test.json",
        "labels": {
            0: "aging",
            1: "chemistry",
            2: "cats",
            3: "glucose",
            4: "potassium",
            5: "lung",
            6: "erythrocytes",
            7: "lymphocytes",
            8: "spleen",
            9: "mutation",
            10: "skin",
            11: "norepinephrine",
            12: "insulin",
            13: "prognosis",
            14: "risk",
            15: "myocardium",
            16: "sodium",
            17: "mathematics",
            18: "swine",
            19: "temperature",
        },
    },
    "R52": {
        "path": "data/R52/R52_Test.json",
        "labels": {
            0: "cocoa",
            1: "earn",
            2: "acq",
            3: "copper",
            4: "housing",
            5: "money-supply",
            6: "coffee",
            7: "sugar",
            8: "trade",
            9: "reserves",
            10: "ship",
            11: "cotton",
            12: "grain",
            13: "crude",
            14: "nat-gas",
            15: "cpi",
            16: "interest",
            17: "money-fx",
            18: "alum",
            19: "tin",
            20: "gold",
            21: "strategic-metal",
            22: "retail",
            23: "ipi",
            24: "iron-steel",
            25: "rubber",
            26: "heat",
            27: "jobs",
            28: "lei",
            29: "bop",
            30: "gnp",
            31: "zinc",
            32: "veg-oil",
            33: "orange",
            34: "carcass",
            35: "pet-chem",
            36: "gas",
            37: "wpi",
            38: "livestock",
            39: "lumber",
            40: "instal-debt",
            41: "meal-feed",
            42: "lead",
            43: "potato",
            44: "nickel",
            45: "cpu",
            46: "fuel",
            47: "jet",
            48: "income",
            49: "platinum",
            50: "dlr",
            51: "tea",
        },
    },
    "Ohsumed": {
        "path": "data/ohsumed/ohsumed_Test_alpaca_noCoT_updated.json",
        "labels": {
            0: "Bacterial Infections and Mycoses",
            1: "Virus Diseases",
            2: "Parasitic Diseases",
            3: "Neoplasms",
            4: "Musculoskeletal Diseases",
            5: "Digestive System Diseases",
            6: "Stomatognathic Diseases",
            7: "Respiratory Tract Diseases",
            8: "Otorhinolaryngologic Diseases",
            9: "Nervous System Diseases",
            10: "Eye Diseases",
            11: "Urologic and Male Genital Diseases",
            12: "Female Genital Diseases and Pregnancy Complications",
            13: "Cardiovascular Diseases",
            14: "Hemic and Lymphatic Diseases",
            15: "Neonatal Diseases and Abnormalities",
            16: "Skin and Connective Tissue Diseases",
            17: "Nutritional and Metabolic Diseases",
            18: "Endocrine Diseases",
            19: "Immunologic Diseases",
            20: "Disorders of Environmental Origin",
            21: "Animal Diseases",
            22: "Pathological Conditions, Signs and Symptoms",
        },
    },
}


COT_TEMPLATE = """Classify the following question into one of these categories by their ID:
{label_mapping_str}

Question: {question}

Let's classify step by step.
At the very end, output the ID of the category in the format <LABEL:ID> (e.g. <LABEL:0>).
Do not output the category name, only the ID number.
"""


MAX_NEW_TOKENS = 256
TEMPERATURE = 0.4
TOP_P = 0.9
DO_SAMPLE = False
BATCH_SIZE = 512


BASE_MODEL_PATH = "/mnt/data1/TC/TextClassDemo/llama3.1-8b"
ADAPTER_PATH = "/mnt/data1/TC/TextClassDemo/LLaMA-Factory/llama3.1-8b_ohsumed_lora_english_zeroshotCoT"
USE_LORA = True


def get_dataset_config(dataset_name: str) -> Tuple[str, Dict[int, str]]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dataset_config = DATASET_CONFIGS[dataset_name]
    return dataset_config["path"], dataset_config["labels"]


def build_label_mapping_str(id2label_map: Dict[int, str]) -> str:
    return "\n".join(f"{label_id}: {label}" for label_id, label in id2label_map.items())


def load_data_random(data_path: str, num_samples: int) -> List[Dict[str, str]]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    if num_samples is None or num_samples >= len(data):
        return data

    return random.sample(data, num_samples)


def resolve_ground_truth_id(sample: Dict[str, str], id2label_map: Dict[int, str]) -> Tuple[int, str]:
    raw_label = sample.get("label", sample.get("output"))
    if raw_label is None:
        return None, None

    if isinstance(raw_label, int):
        true_id = raw_label
    elif isinstance(raw_label, str) and raw_label.isdigit():
        true_id = int(raw_label)
    elif isinstance(raw_label, str) and raw_label.lower().startswith("c") and raw_label[1:].isdigit():
        true_id = int(raw_label[1:]) - 1
    else:
        label2id = {v.lower(): k for k, v in id2label_map.items()}
        clean_label = str(raw_label).lower().strip()
        true_id = label2id.get(clean_label)

    if true_id in id2label_map:
        return true_id, id2label_map[true_id]

    return None, raw_label


def extract_text(sample: Dict[str, str]) -> str:
    for key in ("input", "text", "content", "sentence", "abstract"):
        if key in sample:
            return sample[key]
    raise KeyError(f"Missing text field in sample: {sample.keys()}")


def load_dataset(data_path: str, id2label_map: Dict[int, str], num_samples: int) -> Dataset:
    data = load_data_random(data_path, num_samples)

    texts = []
    labels = []
    for sample in data:
        text = extract_text(sample)
        label_id, _ = resolve_ground_truth_id(sample, id2label_map)
        if label_id is None:
            continue
        texts.append(text)
        labels.append(label_id)

    return Dataset.from_dict({"text": texts, "label": labels})


def build_prompt(text: str, id2label_map: Dict[int, str]) -> str:
    return COT_TEMPLATE.format(
        label_mapping_str=build_label_mapping_str(id2label_map),
        question=text,
    )


def extract_category(output: str, num_classes: int) -> int:
    match = re.search(r"<LABEL:\s*(\d+)\s*>", output)
    if match:
        label_id = int(match.group(1))
        if 0 <= label_id < num_classes:
            return label_id

    fallback_digits = re.findall(r"\b(\d+)\b", output)
    for digit in reversed(fallback_digits):
        label_id = int(digit)
        if 0 <= label_id < num_classes:
            return label_id

    return -1


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], output_dir: str, num_classes: int):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()


def analyze_errors(texts: List[str], labels: List[int], preds: List[int],
                   outputs: List[str], output_dir: str) -> List[Dict[str, str]]:
    errors = []
    for i, (text, label, pred, output) in enumerate(zip(texts, labels, preds, outputs)):
        if label != pred:
            errors.append({
                "index": i,
                "text": text,
                "true_label": label,
                "predicted_label": pred if pred != -1 else "未分类",
                "model_output": output,
            })

    with open(os.path.join(output_dir, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    return errors


def main():
    data_path, label_map = get_dataset_config(TARGET_DATASET_NAME)
    num_classes = len(label_map)

    if USE_LORA:
        output_dir = f"./outputs/{TARGET_DATASET_NAME.lower()}_lora_model"
    else:
        output_dir = f"./outputs/{TARGET_DATASET_NAME.lower()}_base_model"

    os.makedirs(output_dir, exist_ok=True)

    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Chat template: {tokenizer.chat_template}")

    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    if USE_LORA:
        print("加载PEFT适配器...")
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        print("使用基础模型，不加载LoRA适配器")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    print("加载测试数据...")
    test_dataset = load_dataset(data_path, label_map, NUM_SAMPLES)
    texts = test_dataset["text"]
    labels = test_dataset["label"]

    print("开始批处理推理...")
    preds = []
    outputs = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="处理批次"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_prompts = [build_prompt(text, label_map) for text in batch_texts]

        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        batch_chat_inputs = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ) for messages in batch_messages
        ]

        batch_inputs = tokenizer(
            batch_chat_inputs,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            batch_generated_ids = model.generate(
                **batch_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for input_ids, generated_ids_sample in zip(batch_inputs["input_ids"], batch_generated_ids):
            new_tokens = generated_ids_sample[input_ids.shape[0]:]
            output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(output)
            pred = extract_category(output, num_classes)
            preds.append(pred)

    print("计算评估指标...")
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, labels=list(range(num_classes)), output_dict=True)

    print("绘制混淆矩阵...")
    plot_confusion_matrix(labels, preds, output_dir, num_classes)

    print("分析错误样本...")
    errors = analyze_errors(texts, labels, preds, outputs, output_dir)

    results = {
        "accuracy": acc,
        "report": report,
        "error_count": len(errors),
        "total_samples": len(texts),
        "outputs": outputs,
        "use_lora": USE_LORA,
        "model_type": "LoRA" if USE_LORA else "Base Model",
        "dataset": TARGET_DATASET_NAME,
    }

    with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n评估结果：")
    print(f"模型类型: {'LoRA' if USE_LORA else 'Base Model'}")
    print(f"数据集: {TARGET_DATASET_NAME}")
    print(f"准确率：{acc:.4f}")
    print(f"错误样本数：{len(errors)}")
    print(f"总样本数：{len(texts)}")
    print(f"结果保存到: {output_dir}")
    print("\n分类报告：")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def test_show_outputs():
    data_path, label_map = get_dataset_config(TARGET_DATASET_NAME)
    num_classes = len(label_map)

    print("加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Chat template: {tokenizer.chat_template}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    if USE_LORA:
        print("加载PEFT适配器...")
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        print("使用基础模型，不加载LoRA适配器")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    print("加载测试数据...")
    test_dataset = load_dataset(data_path, label_map, NUM_SAMPLES)
    texts = test_dataset["text"]
    labels = test_dataset["label"]

    indices = np.random.choice(len(texts), min(5, len(texts)), replace=False)

    print(f"\n使用模型: {'LoRA' if USE_LORA else 'Base Model'}")
    print("示例输出：")
    for i, idx in enumerate(indices):
        text = texts[idx]
        label = labels[idx]
        prompt = build_prompt(text, label_map)

        messages = [{"role": "user", "content": prompt}]
        chat_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat_input, return_tensors="pt", max_length=2048, truncation=True).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_category(output, num_classes)

        print(f"\n样本{i + 1}：")
        print(f"文本: {text[:200]}...")
        print(f"真实类别: {label}")
        print(f"预测类别: {pred if pred != -1 else '未分类'}")
        print(f"模型推理过程:\n{output}")
        print("-" * 80)


if __name__ == "__main__":
    main()
    # test_show_outputs()
