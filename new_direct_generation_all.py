import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm  # 加载tqdm
import csv
# ---- 1. 配置模型和路径 ----
model_name_or_path = "/mnt/data1/TC/TextClassDemo/LLaMA-Factory/ohsumed_direct_merged"
test_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Test.json"
n_runs = 1  # 每条样本推理次数，取平均
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 2. 加载模型和Tokenizer ----
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---- 3. 准备类别Token ----
class_labels = [f"C{str(i).zfill(2)}" for i in range(1, 24)]
num_added_tokens = tokenizer.add_tokens(class_labels, special_tokens=True)
if num_added_tokens > 0:
    print(f"Warning: {num_added_tokens} class labels were not in the tokenizer and have been added now.")
class_token_ids = tokenizer.convert_tokens_to_ids(class_labels)

# ---- 4. 读取测试集 ----
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# ---- 5. 批量预测 ----
instruction = f"你是一位顶尖的医学信息学家，专长于 OHSUMED 数据集分类。请仔细阅读以下文本摘要。你的任务是进行精准分类，并仅返回唯一的类别ID作为结果。例如，如果文本属于心血管疾病，你的输出就应该是 'C14'。类别ID列表为：{', '.join(class_labels)}。现在，请处理以下文本："
y_true = []
y_pred = []

# 新增：打开csv文件并写入表头
csv_path = "ohsumed_test_pred_detail.csv"
with open(csv_path, "w", encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 表头
    header = ["text", "true_label", "pred_label"] + [f"{label}_prob" for label in class_labels]
    writer.writerow(header)

    for item in tqdm(test_data, desc="Predicting"):
        text_to_classify = item["text"]
        label = item["label"]
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n{text_to_classify}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        n_runs = 10  # 采样次数
        temperature = 0.7  # 可根据需要调整
        all_probs = []
        with torch.no_grad():
            for _ in range(n_runs):
                model.eval()  # 推理时应为eval
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                # 取生成的最后一个token
                gen_token_id = gen_ids[0, -1].item()
                class_probs = torch.zeros(len(class_labels))
                if gen_token_id in class_token_ids:
                    idx = class_token_ids.index(gen_token_id)
                    class_probs[idx] = 1.0
                all_probs.append(class_probs)
        probs_tensor = torch.stack(all_probs)
        avg_probs = probs_tensor.mean(dim=0)
        final_class_index = torch.argmax(avg_probs)
        final_class_label = class_labels[final_class_index]

        y_true.append(label)
        y_pred.append(final_class_label)

        # 写入一行到csv
        row = [text_to_classify, label, final_class_label] + [float(p) for p in avg_probs]
        writer.writerow(row)

# ---- 6. 评估 ----
print("\n评估结果：")
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
print(f"准确率: {acc:.4f}")
print(f"宏平均F1: {f1:.4f}")

print("\n分类报告：")
print(classification_report(y_true, y_pred, labels=class_labels, digits=4))

print("\n混淆矩阵：")
cm = confusion_matrix(y_true, y_pred, labels=class_labels)
np.set_printoptions(linewidth=200)
print(cm)