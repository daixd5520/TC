import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---- 1. 配置模型和路径 ----
model_name_or_path = "/mnt/data1/TC/TextClassDemo/LLaMA-Factory/ohsumed_direct_merged"
# adapter_path = "/mnt/data1/TC/TextClassDemo/LLaMA-Factory/ohsumed_direct_lora" # 你训练好的LoRA模型路径
text_to_classify = "The patient presents with symptoms of arrhythmia and chest pain, suggesting a cardiac issue." # 待分类的文本

# 推理配置
n_runs = 10 # 进行10次推理求平均
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 2. 加载模型和Tokenizer ----
print("Loading model and tokenizer...")
# 加载基础模型和Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16, # 使用bfloat16以获得更好的性能
    device_map="auto"
)
# 加载LoRA适配器
# model = PeftModel.from_pretrained(model, adapter_path)
model.eval() # 设置为评估模式

# 如果tokenizer没有pad_token，添加一个
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---- 3. 准备类别Token ----
class_labels = [f"C{str(i).zfill(2)}" for i in range(1, 24)]
# 重要：确保这些token确实在tokenizer的词汇表中
# LLaMA-Factory在训练时已经添加，但加载时最好确认
num_added_tokens = tokenizer.add_tokens(class_labels, special_tokens=True)
if num_added_tokens > 0:
    print(f"Warning: {num_added_tokens} class labels were not in the tokenizer and have been added now.")
    # 如果在这里添加了新token，理论上模型嵌入层也需要resize，
    # 但由于我们加载的是训练好的适配器，这步通常在训练时已完成。
    # model.resize_token_embeddings(len(tokenizer))

# 获取类别token的整数ID
class_token_ids = tokenizer.convert_tokens_to_ids(class_labels)
print(f"Class token IDs: {class_token_ids}")


# ---- 4. 构建Prompt ----
# 使用和训练时完全一样的模板！
instruction = "你是一位顶尖的医学信息学家，专长于 OHSUMED 数据集分类。请仔细阅读以下文本摘要。你的任务是进行精准分类，并仅返回唯一的类别ID作为结果。例如，如果文本属于心血管疾病，你的输出就应该是 'C14'。现在，请处理以下文本："
# LLaMA-3的聊天模板格式
prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n{text_to_classify}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)


# ---- 5. 执行N次推理并收集概率 ----
all_probs = []
print(f"\nPerforming {n_runs} inference runs...")
with torch.no_grad():
    for i in range(n_runs):
        # 执行前向传播获取logits
        # 注意：这里我们开启了dropout (model.train()会开启)，以引入随机性
        # 如果LoRA训练时dropout为0，可以在这里手动加一个dropout层或采用其他采样方法
        # 对于LLaMA-3，可以通过开启 temperature 和 top_p 采样来引入随机性，但这里我们直接用模型的前向传播
        model.train() # 技巧：开启dropout以在多次前向传播中引入随机性
        outputs = model(**inputs)
        logits = outputs.logits

        # 获取第一个生成位置的logits (即输入序列的最后一个token位置)
        first_gen_logits = logits[:, -1, :]

        # 转换为概率
        probabilities = torch.nn.functional.softmax(first_gen_logits, dim=-1).squeeze()

        # 提取我们关心的23个类别的概率
        class_probs = probabilities[class_token_ids].cpu()
        all_probs.append(class_probs)
        print(f"Run {i+1}: Predicted top class: {class_labels[torch.argmax(class_probs)]} with prob {torch.max(class_probs):.4f}")


# ---- 6. 平均概率并获取最终结果 ----
model.eval() # 推理结束，恢复评估模式
# 将概率列表堆叠成一个张量 (n_runs, num_classes)
probs_tensor = torch.stack(all_probs)

# 计算平均概率
avg_probs = probs_tensor.mean(dim=0)

# 找到平均概率最高的类别
final_class_index = torch.argmax(avg_probs)
final_class_label = class_labels[final_class_index]
final_class_prob = avg_probs[final_class_index]

print("\n" + "="*30)
print("           Final Result")
print("="*30)
print(f"Average probabilities for all classes: \n{avg_probs.to(torch.float32).numpy()}")
print(f"\nFinal predicted class: {final_class_label}")
print(f"Confidence (average probability): {final_class_prob:.4f}")