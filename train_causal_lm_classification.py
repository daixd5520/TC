import os
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ohsumed_dataset(data_dir):
    def load_json_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    train_path = os.path.join(data_dir, "ohsumed_training.json")
    test_path = os.path.join(data_dir, "ohsumed_test.json")
    train_data = load_json_file(train_path)
    test_data = load_json_file(test_path)
    return train_data, test_data

def build_prompt(text):
    return (
        "你是一个医疗文本分类专家。你的任务是将给定的医疗文本分类到23个类别（C01-C23）中的一个。\n\n"
        f"文本: {text}\n\n"
        "请分析这段文本并给出分类结果。"
    )

def build_target(text, category):
    """构建简洁的因果关系目标输出"""
    # 提取文本的第一句话作为主要内容
    main_content = text.split('.')[0].strip()
    return f"这是一篇关于{main_content}的研究，因此属于{category}类别。"

def preprocess_function(examples, tokenizer, max_length=2048):
    """预处理函数，构建输入和目标"""
    inputs = [build_prompt(text) for text in examples["input"]]
    targets = [build_target(text, cat) for text, cat in zip(examples["input"], examples["category"])]
    
    # 构建完整的对话格式
    messages = []
    for inp, tgt in zip(inputs, targets):
        messages.append([
            {"role": "user", "content": inp},
            {"role": "assistant", "content": tgt}
        ])
    
    # 应用聊天模板
    full_texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                 for msg in messages]
    
    # 对文本进行编码
    model_inputs = tokenizer(
        full_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # 设置标签
    labels = model_inputs["input_ids"].clone()
    for i, msg in enumerate(messages):
        # 计算用户输入部分的长度
        user_input = tokenizer(msg[0]["content"], truncation=True, max_length=max_length)["input_ids"]
        prompt_len = len(user_input)
        # 将用户输入部分的标签设为-100
        labels[i, :prompt_len] = -100
    
    model_inputs["labels"] = labels
    return model_inputs

def main():
    # 路径与train.py一致
    base_model_path = "/root/autodl-tmp/miggs/llama3.1_3.18/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    data_dir = "/root/autodl-tmp/miggs/ohsumed_converted"
    output_dir = "/root/autodl-tmp/miggs/outputs/ohsumed_causal_lm_classification_simple"
    os.makedirs(output_dir, exist_ok=True)

    # 量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载数据
    train_data, test_data = load_ohsumed_dataset(data_dir)
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    # 预处理
    def preprocess_train(examples):
        return preprocess_function(examples, tokenizer)
    train_dataset = train_dataset.map(preprocess_train, batched=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(preprocess_train, batched=True, remove_columns=test_dataset.column_names)
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,  # 略微提高学习率，因为任务更简单了
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="steps",
        save_steps=300,
        evaluation_strategy="steps",
        eval_steps=300,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="tensorboard",
        seed=42,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_steps=-1,
        save_total_limit=3,
        dataloader_num_workers=0
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # 训练
    trainer.train()
    trainer.save_model()
    print("训练完成，模型已保存。")

if __name__ == "__main__":
    main() 