import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoConfig,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import logging
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ohsumed_dataset(data_dir):
    """加载OHSUMED数据集
    
    Args:
        data_dir: 数据集目录，包含ohsumed_training.json和ohsumed_test.json
        
    Returns:
        dict: 包含训练集和测试集的字典
    """
    def load_json_file(file_path):
        print(f"正在读取文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 读取整个文件内容
                content = f.read()
                # 解析JSON数据
                data = json.loads(content)
                
                # 验证数据格式
                if not isinstance(data, list):
                    print(f"错误：数据不是列表格式")
                    return []
                
                # 验证每个数据项
                valid_data = []
                for i, item in enumerate(data, 1):
                    if not isinstance(item, dict):
                        print(f"警告：第{i}项不是字典格式")
                        continue
                    
                    # 检查必需字段
                    required_fields = ["instruction", "input", "output", "category"]
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        print(f"警告：第{i}项缺少字段 {missing_fields}")
                        continue
                    
                    # 检查字段类型
                    if not all(isinstance(item[field], str) for field in required_fields):
                        print(f"警告：第{i}项字段类型不正确")
                        continue
                    
                    valid_data.append(item)
                    
                    # 打印前5条数据
                    if i <= 5:
                        print(f"数据示例 {i}:")
                        print(json.dumps(item, indent=2, ensure_ascii=False))
                
                print(f"成功加载 {len(valid_data)} 条数据")
                return valid_data
                
        except json.JSONDecodeError as e:
            print(f"错误：JSON解析失败: {e}")
            return []
        except Exception as e:
            print(f"错误：无法读取文件 {file_path}: {e}")
            return []
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误：目录不存在: {data_dir}")
        return None
        
    # 加载训练集和测试集
    train_path = os.path.join(data_dir, "ohsumed_training.json")
    test_path = os.path.join(data_dir, "ohsumed_test.json")
    
    if not os.path.exists(train_path):
        print(f"错误：训练集文件不存在: {train_path}")
        return None
        
    if not os.path.exists(test_path):
        print(f"错误：测试集文件不存在: {test_path}")
        return None
    
    print("\n=== 加载训练集 ===")
    train_data = load_json_file(train_path)
    
    print("\n=== 加载测试集 ===")
    test_data = load_json_file(test_path)
    
    if not train_data or not test_data:
        print("错误：数据加载失败")
        return None
    
    # 统计类别分布
    train_categories = {}
    test_categories = {}
    
    for item in train_data:
        cat = item["category"]
        train_categories[cat] = train_categories.get(cat, 0) + 1
    
    for item in test_data:
        cat = item["category"]
        test_categories[cat] = test_categories.get(cat, 0) + 1
    
    print("\n=== 类别分布 ===")
    print("训练集:")
    for cat, count in sorted(train_categories.items()):
        print(f"{cat}: {count}")
    print("\n测试集:")
    for cat, count in sorted(test_categories.items()):
        print(f"{cat}: {count}")
    
    return {
        "train": train_data,
        "test": test_data,
        "train_categories": train_categories,
        "test_categories": test_categories
    }

def main():
    # 设置路径
    model_path = "/root/autodl-tmp/miggs/llama3.1_3.18/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    data_dir = "/root/autodl-tmp/miggs/ohsumed_converted"  # 修改为处理后的数据目录
    output_dir = "/root/autodl-tmp/miggs/outputs/ohsumed_classification"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置4-bit量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 打印tokenizer的初始配置
    print("\n=== Tokenizer 初始配置 ===")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"Pad token id: {tokenizer.pad_token_id}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"EOS token id: {tokenizer.eos_token_id}")
    print(f"Padding side: {tokenizer.padding_side}")
    
    # 设置padding token
    if tokenizer.pad_token is None:
        print("\n=== 设置 Padding Token ===")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"设置 pad_token 为: {tokenizer.pad_token}")
        print(f"设置 pad_token_id 为: {tokenizer.pad_token_id}")
    
    # 验证tokenizer配置
    print("\n=== Tokenizer 最终配置 ===")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"Pad token id: {tokenizer.pad_token_id}")
    print(f"Padding side: {tokenizer.padding_side}")
    
    # 加载数据集
    dataset = load_ohsumed_dataset(data_dir)
    if dataset is None:
        logger.error("数据集加载失败，退出训练")
        return
    
    # 将列表转换为Dataset格式
    from datasets import Dataset
    
    # 创建训练集
    train_data = {
        "text": [item["input"] for item in dataset["train"]],
        "label": [int(item["category"][1:]) - 1 for item in dataset["train"]]  # 将C1-C23转换为0-22
    }
    train_dataset = Dataset.from_dict(train_data)
    
    # 创建测试集
    test_data = {
        "text": [item["input"] for item in dataset["test"]],
        "label": [int(item["category"][1:]) - 1 for item in dataset["test"]]  # 将C1-C23转换为0-22
    }
    test_dataset = Dataset.from_dict(test_data)
    
    # 预处理数据集
    def preprocess(examples):
        # 确保只使用需要的字段
        texts = examples["text"]
        labels = examples["label"]
        
        # 构建更适合分类任务的prompt
        prompts = [
            f"### Instruction: Classify the following medical text into one of the 23 categories.\n"
            f"### Text: {text}\n"
            f"### Categories: C1-C23\n"
            f"### Response: The text belongs to category " 
            for text in texts
        ]
        
        # tokenize，使用批处理
        tokenized = tokenizer(
            prompts,
            padding="max_length",  # 使用固定长度padding
            truncation=True,
            max_length=400,  # 改为400
            return_tensors="pt"
        )
        
        # 添加标签
        tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
        
        # 打印详细的调试信息
        print("\n=== 数据预处理调试信息 ===")
        print(f"输入文本数量: {len(texts)}")
        print(f"标签数量: {len(labels)}")
        print(f"Input IDs shape: {tokenized['input_ids'].shape}")
        print(f"Attention Mask shape: {tokenized['attention_mask'].shape}")
        print(f"Labels shape: {tokenized['labels'].shape}")
        print(f"Input IDs 示例: {tokenized['input_ids'][0][:10]}")
        print(f"Labels 示例: {tokenized['labels'][0]}")
        print("========================\n")
        
        return tokenized
    
    # 转换数据集，使用批处理
    tokenized_dataset = {
        "train": train_dataset.map(
            preprocess,
            batched=True,
            batch_size=1,  # 暂时设置为1，避免padding问题
            remove_columns=train_dataset.column_names,
            desc="处理训练集",
            num_proc=1  # 使用单进程
        ),
        "test": test_dataset.map(
            preprocess,
            batched=True,
            batch_size=1,  # 暂时设置为1，避免padding问题
            remove_columns=test_dataset.column_names,
            desc="处理测试集",
            num_proc=1  # 使用单进程
        )
    }
    
    # 设置数据集格式
    tokenized_dataset["train"].set_format("torch")
    tokenized_dataset["test"].set_format("torch")
    
    # 加载模型配置
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.num_labels = 23  # 设置类别数量
    config.problem_type = "single_label_classification"  # 设置问题类型
    config.use_cache = False  # 禁用缓存以支持梯度检查点
    
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 准备模型进行k-bit训练
    model = prepare_model_for_kbit_training(model)
    
    # 设置LoRA配置
    lora_config = LoraConfig(
        r=8,  # LoRA的秩
        lora_alpha=16,  # LoRA的alpha参数
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS  # 修改为序列分类任务
    )
    
    # 获取PEFT模型
    model = get_peft_model(model, lora_config)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=1,  # 暂时使用较小的批次大小
        per_device_eval_batch_size=1,   # 暂时使用较小的批次大小
        gradient_accumulation_steps=16,  # 增加梯度累积步数
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
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
        dataloader_num_workers=0  # 设置为0以避免多进程问题
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda eval_pred: {
            "accuracy": (eval_pred.predictions.argmax(-1) == eval_pred.label_ids).mean()
        }
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    
    # 评估模型
    eval_results = trainer.evaluate()
    logger.info("评估结果: %s", eval_results)

if __name__ == "__main__":
    main() 