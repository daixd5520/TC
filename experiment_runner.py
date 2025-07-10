"""
实验运行器
支持配置文件和命令行参数的多轮实验
"""
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
from typing import List, Tuple, Dict, Any
import logging
from datetime import datetime

from utils.config_manager import ConfigManager, ExperimentConfig, PromptManager


class MedicalTextClassifier:
    """医学文本分类器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.logger = self._setup_logger()
        self.prompt_manager = PromptManager()
        self.num_classes = self.config.data.num_classes
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"MedicalClassifier_{self.config.output.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # 创建输出目录
        output_dir = self.config.get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        # 文件处理器
        log_file = os.path.join(output_dir, "experiment.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_model(self):
        """加载模型和tokenizer"""
        self.logger.info("开始加载模型和tokenizer...")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_path, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        self.logger.info(f"Chat template: {self.tokenizer.chat_template}")
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # 加载LoRA适配器（如果启用）
        if self.config.model.use_lora:
            self.logger.info("加载PEFT适配器...")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config.model.adapter_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            self.logger.info("使用基础模型，不加载LoRA适配器")
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        
        self.logger.info("模型加载完成")
    
    def load_dataset(self) -> Dataset:
        """加载数据集"""
        self.logger.info(f"加载数据集: {self.config.data.data_path}")
        
        with open(self.config.data.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['input'] for item in data]
        labels = [int(item['output'][1:]) - 1 for item in data]  # C1-C23 -> 0-22
        
        return Dataset.from_dict({"text": texts, "label": labels})
    
    def build_prompt(self, text: str) -> str:
        """构建提示词"""
        return self.prompt_manager.build_prompt(
            dataset_name=self.config.data.dataset_name,
            text=text,
            category_mapping=""  # 添加默认的category_mapping参数
        )
    
    def extract_category(self, output: str) -> int:
        """从推理过程中提取类别编号"""
        match = re.search(r"最终分类结果：C(\d{2})", output)
        if match:
            category_num = int(match.group(1))
            if 1 <= category_num <= self.num_classes:
                return category_num - 1
        match = re.search(r"C(\d{2})", output)
        if match:
            category_num = int(match.group(1))
            if 1 <= category_num <= self.num_classes:
                return category_num - 1
        return -1
    
    def predict_batch(self, texts: List[str]) -> Tuple[List[int], List[str]]:
        """批量预测"""
        self.logger.info("开始批量推理...")
        
        preds = []
        outputs = []
        
        for i in tqdm(range(0, len(texts), self.config.training.batch_size), desc="处理批次"):
            batch_texts = texts[i:i+self.config.training.batch_size]
            batch_prompts = [self.build_prompt(text) for text in batch_texts]
            
            # 构建批处理的对话格式
            batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
            batch_chat_inputs = [
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                ) for messages in batch_messages
            ]
            
            # 批处理tokenization
            batch_inputs = self.tokenizer(
                batch_chat_inputs, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True
            ).to(self.model.device)
            
            with torch.no_grad():
                batch_generated_ids = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=self.config.generation.max_new_tokens,
                    temperature=self.config.generation.temperature,
                    top_p=self.config.generation.top_p,
                    do_sample=self.config.generation.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码每个样本的输出
            for j, (input_ids, generated_ids_sample) in enumerate(zip(batch_inputs["input_ids"], batch_generated_ids)):
                new_tokens = generated_ids_sample[input_ids.shape[0]:]
                output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                outputs.append(output)
                pred = self.extract_category(output)
                preds.append(pred)
        
        return preds, outputs
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], output_dir: str):
        """绘制并保存混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
    
    def analyze_errors(self, texts: List[str], labels: List[int], preds: List[int], 
                      outputs: List[str], output_dir: str) -> List[Dict[str, Any]]:
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
    
    def run_evaluation(self):
        """运行完整评估"""
        self.logger.info("开始运行评估实验...")
        
        # 加载模型
        self.load_model()
        
        # 加载数据集
        test_dataset = self.load_dataset()
        texts = test_dataset["text"]
        labels = test_dataset["label"]
        
        # 批量预测
        preds, outputs = self.predict_batch(texts)
        
        # 计算评估指标
        self.logger.info("计算评估指标...")
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, output_dict=True)
        
        # 获取输出目录
        output_dir = self.config.get_output_dir()
        
        # 绘制混淆矩阵
        self.logger.info("绘制混淆矩阵...")
        self.plot_confusion_matrix(labels, preds, output_dir)
        
        # 分析错误样本
        self.logger.info("分析错误样本...")
        errors = self.analyze_errors(texts, labels, preds, outputs, output_dir)
        
        # 保存评估结果
        results = {
            "accuracy": acc,
            "report": report,
            "error_count": len(errors),
            "total_samples": len(texts),
            "outputs": outputs,
            "use_lora": self.config.model.use_lora,
            "model_type": "LoRA" if self.config.model.use_lora else "Base Model",
            "config": {
                "model": {
                    "base_model_path": self.config.model.base_model_path,
                    "adapter_path": self.config.model.adapter_path,
                    "use_lora": self.config.model.use_lora
                },
                "data": {
                    "data_path": self.config.data.data_path,
                    "dataset_name": self.config.data.dataset_name
                },
                "generation": {
                    "max_new_tokens": self.config.generation.max_new_tokens,
                    "temperature": self.config.generation.temperature,
                    "top_p": self.config.generation.top_p,
                    "do_sample": self.config.generation.do_sample
                },
                "training": {
                    "batch_size": self.config.training.batch_size
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 打印主要结果
        self.logger.info("\n评估结果：")
        self.logger.info(f"模型类型: {'LoRA' if self.config.model.use_lora else 'Base Model'}")
        self.logger.info(f"数据集: {self.config.data.dataset_name}")
        self.logger.info(f"准确率：{acc:.4f}")
        self.logger.info(f"错误样本数：{len(errors)}")
        self.logger.info(f"总样本数：{len(texts)}")
        self.logger.info(f"结果保存到: {output_dir}")
        
        return results
    
    def predict_with_vote(self, text: str, vote_count: int = None) -> (int, list):
        """
        对单个文本进行多次推理并majority vote，返回最终类别和所有输出
        """
        if vote_count is None:
            vote_count = self.config.training.vote_count
        preds = []
        outputs = []
        for _ in range(vote_count):
            prompt = self.build_prompt(text)
            messages = [{"role": "user", "content": prompt}]
            chat_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(chat_input, return_tensors="pt", max_length=2048, truncation=True).to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.generation.max_new_tokens,
                    temperature=self.config.generation.temperature,
                    top_p=self.config.generation.top_p,
                    do_sample=self.config.generation.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            output = self.tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = self.extract_category(output)
            preds.append(pred)
            outputs.append(output)
        # 统计投票
        valid_preds = [p for p in preds if p != -1]
        if not valid_preds:
            final_pred = -1
        else:
            # 多数投票，平票时随机选一个
            from collections import Counter
            counter = Counter(valid_preds)
            most_common = counter.most_common()
            max_count = most_common[0][1]
            candidates = [k for k, v in most_common if v == max_count]
            import random
            final_pred = random.choice(candidates)
        return final_pred, outputs
    
    def run_test_samples(self, num_samples: int = 5, use_vote: bool = True):
        """运行测试样本展示，支持majority vote"""
        self.logger.info("开始运行测试样本展示...")
        self.load_model()
        test_dataset = self.load_dataset()
        texts = test_dataset["text"]
        labels = test_dataset["label"]
        indices = np.random.choice(len(texts), min(num_samples, len(texts)), replace=False)
        self.logger.info(f"\n使用模型: {'LoRA' if self.config.model.use_lora else 'Base Model'}")
        self.logger.info(f"数据集: {self.config.data.dataset_name}")
        self.logger.info("示例输出：")
        for i, idx in enumerate(indices):
            text = texts[idx]
            label = f"C{labels[idx]+1:02d}"
            if use_vote:
                pred, all_outputs = self.predict_with_vote(text)
                pred_label = f"C{pred+1:02d}" if pred != -1 else "未分类"
                self.logger.info(f"\n样本{i+1}：")
                self.logger.info(f"文本: {text[:200]}...")
                self.logger.info(f"真实类别: {label}")
                self.logger.info(f"预测类别: {pred_label}")
                self.logger.info(f"所有推理输出：")
                for j, out in enumerate(all_outputs):
                    self.logger.info(f"  [推理{j+1}]: {out}")
                self.logger.info("-"*80)
            else:
                prompt = self.build_prompt(text)
                messages = [{"role": "user", "content": prompt}]
                chat_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(chat_input, return_tensors="pt", max_length=2048, truncation=True).to(self.model.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.generation.max_new_tokens,
                        temperature=self.config.generation.temperature,
                        top_p=self.config.generation.top_p,
                        do_sample=self.config.generation.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                output = self.tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred = self.extract_category(output)
                pred_label = f"C{pred+1:02d}" if pred != -1 else "未分类"
                self.logger.info(f"\n样本{i+1}：")
                self.logger.info(f"文本: {text[:200]}...")
                self.logger.info(f"真实类别: {label}")
                self.logger.info(f"预测类别: {pred_label}")
                self.logger.info(f"模型推理过程:\n{output}")
                self.logger.info("-"*80)


def main():
    """主函数"""
    from utils.config_manager import create_argument_parser, ConfigManager
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    # 加载配置
    if args.config:
        # 从配置文件加载
        config = config_manager.load_from_yaml(args.config)
        print(f"从配置文件加载配置: {args.config}")
    else:
        # 从命令行参数创建配置
        config = config_manager.create_from_args(args)
        print("从命令行参数创建配置")
    
    # 处理LoRA参数冲突
    if args.no_lora:
        config.model.use_lora = False
    
    # 创建分类器并运行
    classifier = MedicalTextClassifier(config)
    
    if args.mode == "eval":
        classifier.run_evaluation()
    elif args.mode == "test":
        classifier.run_test_samples()


if __name__ == "__main__":
    main()