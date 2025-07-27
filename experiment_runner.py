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
    AutoTokenizer
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
from accelerate import Accelerator
import torch.distributed as dist

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
        self.accelerator = Accelerator()
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        
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
        self.logger.info("="*80)
        self.logger.info("开始加载模型和tokenizer...")
        self.logger.info(f"基础模型路径: {self.config.model.base_model_path}")
        self.logger.info(f"是否使用LoRA: {self.config.model.use_lora}")
        if self.config.model.use_lora:
            self.logger.info(f"LoRA适配器路径: {self.config.model.adapter_path}")
        self.logger.info("="*80)
        
        import os
        def is_distributed():
            return (
                int(os.environ.get("WORLD_SIZE", "1")) > 1 or
                int(os.environ.get("LOCAL_WORLD_SIZE", "1")) > 1 or
                int(os.environ.get("LOCAL_RANK", "-1")) >= 0 or
                int(os.environ.get("RANK", "-1")) >= 0
            )
        model_path = self.config.model.base_model_path
        is_70b = "70b" in model_path.lower()
        distributed = is_distributed()
        if is_70b:
            if distributed:
                device_map = None
                self.logger.info("检测到70B大模型，当前为多进程（torchrun/accelerate），采用数据并行（device_map=None）")
            else:
                device_map = "auto"
                self.logger.info("检测到70B大模型，单进程，采用模型并行（device_map=auto）")
        else:
            device_map = None
            if distributed:
                self.logger.info("非70B模型，当前为多进程，采用数据并行（device_map=None）")
            else:
                self.logger.info("非70B模型，单进程，采用单卡推理（device_map=None）")
        
        # 加载tokenizer
        self.logger.info(f"正在加载tokenizer: {self.config.model.base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_path, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        self.logger.info(f"Tokenizer加载完成")
        self.logger.info(f"Chat template: {self.tokenizer.chat_template}")
        
        # 检测模型类型并设置模板策略
        self._setup_template_strategy()
        
        # 加载基础模型
        self.logger.info(f"正在加载基础模型: {self.config.model.base_model_path}")

        # 强制全部加载到cuda:0，避免device_map="auto"导致碎片化或部分权重在CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_path,
            # device_map={"": "cuda:0"},
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        # 不需要再 .to(self.accelerator.device)

        self.logger.info(f"基础模型加载完成")
        
        # 加载LoRA适配器（如果启用）
        if self.config.model.use_lora:
            self.logger.info(f"正在加载PEFT适配器: {self.config.model.adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config.model.adapter_path,
                device_map={"": "cuda:0"},
                torch_dtype=torch.float16
            )
            # 不需要再 .to(self.accelerator.device)
            self.logger.info("PEFT适配器加载完成")
        else:
            self.logger.info("使用基础模型，不加载LoRA适配器")
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        
        self.logger.info("="*80)
        self.logger.info("模型加载完成！")
        self.logger.info(f"最终使用的模型: {'LoRA适配模型' if self.config.model.use_lora else '基础模型'}")
        self.logger.info(f"模型路径: {self.config.model.base_model_path}")
        if self.config.model.use_lora:
            self.logger.info(f"适配器路径: {self.config.model.adapter_path}")
        self.logger.info(f"模板策略: {self.template_strategy}")
        self.logger.info("="*80)
    
    def _setup_template_strategy(self):
        """设置模板策略，根据模型类型选择合适的模板方式"""
        model_name = self.config.model.base_model_path.lower()
        # 检测模型类型
        if ("llama" in model_name or "llama3" in model_name):
            if getattr(self.tokenizer, "chat_template", None):
                self.template_strategy = "chat_template"
                self.logger.info("检测到Llama模型，使用chat_template")
            else:
                self.template_strategy = "direct_prompt"
                self.logger.info("检测到Llama模型但无chat_template，降级为直接提示词")
        elif "gemma" in model_name:
            self.template_strategy = "direct_prompt"
            self.logger.info("检测到Gemma模型，使用直接提示词（避免重复输出）")
        elif "qwen" in model_name:
            if getattr(self.tokenizer, "chat_template", None):
                self.template_strategy = "chat_template"
                self.logger.info("检测到Qwen模型，使用chat_template")
            else:
                self.template_strategy = "direct_prompt"
                self.logger.info("检测到Qwen模型但无chat_template，降级为直接提示词")
        elif "chatglm" in model_name:
            self.template_strategy = "direct_prompt"
            self.logger.info("检测到ChatGLM模型，使用直接提示词")
        else:
            # 默认策略：如果tokenizer有chat_template就使用，否则直接提示词
            if getattr(self.tokenizer, "chat_template", None):
                self.template_strategy = "chat_template"
                self.logger.info("使用默认chat_template策略")
            else:
                self.template_strategy = "direct_prompt"
                self.logger.info("模型无chat_template，使用直接提示词")
        self.logger.info(f"模板策略: {self.template_strategy}")
    
    def _format_prompt(self, prompt: str) -> str:
        """根据模板策略格式化提示词"""
        if self.template_strategy == "chat_template":
            # 使用chat template
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 直接使用提示词
            return prompt
    
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
            text=text
        )
    
    def extract_category(self, output: str) -> int:
        """从推理过程中提取类别编号"""
        # 方案1：优先查找"最终分类结果：Cxx"格式
        matches = re.findall(r"最终分类结果：C(\d{2})", output)
        if matches:
            category_num = int(matches[-1])  # 取最后一个匹配
            if 1 <= category_num <= self.num_classes:
                return category_num - 1
        
        # 方案2：查找"答案是Cxx"或"结论是Cxx"等明确结论格式
        conclusion_patterns = [
            r"答案是\s*C(\d{2})",
            r"结论是\s*C(\d{2})", 
            r"因此\s*C(\d{2})",
            r"所以\s*C(\d{2})",
            r"最终答案\s*C(\d{2})",
            r"分类结果\s*C(\d{2})"
        ]
        
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, output)
            if matches:
                category_num = int(matches[-1])
                if 1 <= category_num <= self.num_classes:
                    return category_num - 1
        
        # 方案3：查找所有Cxx格式，但优先选择后半部分的匹配
        all_matches = list(re.finditer(r"C(\d{2})", output))
        if all_matches:
            # 如果只有一个匹配，直接使用
            if len(all_matches) == 1:
                category_num = int(all_matches[0].group(1))
                if 1 <= category_num <= self.num_classes:
                    return category_num - 1
            else:
                # 如果有多个匹配，优先选择后半部分的匹配
                mid_point = len(output) // 2
                for match in reversed(all_matches):  # 从后往前遍历
                    if match.start() >= mid_point:  # 在后半部分
                        category_num = int(match.group(1))
                        if 1 <= category_num <= self.num_classes:
                            return category_num - 1
                
                # 如果后半部分没有有效匹配，使用最后一个
                category_num = int(all_matches[-1].group(1))
                if 1 <= category_num <= self.num_classes:
                    return category_num - 1
        
        # 方案4：兜底方案 - 查找数字并转换为Cxx格式
        # 优先查找后半部分的数字（先找二位数字，再找一位数字）
        
        # 4.1 查找二位数字
        all_digit_matches = list(re.finditer(r"\b(\d{2})\b", output))
        if all_digit_matches:
            # 如果只有一个匹配，直接使用
            if len(all_digit_matches) == 1:
                category_num = int(all_digit_matches[0].group(1))
                if 1 <= category_num <= self.num_classes:
                    # 将数字转换为Cxx格式，然后重新匹配
                    converted_output = output.replace(
                        all_digit_matches[0].group(0), 
                        f"C{category_num:02d}"
                    )
                    # 重新在转换后的文本中查找Cxx格式
                    c_matches = re.findall(r"C(\d{2})", converted_output)
                    if c_matches:
                        return int(c_matches[-1]) - 1
                    return category_num - 1
            else:
                # 如果有多个匹配，优先选择后半部分的匹配
                mid_point = len(output) // 2
                for match in reversed(all_digit_matches):  # 从后往前遍历
                    if match.start() >= mid_point:  # 在后半部分
                        category_num = int(match.group(1))
                        if 1 <= category_num <= self.num_classes:
                            # 将数字转换为Cxx格式，然后重新匹配
                            converted_output = output.replace(
                                match.group(0), 
                                f"C{category_num:02d}"
                            )
                            # 重新在转换后的文本中查找Cxx格式
                            c_matches = re.findall(r"C(\d{2})", converted_output)
                            if c_matches:
                                return int(c_matches[-1]) - 1
                            return category_num - 1
                
                # 如果后半部分没有有效匹配，使用最后一个
                category_num = int(all_digit_matches[-1].group(1))
                if 1 <= category_num <= self.num_classes:
                    # 将数字转换为Cxx格式，然后重新匹配
                    converted_output = output.replace(
                        all_digit_matches[-1].group(0), 
                        f"C{category_num:02d}"
                    )
                    # 重新在转换后的文本中查找Cxx格式
                    c_matches = re.findall(r"C(\d{2})", converted_output)
                    if c_matches:
                        return int(c_matches[-1]) - 1
                    return category_num - 1
        
        # 4.2 查找一位数字（1-9）
        all_single_digit_matches = list(re.finditer(r"\b(\d)\b", output))
        if all_single_digit_matches:
            # 如果只有一个匹配，直接使用
            if len(all_single_digit_matches) == 1:
                category_num = int(all_single_digit_matches[0].group(1))
                if 1 <= category_num <= 9:  # 一位数字只考虑1-9
                    # 将数字转换为Cxx格式，然后重新匹配
                    converted_output = output.replace(
                        all_single_digit_matches[0].group(0), 
                        f"C{category_num:02d}"
                    )
                    # 重新在转换后的文本中查找Cxx格式
                    c_matches = re.findall(r"C(\d{2})", converted_output)
                    if c_matches:
                        return int(c_matches[-1]) - 1
                    return category_num - 1
            else:
                # 如果有多个匹配，优先选择后半部分的匹配
                mid_point = len(output) // 2
                for match in reversed(all_single_digit_matches):  # 从后往前遍历
                    if match.start() >= mid_point:  # 在后半部分
                        category_num = int(match.group(1))
                        if 1 <= category_num <= 9:  # 一位数字只考虑1-9
                            # 将数字转换为Cxx格式，然后重新匹配
                            converted_output = output.replace(
                                match.group(0), 
                                f"C{category_num:02d}"
                            )
                            # 重新在转换后的文本中查找Cxx格式
                            c_matches = re.findall(r"C(\d{2})", converted_output)
                            if c_matches:
                                return int(c_matches[-1]) - 1
                            return category_num - 1
                
                # 如果后半部分没有有效匹配，使用最后一个
                category_num = int(all_single_digit_matches[-1].group(1))
                if 1 <= category_num <= 9:  # 一位数字只考虑1-9
                    # 将数字转换为Cxx格式，然后重新匹配
                    converted_output = output.replace(
                        all_single_digit_matches[-1].group(0), 
                        f"C{category_num:02d}"
                    )
                    # 重新在转换后的文本中查找Cxx格式
                    c_matches = re.findall(r"C(\d{2})", converted_output)
                    if c_matches:
                        return int(c_matches[-1]) - 1
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
    
    def gather_object_across_processes(self, obj):
        """
        用于多进程收集 Python 对象（如list），兼容单卡和多卡
        """
        if self.world_size == 1:
            return [obj]
        output_list = [None for _ in range(self.world_size)]
        dist.all_gather_object(output_list, obj)
        return output_list

    def run_evaluation(self):
        """运行完整评估"""
        self.logger.info("开始运行评估实验...")
        self.logger.info(f"实验配置:")
        self.logger.info(f"  - 基础模型: {self.config.model.base_model_path}")
        self.logger.info(f"  - 使用LoRA: {self.config.model.use_lora}")
        if self.config.model.use_lora:
            self.logger.info(f"  - 适配器路径: {self.config.model.adapter_path}")
        self.logger.info(f"  - 数据集: {self.config.data.dataset_name}")
        self.logger.info(f"  - 数据路径: {self.config.data.data_path}")
        
        # 加载模型
        self.load_model()
        
        # 加载数据集
        test_dataset = self.load_dataset()
        texts = test_dataset["text"]
        labels = test_dataset["label"]

        # ====== 数据分片，每个进程只处理自己那一份 ======
        total = len(texts)
        indices = list(range(total))
        my_indices = indices[self.rank::self.world_size]
        my_texts = [texts[i] for i in my_indices]
        my_labels = [labels[i] for i in my_indices]
        my_index_list = my_indices  # 新增
        # 推理
        if self.config.training.vote_count > 1:
            self.logger.info(f"使用投票推理模式，投票次数：{self.config.training.vote_count}")
            my_preds = []
            my_outputs = []
            for i in tqdm(range(len(my_texts)), desc="投票推理"):
                pred, all_outputs = self.predict_with_vote(my_texts[i])
                my_preds.append(pred)
                my_outputs.append(all_outputs)
        else:
            self.logger.info("使用单一推理模式")
            my_preds, my_outputs = self.predict_batch(my_texts)

        # ====== gather 前补齐到最大长度 ======
        # 1. 先 gather 所有进程的长度
        my_len = len(my_preds)
        len_tensor = torch.tensor([my_len], device=self.accelerator.device)
        all_lens = self.accelerator.gather_for_metrics(len_tensor).tolist()
        max_len = max(all_lens)

        pad_num = max_len - my_len
        my_preds_pad = my_preds + [-1] * pad_num
        my_labels_pad = my_labels + [-1] * pad_num
        my_index_list_pad = my_index_list + [-1] * pad_num
        my_outputs_pad = my_outputs + [None] * pad_num

        # 3. gather
        preds_tensor = torch.tensor(my_preds_pad, device=self.accelerator.device)
        labels_tensor = torch.tensor(my_labels_pad, device=self.accelerator.device)
        index_tensor = torch.tensor(my_index_list_pad, device=self.accelerator.device)

        all_preds = self.accelerator.gather_for_metrics(preds_tensor).tolist()
        all_labels = self.accelerator.gather_for_metrics(labels_tensor).tolist()
        all_indices = self.accelerator.gather_for_metrics(index_tensor).tolist()
        all_outputs = self.gather_object_across_processes(my_outputs_pad)  # 还是用 all_gather_object

        # 4. flatten
        flat_outputs = []
        for part in all_outputs:
            flat_outputs.extend(part)

        # 5. 只保留有效 index
        valid = [(idx, pred, label, out) for idx, pred, label, out in zip(all_indices, all_preds, all_labels, flat_outputs) if idx != -1]
        valid.sort(key=lambda x: x[0])  # 按原始顺序
        all_preds = [v[1] for v in valid]
        all_labels = [v[2] for v in valid]
        flat_outputs = [v[3] for v in valid]

        # 只在主进程保存和输出
        if self.rank == 0:
            # 计算评估指标
            self.logger.info("计算评估指标...")
            acc = accuracy_score(all_labels, all_preds)
            report = classification_report(all_labels, all_preds, output_dict=True)
            
            # 获取输出目录
            output_dir = self.config.get_output_dir()
            
            # 绘制混淆矩阵
            self.logger.info("绘制混淆矩阵...")
            self.plot_confusion_matrix(all_labels, all_preds, output_dir)
            
            # 分析错误样本
            self.logger.info("分析错误样本...")
            # 合并所有进程的 outputs
            errors = self.analyze_errors(texts, labels, all_preds, flat_outputs, output_dir)
            
            # 保存评估结果
            results = {
                "accuracy": acc,
                "report": report,
                "error_count": len(errors),
                "total_samples": len(texts),
                "outputs": flat_outputs,
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
                        "batch_size": self.config.training.batch_size,
                        "vote_count": self.config.training.vote_count
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
            self.logger.info(f"推理模式: {'投票推理' if self.config.training.vote_count > 1 else '单一推理'}")
            if self.config.training.vote_count > 1:
                self.logger.info(f"投票次数: {self.config.training.vote_count}")
            self.logger.info(f"准确率：{acc:.4f}")
            self.logger.info(f"错误样本数：{len(errors)}")
            self.logger.info(f"总样本数：{len(texts)}")
            self.logger.info(f"结果保存到: {output_dir}")
            return results
        else:
            return None
    

    
    def predict_with_vote(self, text: str, vote_count: int = None) -> tuple[int, list]:
        """
        对单个文本进行多次推理并智能投票，返回最终类别和所有输出
        改进策略：
        1. 使用不同的生成参数增加多样性
        2. 基于置信度的加权投票
        3. 考虑推理质量的自适应权重
        """
        if vote_count is None:
            vote_count = self.config.training.vote_count
        
        preds = []
        outputs = []
        confidences = []
        
        # 使用不同的生成参数增加多样性
        temperature_variations = [0.3, 0.5, 0.7, 0.9, 1.1]  # 不同的温度值
        top_p_variations = [0.8, 0.85, 0.9, 0.95, 0.98]    # 不同的top_p值
        
        for i in range(vote_count):
            # 随机选择生成参数
            temp = temperature_variations[i % len(temperature_variations)]
            top_p = top_p_variations[i % len(top_p_variations)]
            
            prompt = self.build_prompt(text)
            formatted_prompt = self._format_prompt(prompt)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.generation.max_new_tokens,
                    temperature=temp,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            output = self.tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = self.extract_category(output)
            preds.append(pred)
            outputs.append(output)
            
            # 计算置信度分数
            confidence = self._calculate_confidence(output, pred)
            confidences.append(confidence)
        
        # 智能投票策略
        final_pred = self._smart_voting(preds, confidences)
        
        return final_pred, outputs
    
    def _calculate_confidence(self, output: str, pred: int) -> float:
        """
        计算推理输出的置信度分数
        基于多个因素：推理长度、结论明确性、关键词匹配等
        """
        if pred == -1:
            return 0.0
        
        confidence = 0.0
        
        # 1. 推理长度分数 (适中的长度更好)
        length = len(output)
        if 100 <= length <= 500:
            confidence += 0.3
        elif 50 <= length <= 800:
            confidence += 0.2
        else:
            confidence += 0.1
        
        # 2. 结论明确性分数
        conclusion_keywords = [
            "最终分类结果", "答案是", "结论是", "因此", "所以", 
            "最终答案", "分类结果", "综上所述"
        ]
        for keyword in conclusion_keywords:
            if keyword in output:
                confidence += 0.2
                break
        
        # 3. 推理逻辑分数 (包含分析步骤)
        reasoning_keywords = [
            "分析", "考虑", "因为", "由于", "基于", "根据", 
            "特征", "症状", "表现", "检查", "诊断"
        ]
        reasoning_count = sum(1 for keyword in reasoning_keywords if keyword in output)
        confidence += min(reasoning_count * 0.1, 0.3)
        
        # 4. 类别编号出现次数 (多次出现可能更确定)
        category_pattern = f"C{pred+1:02d}"
        category_count = output.count(category_pattern)
        confidence += min(category_count * 0.1, 0.2)
        
        return min(confidence, 1.0)
    
    def _smart_voting(self, preds: List[int], confidences: List[float]) -> int:
        """
        智能投票策略
        1. 基于置信度的加权投票
        2. 考虑推理质量
        3. 处理平票情况
        """
        from collections import Counter, defaultdict
        
        # 过滤无效预测
        valid_data = [(pred, conf) for pred, conf in zip(preds, confidences) if pred != -1]
        
        if not valid_data:
            return -1
        
        # 计算加权投票
        weighted_votes = defaultdict(float)
        for pred, conf in valid_data:
            weighted_votes[pred] += conf
        
        # 找到最高权重的预测
        max_weight = max(weighted_votes.values())
        best_candidates = [pred for pred, weight in weighted_votes.items() if weight == max_weight]
        
        if len(best_candidates) == 1:
            return best_candidates[0]
        
        # 处理平票情况：使用传统多数投票作为tie-breaker
        valid_preds = [pred for pred in preds if pred != -1]
        if not valid_preds:
            return -1
        
        counter = Counter(valid_preds)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        tie_candidates = [k for k, v in most_common if v == max_count]
        
        # 在平票候选中选择置信度最高的
        if len(tie_candidates) == 1:
            return tie_candidates[0]
        
        # 如果还是平票，选择置信度最高的
        best_confidence = 0.0
        best_pred = tie_candidates[0]
        
        for pred in tie_candidates:
            pred_confidences = [conf for p, conf in zip(preds, confidences) if p == pred]
            if pred_confidences:
                avg_confidence = sum(pred_confidences) / len(pred_confidences)
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_pred = pred
        
        return best_pred
    
    def run_test_samples(self, num_samples: int = 5):
        """运行测试样本展示，根据配置自动选择推理模式"""
        self.logger.info("开始运行测试样本展示...")
        self.logger.info(f"测试配置:")
        self.logger.info(f"  - 基础模型: {self.config.model.base_model_path}")
        self.logger.info(f"  - 使用LoRA: {self.config.model.use_lora}")
        if self.config.model.use_lora:
            self.logger.info(f"  - 适配器路径: {self.config.model.adapter_path}")
        self.logger.info(f"  - 数据集: {self.config.data.dataset_name}")
        self.logger.info(f"  - 测试样本数: {num_samples}")
        
        self.load_model()
        test_dataset = self.load_dataset()
        texts = test_dataset["text"]
        labels = test_dataset["label"]
        indices = np.random.choice(len(texts), min(num_samples, len(texts)), replace=False)
        
        # 根据配置决定推理模式
        use_vote = self.config.training.vote_count > 1
        
        self.logger.info(f"\n使用模型: {'LoRA' if self.config.model.use_lora else 'Base Model'}")
        self.logger.info(f"数据集: {self.config.data.dataset_name}")
        self.logger.info(f"推理模式: {'投票推理' if use_vote else '单一推理'}")
        if use_vote:
            self.logger.info(f"投票次数: {self.config.training.vote_count}")
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
                formatted_prompt = self._format_prompt(prompt)
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt", max_length=2048, truncation=True).to(self.model.device)
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