import torch
from torch import nn
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class CoTClassifier(nn.Module):
    def __init__(
        self,
        model=None,
        tokenizer=None,
        device='cpu',
        num_classes=23,
        num_samples=5
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_classes = num_classes
        self.num_samples = num_samples
        
        # 启用梯度检查点以节省内存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # 设置CoT提示模板，强约束输出格式为数字
        self.prompt = (
            "You are a medical text classifier. Your task is to classify the given medical text into one of the 23 categories (1-23).\n\n"
            "Text: {text}\n\n"
            "Let's analyze this step by step:\n"
            "1. Carefully read and summarize the main medical topic or condition discussed in the text.\n"
            "2. Identify and list all specific medical terms, procedures, or treatments mentioned.\n"
            "3. Explain the primary focus of the text: is it about diagnosis, treatment, research, prevention, or something else?\n"
            "4. Based on the above analysis, reason step by step which category (1-23) the text belongs to, and explain your reasoning in detail.\n"
            "5. Finally, output ONLY the category number (1-23) on a new line.\n"
            "For example: 1\n\n"
            "Now, please classify the text and show your reasoning:\nCategory: "
        )
        
        # 获取数字1-23在词表中的位置
        self.num_locs = {}
        for i in range(self.num_classes):
            num_token = str(i+1)
            token_ids = tokenizer(num_token, add_special_tokens=False)['input_ids']
            print(f"{num_token} -> token_ids: {token_ids}")
            if len(token_ids) == 1:
                self.num_locs[i] = token_ids[0]
            else:
                print(f"Warning: {num_token} is not a single token!")
    
    def forward(self, batch):
        try:
            outputs = self.model.generate(
                **batch,
                max_new_tokens=5,  # 允许生成多个token，确保能生成数字
                temperature=0.1,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            
            # 打印调试信息
            if hasattr(outputs, 'logits') and len(outputs.logits) > 0:
                print(f"\nGenerated outputs shape: {outputs.logits[0].shape}")
                generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                print(f"Generated text: {generated_text}")
                print(f"outputs.sequences: {outputs.sequences}")
                print(f"decoded: {self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)}")
            
            # 检查tokenizer分词
            for i in range(self.num_classes):
                num_token = str(i+1)
                token_ids = self.tokenizer(num_token, add_special_tokens=False)['input_ids']
                print(f"{num_token} -> token_ids: {token_ids}")
            
            # 打印生成的token序列
            print("outputs.sequences:", outputs.sequences)
            print("decoded:", self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False))
            
            return outputs
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict(self, texts, batch_size=1):
        predictions = []
        probabilities = []

        for start_idx in tqdm.tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[start_idx:start_idx + batch_size]
            batch_inputs = self.preprocess(batch_texts)

            outputs = self.model.generate(
                **batch_inputs,
                max_new_tokens=128,  # 允许生成完整推理链+数字
                temperature=0.7,
                do_sample=True,
                num_beams=1,
                num_return_sequences=self.num_samples,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            all_token_vectors = []
            for sequence, logits_arr in zip(outputs.sequences, outputs.logits):
                # 直接取最后一个生成token的logits
                last_token_logits = logits_arr[-1, :]
                token_vector = [last_token_logits[self.num_locs[j]].item() for j in range(self.num_classes)]
                all_token_vectors.append(token_vector)

            # 对n条思维链的token vector做平均
            avg_token_vector = np.mean(all_token_vectors, axis=0)
            # 归一化
            avg_token_vector = torch.tensor(avg_token_vector)
            avg_probs = torch.softmax(avg_token_vector, dim=0).cpu().numpy()
            pred_class = int(np.argmax(avg_probs))
            predictions.append(pred_class)
            probabilities.append(avg_probs)

        return predictions, probabilities
    
    def preprocess(self, texts):
        messages = []
        for text in texts:
            prompt = self.prompt.format(text=text)
            messages.append([{"role": "user", "content": prompt}])
        
        # 应用聊天模板
        text_batch = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 转换为模型输入
        inputs = self.tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        ).to(self.device)
        
        # 打印输入信息
        print(f"\nInput shape: {inputs['input_ids'].shape}")
        print(f"Input text: {self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)}")
        
        return inputs

def main():
    # 设置路径
    base_model_path = "/root/autodl-tmp/miggs/llama3.1_3.18/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    adapter_path = "/root/autodl-tmp/miggs/outputs/ohsumed_classification/checkpoint-5216"
    data_path = "/root/autodl-tmp/miggs/ohsumed_converted/ohsumed_test_2000.json"
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载测试数据
    print("加载测试数据...")
    with open(data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 验证标签格式和分布
    print("\n验证标签分布:")
    label_counts = {}
    for item in test_data:
        category = item['category']
        if category not in label_counts:
            label_counts[category] = 0
        label_counts[category] += 1
    
    print("原始标签分布:")
    for cat, count in sorted(label_counts.items()):
        print(f"{cat}: {count}条")
    
    # 准备数据 - 使用与训练代码相同的方式处理标签
    texts = [item['input'] for item in test_data]
    true_labels = [int(item['category'][1:]) - 1 for item in test_data]  # C1-C23 -> 0-22
    
    print(f"\n转换后的标签范围: {min(true_labels)} - {max(true_labels)}")
    print(f"标签数量: {len(true_labels)}")
    
    # 配置4位量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # 加载模型和tokenizer
    print("\n加载基础模型和tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 加载adapter权重
    print("加载checkpoint-5216的adapter权重...")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side="left",
        trust_remote_code=True
    )
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 创建分类器
    classifier = CoTClassifier(model, tokenizer, device)
    
    # 进行预测
    print("\n开始预测...")
    predictions, probabilities = classifier.predict(texts)
    
    # 验证预测结果
    print("\n预测结果验证:")
    print(f"预测数量: {len(predictions)}")
    print(f"预测范围: {min(predictions)} - {max(predictions)}")
    
    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\n准确率: {accuracy:.4f}")
    
    # 打印详细的分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, predictions, 
                              target_names=[f'C{i+1}' for i in range(23)]))
    
    # 分析错误预测
    print("\n错误预测分析:")
    error_cases = []
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        if true != pred:
            error_cases.append({
                'text': texts[i],
                'true_label': f'C{true+1}',
                'predicted_label': f'C{pred+1}',
                'true_prob': probabilities[i][true],
                'pred_prob': probabilities[i][pred]
            })
    
    # 保存结果
    results = {
        'predictions': [int(x) for x in predictions],
        'true_labels': [int(x) for x in true_labels],
        'probabilities': [prob.tolist() if hasattr(prob, 'tolist') else prob for prob in probabilities],
        'error_cases': error_cases[:10]  # 保存前10个错误案例
    }
    
    with open('cot_classification_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到 cot_classification_results.json")
    print(f"错误案例数量: {len(error_cases)}")
    if error_cases:
        print("\n前5个错误案例:")
        for i, case in enumerate(error_cases[:5]):
            print(f"\n案例 {i+1}:")
            print(f"文本: {case['text'][:100]}...")  # 只显示前100个字符
            print(f"真实标签: {case['true_label']} (概率: {case['true_prob']:.4f})")
            print(f"预测标签: {case['predicted_label']} (概率: {case['pred_prob']:.4f})")

if __name__ == "__main__":
    main() 