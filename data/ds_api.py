import json
import time
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class DeepSeekCoTGenerator:
    def __init__(self):
        # 5个API配置
        self.api_configs = [
            {
                "api_key": "sk-jguwpodlneddocxgvqiimjjrbhbhvglodfyagypskvvcktie",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            },
            {
                "api_key": "sk-xrqkuqemtzbspyofxlybphoiqdmssrwrgtpvdeyvureaxcbq",  # 请替换为您的其他API key
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            },
            {
                "api_key": "sk-ziyvhkysmskpeiywfrkmrhnapzxikqsongenuudigosompsb",  # 请替换为您的其他API key
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            },
            {
                "api_key": "sk-rgtyodzwijqqzjyuxtbxmmlhurqctnpuvagxcdjmoszfpfwm",  # 请替换为您的其他API key
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            },
            {
                "api_key": "sk-wrpfhdjyrfxryxcsxsjdenwcdklehxswujqwnqoaykyogqrr",  # 请替换为您的其他API key
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            }
        ]
        
        # 创建5个客户端
        self.clients = []
        for config in self.api_configs:
            client = OpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"]
            )
            self.clients.append(client)
        
        # API轮换索引
        self.api_index = 0
        self.api_lock = threading.Lock()
        
        # 类别映射
        self.category_mapping = {
            "C01": "Bacterial Infections and Mycoses",
            "C02": "Virus Diseases", 
            "C03": "Parasitic Diseases",
            "C04": "Neoplasms",
            "C05": "Musculoskeletal Diseases",
            "C06": "Digestive System Diseases",
            "C07": "Stomatognathic Diseases",
            "C08": "Respiratory Tract Diseases",
            "C09": "Otorhinolaryngologic Diseases",
            "C10": "Nervous System Diseases",
            "C11": "Eye Diseases",
            "C12": "Urologic and Male Genital Diseases",
            "C13": "Female Genital Diseases and Pregnancy Complications",
            "C14": "Cardiovascular Diseases",
            "C15": "Hemic and Lymphatic Diseases",
            "C16": "Neonatal Diseases and Abnormalities",
            "C17": "Skin and Connective Tissue Diseases",
            "C18": "Nutritional and Metabolic Diseases",
            "C19": "Endocrine Diseases",
            "C20": "Immunologic Diseases",
            "C21": "Disorders of Environmental Origin",
            "C22": "Animal Diseases",
            "C23": "Pathological Conditions, Signs and Symptoms"
        }
    
    def get_next_api(self):
        """轮换获取下一个API"""
        with self.api_lock:
            api_config = self.api_configs[self.api_index]
            client = self.clients[self.api_index]
            self.api_index = (self.api_index + 1) % len(self.api_configs)
            return api_config, client
    
    def generate_cot_response(self, text, true_label, max_retries=3):
        """生成CoT推理过程，基于真实标签补全推理"""
        
        system_prompt = (
            "You are a medical text classification expert. Given a medical text and its correct classification, "
            "please provide a detailed step-by-step reasoning process that leads to the correct classification.\n\n"
            "Available categories:\n"
        )
        
        # 添加类别映射
        for code, name in self.category_mapping.items():
            system_prompt += f"{code} - {name}\n"
        
        system_prompt += (
            "\nPlease follow this format:\n"
            "1. Identify key medical terms and concepts in the text\n"
            "2. Analyze the body systems involved\n"
            "3. Consider the disease type and pathology\n"
            "4. Explain why this text belongs to the given category\n"
            "5. Provide final classification with confidence\n\n"
            "End your response with: 'Final classification: [CATEGORY_CODE]'\n\n"
            "Important: The final classification must match the given correct label."
        )
        
        user_prompt = (
            f"Please provide a detailed reasoning process for this medical text classification:\n\n"
            f"Text: {text}\n"
            f"Correct classification: {true_label}\n\n"
            f"Please explain step by step why this text belongs to category {true_label}."
        )
        
        for attempt in range(max_retries):
            try:
                api_config, client = self.get_next_api()
                print(f"使用API {self.api_index}: {api_config['model']}")
                
                response = client.chat.completions.create(
                    model=api_config["model"],
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                    stream=False,
                    timeout=30
                )
                
                cot_response = response.choices[0].message.content
                
                # 验证最终分类是否正确
                if true_label in cot_response:
                    return cot_response
                else:
                    # 如果最终分类不正确，强制修正
                    corrected_response = cot_response.rstrip() + f"\n\nFinal classification: {true_label}"
                    print(f"修正最终分类为: {true_label}")
                    return corrected_response
                
            except Exception as e:
                print(f"API调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print("达到最大重试次数，跳过此样本")
                    return None
    
    def process_batch_parallel(self, batch_data):
        """并行处理一批数据"""
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:  # 使用5个线程
            # 提交所有任务
            future_to_item = {
                executor.submit(self.generate_cot_response, item['input'], item['output']): item 
                for item in batch_data
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_item), total=len(batch_data), desc="并行处理"):
                item = future_to_item[future]
                try:
                    cot_response = future.result()
                    if cot_response:
                        cot_item = {
                            "instruction": (
                                "你是一个医疗文本分类专家。请仔细分析给定的医疗文本，"
                                "提供详细的推理过程，然后给出最终的分类结果。"
                            ),
                            "input": item['input'],
                            "output": cot_response
                        }
                        results.append(cot_item)
                    else:
                        print(f"跳过样本: API调用失败")
                except Exception as e:
                    print(f"处理样本时出错: {e}")
        
        return results
    
    def test_connection(self):
        """测试API连接"""
        print("测试API连接...")
        try:
            api_config, client = self.get_next_api()
            response = client.chat.completions.create(
                model=api_config["model"],
                messages=[{'role': 'user', 'content': 'Hello'}],
                max_tokens=10,
                timeout=10
            )
            print("API连接成功！")
            return True
        except Exception as e:
            print(f"API连接失败: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_file):
        """加载断点数据"""
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_checkpoint(self, data, checkpoint_file):
        """保存断点数据"""
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_next_start_index(self, original_data, processed_data):
        """自动计算下一个开始索引"""
        if not processed_data:
            return 0
        
        # 获取已处理的文本列表
        processed_texts = {item['input'] for item in processed_data}
        
        # 找到第一个未处理的样本索引
        for i, item in enumerate(original_data):
            if item['input'] not in processed_texts:
                return i
        
        # 如果所有样本都已处理，返回总长度
        return len(original_data)
    
    def process_dataset(self, input_file, output_file, checkpoint_file, batch_size=10, max_samples=None):
        """处理数据集，支持并行处理"""
        
        # 测试连接
        if not self.test_connection():
            print("API连接失败，请检查网络和API配置")
            return []
        
        # 读取原始数据
        print(f"读取原始数据: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # 加载已处理的数据
        processed_data = self.load_checkpoint(checkpoint_file)
        print(f"已加载断点数据: {len(processed_data)} 个样本")
        
        # 自动计算开始索引
        start_index = self.get_next_start_index(original_data, processed_data)
        print(f"自动检测到断点位置: {start_index}")
        
        # 检查是否已完成所有数据
        if start_index >= len(original_data):
            print("所有数据已处理完成！")
            return processed_data
        
        # 确定处理范围
        if max_samples:
            end_index = min(start_index + max_samples, len(original_data))
        else:
            end_index = len(original_data)
        
        print(f"处理范围: {start_index} - {end_index} (共 {end_index - start_index} 个样本)")
        print(f"总进度: {len(processed_data)}/{len(original_data)} ({len(processed_data)/len(original_data)*100:.1f}%)")
        print(f"使用 {len(self.api_configs)} 个API并行处理，批次大小: {batch_size}")
        print("模式: 基于真实标签生成CoT推理过程")
        
        # 分批处理
        for batch_start in range(start_index, end_index, batch_size):
            batch_end = min(batch_start + batch_size, end_index)
            batch_data = original_data[batch_start:batch_end]
            
            print(f"\n处理批次: {batch_start}-{batch_end}")
            
            # 并行处理当前批次
            batch_results = self.process_batch_parallel(batch_data)
            
            # 添加到已处理数据
            processed_data.extend(batch_results)
            
            # 保存断点
            self.save_checkpoint(processed_data, checkpoint_file)
            print(f"已保存断点: {len(processed_data)} 个样本")
            
            # 批次间延迟
            time.sleep(1)
        
        # 最终保存
        self.save_checkpoint(processed_data, output_file)
        print(f"完成！共生成 {len(processed_data)} 个CoT样本")
        print(f"结果保存到: {output_file}")
        print(f"最终进度: {len(processed_data)}/{len(original_data)} ({len(processed_data)/len(original_data)*100:.1f}%)")
        
        return processed_data

def main():
    # 配置
    input_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Train_alpaca_noCoT_updated.json"
    output_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Train_cot.json"
    checkpoint_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Train_cot_checkpoint.json"
    
    # 处理参数
    batch_size = 5  # 每批处理5个样本
    max_samples = 1000  # 每次处理50个样本，None表示处理全部剩余样本
    
    # 创建生成器
    generator = DeepSeekCoTGenerator()
    
    # 生成CoT数据
    cot_data = generator.process_dataset(
        input_file=input_file,
        output_file=output_file,
        checkpoint_file=checkpoint_file,
        batch_size=batch_size,
        max_samples=max_samples
    )
    
    # 显示示例
    if cot_data:
        print("\n最新生成的CoT数据示例:")
        print(json.dumps(cot_data[-1], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()