#!/usr/bin/env python3
"""
TREC数据集CoT数据生成脚本
"""

import json
import time
import os
import threading
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TRECCoTGenerator:
    def __init__(self):
        # TREC类别映射
        self.category_mapping = {
            "C01": "entity",
            "C02": "human", 
            "C03": "description",
            "C04": "numeric",
            "C05": "location",
            "C06": "abbreviation"
        }
        
        # API配置
        self.api_configs = [
            {
                "name": "API_1",
                "api_key": "sk-jguwpodlneddocxgvqiimjjrbhbhvglodfyagypskvvcktie",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
            },
            {
                "name": "API_2",
                "api_key": "sk-xrqkuqemtzbspyofxlybphoiqdmssrwrgtpvdeyvureaxcbq",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            },
            {
                "name": "API_3",
                "api_key": "sk-bsfexwpjznbvouitlwqafqybxeqvfqvakjrucablodwqltag",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            },
            {
                "name": "API_4",
                "api_key": "sk-rgtyodzwijqqzjyuxtbxmmlhurqctnpuvagxcdjmoszfpfwm",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            },
            {
                "name": "API_5",
                "api_key": "sk-yyiefuouvvfyswkpzmuqbddwfhymazsyriuhpfdtauwiijcf",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            }
        ]
        
        # 创建客户端
        self.clients = []
        for config in self.api_configs:
            try:
                client = OpenAI(
                    api_key=config["api_key"],
                    base_url=config["base_url"]
                )
                self.clients.append((client, config))
                logger.info(f"初始化 {config['name']} 成功")
            except Exception as e:
                logger.error(f"初始化 {config['name']} 失败: {e}")
        
        # API轮换相关
        self.api_index = 0
        self.api_lock = threading.Lock()
    
    def get_next_api(self):
        """轮换获取下一个可用的API"""
        with self.api_lock:
            if not self.clients:
                raise Exception("没有可用的API客户端")
            
            client, config = self.clients[self.api_index % len(self.clients)]
            self.api_index = (self.api_index + 1) % len(self.clients)
            return client, config
    
    def _build_system_prompt(self):
        """构建系统提示词 - 针对问答分类任务"""
        prompt = (
            "You are a question classification expert. Given a question and its correct classification label, "
            "your task is to provide a detailed, step-by-step reasoning process that logically leads to the correct classification. "
            "Your explanation should analyze the question type, expected answer format, and key information being sought.\n\n"
            "Available categories:\n"
        )
        
        for code, name in self.category_mapping.items():
            prompt += f"{code} - {name}\n"
        
        prompt += (
            "\nCategory descriptions:\n"
            "- entity: Questions asking about specific objects, concepts, or things\n"
            "- human: Questions asking about people, individuals, or human-related information\n"
            "- description: Questions asking for explanations, definitions, or detailed information\n"
            "- numeric: Questions asking for numbers, dates, quantities, or measurements\n"
            "- location: Questions asking about places, geographical locations, or spatial information\n"
            "- abbreviation: Questions asking for the meaning of abbreviations or acronyms\n\n"
            "Please strictly follow this format:\n"
            "Step 1: Identify the question type and what information is being requested.\n"
            "Step 2: Analyze the expected answer format and characteristics.\n"
            "Step 3: Consider the key words and phrases that indicate the question category.\n"
            "Step 4: Explain in detail why the question should be classified under the given category.\n"
            "Step 5: State your final classification with a confidence statement.\n\n"
            "End your response with: 'Final classification: [CATEGORY_CODE]'\n"
            "Important: The final classification code must exactly match the provided correct label."
        )
        
        return prompt
    
    def _build_user_prompt(self, text, true_label):
        """构建用户提示词"""
        return (
            "Please provide a detailed, step-by-step reasoning process for the following question classification task:\n\n"
            f"Question: {text}\n"
            f"Correct classification label: {true_label}\n\n"
            f"Explain, step by step, why this question belongs to category {true_label}, following the required format."
        )
    
    def generate_cot_response(self, text, true_label, sample_id=None, max_retries=3):
        """生成CoT推理过程，带有重试机制"""
        
        # 数据验证
        if not text or not text.strip():
            logger.warning(f"样本 {sample_id} 的文本为空，跳过")
            return None
        
        if not true_label or not true_label.strip():
            logger.warning(f"样本 {sample_id} 的标签为空，跳过")
            return None
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(text, true_label)
        
        for attempt in range(max_retries):
            try:
                client, api_config = self.get_next_api()
                logger.debug(f"样本 {sample_id} 使用 {api_config['name']} (尝试 {attempt + 1})")
                
                response = client.chat.completions.create(
                    model=api_config["model"],
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                    timeout=45
                )
                
                cot_response = response.choices[0].message.content
                
                # 验证响应质量
                if self._validate_response(cot_response, true_label):
                    logger.debug(f"样本 {sample_id} 处理成功")
                    return cot_response
                else:
                    # 强制修正
                    corrected_response = cot_response.rstrip() + f"\n\nFinal classification: {true_label}"
                    logger.info(f"样本 {sample_id} 分类修正为: {true_label}")
                    return corrected_response
                
            except Exception as e:
                wait_time = min(2 ** attempt, 16)  # 指数退避，最大16秒
                logger.warning(f"样本 {sample_id} API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"样本 {sample_id} 达到最大重试次数，跳过")
                    return None
    
    def _validate_response(self, response, true_label):
        """验证响应质量"""
        if not response:
            return False
        
        # 检查是否包含正确的标签
        return true_label in response
    
    def process_batch_parallel(self, batch_data, batch_id):
        """并行处理一批数据"""
        results = []
        failed_count = 0
        
        # 过滤有效数据 - TREC使用text字段
        valid_batch_data = []
        for i, item in enumerate(batch_data):
            if isinstance(item, dict) and 'text' in item and 'label' in item:
                valid_batch_data.append((i, item))
            else:
                logger.warning(f"批次 {batch_id} 样本 {i} 数据格式无效，跳过")
                failed_count += 1
        
        if not valid_batch_data:
            logger.warning(f"批次 {batch_id} 没有有效数据")
            return results, failed_count
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=min(len(valid_batch_data), 5)) as executor:
            future_to_item = {}
            
            for idx, (original_idx, item) in enumerate(valid_batch_data):
                sample_id = f"batch_{batch_id}_sample_{original_idx}"
                future = executor.submit(
                    self.generate_cot_response,
                    item['text'],
                    item['label'],
                    sample_id
                )
                future_to_item[future] = (original_idx, item, sample_id)
            
            # 收集结果
            for future in tqdm(as_completed(future_to_item), 
                             total=len(future_to_item), 
                             desc=f"处理批次 {batch_id}"):
                original_idx, item, sample_id = future_to_item[future]
                
                try:
                    cot_response = future.result()
                    if cot_response:
                        # 构建结果数据
                        result_item = {
                            'text': item['text'],
                            'label': item['label'],
                            'cot': cot_response
                        }
                        results.append((original_idx, result_item))
                        logger.debug(f"样本 {sample_id} 处理完成")
                    else:
                        logger.warning(f"样本 {sample_id} 处理失败")
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"样本 {sample_id} 处理异常: {e}")
                    failed_count += 1
        
        # 按原始索引排序
        results.sort(key=lambda x: x[0])
        return [item for _, item in results], failed_count
    
    def test_all_apis(self):
        """测试所有API是否可用"""
        test_text = "what is the capital of france?"
        test_label = "location"
        
        logger.info("开始测试所有API...")
        
        for i, (client, config) in enumerate(self.clients):
            try:
                logger.info(f"测试 {config['name']}...")
                
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt(test_text, test_label)
                
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    timeout=30
                )
                
                result = response.choices[0].message.content
                logger.info(f"{config['name']} 测试成功: {result[:100]}...")
                
            except Exception as e:
                logger.error(f"{config['name']} 测试失败: {e}")
    
    def load_checkpoint(self, checkpoint_file):
        """加载检查点数据"""
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"从检查点加载了 {len(data)} 条数据")
                return data
            except Exception as e:
                logger.error(f"加载检查点失败: {e}")
                return []
        return []
    
    def save_checkpoint(self, data, checkpoint_file):
        """保存检查点数据"""
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"检查点已保存: {len(data)} 条数据")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def get_next_start_index(self, original_data, processed_data):
        """获取下一个开始处理的索引"""
        if not processed_data:
            return 0
        
        processed_texts = {item['text'] for item in processed_data}
        
        for i, item in enumerate(original_data):
            if item.get('text') not in processed_texts:
                return i
        
        return len(original_data)
    
    def get_processed_count(self, original_data, processed_data):
        """获取已处理的数据数量"""
        if not processed_data:
            return 0
        
        processed_texts = {item['text'] for item in processed_data}
        count = 0
        
        for item in original_data:
            if item.get('text') in processed_texts:
                count += 1
        
        return count
    
    def process_dataset(self, input_file, output_file, checkpoint_file, batch_size=5, max_samples=None):
        """处理整个数据集"""
        logger.info(f"开始处理数据集: {input_file}")
        
        # 加载原始数据
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            logger.info(f"加载原始数据: {len(original_data)} 条")
        except Exception as e:
            logger.error(f"加载原始数据失败: {e}")
            return
        
        # 限制样本数量
        if max_samples and max_samples < len(original_data):
            original_data = original_data[:max_samples]
            logger.info(f"限制样本数量为: {max_samples}")
        
        # 加载检查点
        processed_data = self.load_checkpoint(checkpoint_file)
        
        # 获取开始索引
        start_index = self.get_next_start_index(original_data, processed_data)
        logger.info(f"从索引 {start_index} 开始处理")
        
        if start_index >= len(original_data):
            logger.info("所有数据已处理完成")
            self.save_final_result(processed_data, output_file)
            return
        
        # 分批处理
        total_batches = (len(original_data) - start_index + batch_size - 1) // batch_size
        current_batch = 0
        
        for i in range(start_index, len(original_data), batch_size):
            current_batch += 1
            batch_end = min(i + batch_size, len(original_data))
            batch_data = original_data[i:batch_end]
            
            logger.info(f"处理批次 {current_batch}/{total_batches} (索引 {i}-{batch_end-1})")
            
            try:
                batch_results, failed_count = self.process_batch_parallel(batch_data, current_batch)
                
                # 添加到已处理数据
                processed_data.extend(batch_results)
                
                # 保存检查点
                self.save_checkpoint(processed_data, checkpoint_file)
                
                logger.info(f"批次 {current_batch} 完成: 成功 {len(batch_results)} 条, 失败 {failed_count} 条")
                
                # 进度统计
                total_processed = self.get_processed_count(original_data, processed_data)
                progress = (total_processed / len(original_data)) * 100
                logger.info(f"总进度: {total_processed}/{len(original_data)} ({progress:.1f}%)")
                
            except Exception as e:
                logger.error(f"批次 {current_batch} 处理失败: {e}")
                # 保存当前进度
                self.save_checkpoint(processed_data, checkpoint_file)
                continue
        
        # 保存最终结果
        self.save_final_result(processed_data, output_file)
        logger.info("数据集处理完成")
    
    def save_final_result(self, data, output_file):
        """保存最终结果"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"最终结果已保存: {output_file} ({len(data)} 条数据)")
        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")

def main():
    """主函数"""
    # 配置路径
    input_file = "TREC/TREC_Train_Cxx.json"
    output_file = "TREC/TREC_Train_Cot.json"
    checkpoint_file = "TREC/TREC_Train_Cot_checkpoint.json"
    
    # 创建生成器
    generator = TRECCoTGenerator()
    
    # 测试API
    generator.test_all_apis()
    
    # 处理数据集
    generator.process_dataset(
        input_file=input_file,
        output_file=output_file,
        checkpoint_file=checkpoint_file,
        batch_size=5,
        max_samples=None  # 设置为具体数字以限制样本数量
    )

if __name__ == "__main__":
    main() 