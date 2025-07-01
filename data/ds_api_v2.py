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

class DeepSeekCoTGenerator:
    def __init__(self):
        # 5个API配置
        self.api_configs = [
            {
                "name": "API_1",
                "api_key": "sk-jguwpodlneddocxgvqiimjjrbhbhvglodfyagypskvvcktie",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            },
            {
                "name": "API_2",
                "api_key": "sk-xrqkuqemtzbspyofxlybphoiqdmssrwrgtpvdeyvureaxcbq",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            },
            {
                "name": "API_3",
                "api_key": "sk-ziyvhkysmskpeiywfrkmrhnapzxikqsongenuudigosompsb",
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
                "api_key": "sk-wrpfhdjyrfxryxcsxsjdenwcdklehxswujqwnqoaykyogqrr",
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
        """轮换获取下一个可用的API"""
        with self.api_lock:
            if not self.clients:
                raise Exception("没有可用的API客户端")
            
            client, config = self.clients[self.api_index % len(self.clients)]
            self.api_index = (self.api_index + 1) % len(self.clients)
            return client, config
    
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
                    timeout=45  # 增加超时时间
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
    
    def _build_system_prompt(self):
        """构建系统提示词"""
        prompt = (
            "You are a medical text classification expert. Given a medical text and its correct classification label, "
            "your task is to provide a rigorous, step-by-step reasoning process that logically leads to the correct classification. "
            "Your explanation should be detailed, professional, and reference key information from the text.\n\n"
            "Available categories:\n"
        )
        
        for code, name in self.category_mapping.items():
            prompt += f"{code} - {name}\n"
        
        prompt += (
            "\nPlease strictly follow this format:\n"
            "Step 1: Identify and list key medical terms, symptoms, or concepts mentioned in the text.\n"
            "Step 2: Analyze which body systems or organs are involved, based on the identified terms.\n"
            "Step 3: Consider the type of disease or pathology (e.g., infection, neoplasm, metabolic disorder, etc.).\n"
            "Step 4: Explain in detail why the text should be classified under the given category, referencing evidence from the text.\n"
            "Step 5: State your final classification with a confidence statement.\n\n"
            "End your response with: 'Final classification: [CATEGORY_CODE]'\n"
            "Important: The final classification code must exactly match the provided correct label."
        )
        
        return prompt
    
    def _build_user_prompt(self, text, true_label):
        """构建用户提示词"""
        return (
            "Please provide a detailed, step-by-step reasoning process for the following medical text classification task:\n\n"
            f"Text: {text}\n"
            f"Correct classification label: {true_label}\n\n"
            f"Explain, step by step, why this text belongs to category {true_label}, following the required format."
        )
    
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
        
        # 过滤有效数据
        valid_batch_data = []
        for i, item in enumerate(batch_data):
            if item.get('input') and item.get('input').strip() and item.get('output') and item.get('output').strip():
                valid_batch_data.append((item, f"{batch_id}_{i}"))
            else:
                logger.warning(f"样本 {batch_id}_{i} 数据无效，跳过")
                failed_count += 1
        
        if not valid_batch_data:
            logger.warning(f"批次 {batch_id} 没有有效数据")
            return results
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=min(5, len(valid_batch_data))) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(
                    self.generate_cot_response, 
                    item['input'], 
                    item['output'], 
                    sample_id
                ): (item, sample_id) 
                for item, sample_id in valid_batch_data
            }
            
            # 收集结果
            with tqdm(total=len(future_to_item), desc=f"批次 {batch_id}", leave=False) as pbar:
                for future in as_completed(future_to_item):
                    item, sample_id = future_to_item[future]
                    try:
                        cot_response = future.result(timeout=60)  # 设置future超时
                        if cot_response:
                            cot_item = {
                                "instruction": (
                                    "You are a medical text classification expert. Carefully analyze the given medical text, "
                                    "provide a detailed step-by-step reasoning process, and then give the final classification result."
                                ),
                                "input": item['input'],
                                "output": cot_response
                            }
                            results.append(cot_item)
                        else:
                            failed_count += 1
                            logger.warning(f"样本 {sample_id} 生成失败")
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"样本 {sample_id} 处理异常: {e}")
                    finally:
                        pbar.update(1)
        
        logger.info(f"批次 {batch_id} 完成: 成功 {len(results)}, 失败 {failed_count}")
        return results
    
    def test_all_apis(self):
        """测试所有API连接"""
        logger.info("测试所有API连接...")
        working_apis = 0
        
        for i, (client, config) in enumerate(self.clients):
            try:
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=[{'role': 'user', 'content': 'Hello'}],
                    max_tokens=10,
                    timeout=15
                )
                logger.info(f"{config['name']} 连接正常")
                working_apis += 1
            except Exception as e:
                logger.error(f"{config['name']} 连接失败: {e}")
        
        logger.info(f"可用API数量: {working_apis}/{len(self.clients)}")
        return working_apis > 0
    
    def load_checkpoint(self, checkpoint_file):
        """加载断点数据"""
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"加载断点数据: {len(data)} 个样本")
                return data
            except Exception as e:
                logger.error(f"加载断点失败: {e}")
                return []
        return []
    
    def save_checkpoint(self, data, checkpoint_file):
        """保存断点数据"""
        try:
            # 备份现有文件
            if os.path.exists(checkpoint_file):
                backup_file = checkpoint_file + ".backup"
                os.rename(checkpoint_file, backup_file)
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 删除备份文件
            backup_file = checkpoint_file + ".backup"
            if os.path.exists(backup_file):
                os.remove(backup_file)
                
            logger.debug(f"保存断点: {len(data)} 个样本")
        except Exception as e:
            logger.error(f"保存断点失败: {e}")
            # 恢复备份
            backup_file = checkpoint_file + ".backup"
            if os.path.exists(backup_file):
                os.rename(backup_file, checkpoint_file)
    
    def get_next_start_index(self, original_data, processed_data):
        """计算下一个开始索引"""
        if not processed_data:
            return 0
        
        processed_inputs = {item['input'] for item in processed_data if item.get('input')}
        
        for i, item in enumerate(original_data):
            if item.get('input') and item['input'] not in processed_inputs:
                return i
        
        return len(original_data)
    
    def process_dataset(self, input_file, output_file, checkpoint_file, batch_size=5, max_samples=None):
        """处理数据集"""
        
        # 测试API连接
        if not self.test_all_apis():
            logger.error("没有可用的API，退出处理")
            return []
        
        # 读取原始数据
        logger.info(f"读取原始数据: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
        except Exception as e:
            logger.error(f"读取原始数据失败: {e}")
            return []
        
        # 加载已处理数据
        processed_data = self.load_checkpoint(checkpoint_file)
        
        # 计算开始索引
        start_index = self.get_next_start_index(original_data, processed_data)
        logger.info(f"断点位置: {start_index}")
        
        if start_index >= len(original_data):
            logger.info("所有数据已处理完成！")
            return processed_data
        
        # 确定处理范围
        if max_samples:
            end_index = min(start_index + max_samples, len(original_data))
        else:
            end_index = len(original_data)
        
        total_to_process = end_index - start_index
        logger.info(f"处理范围: {start_index} - {end_index} (共 {total_to_process} 个样本)")
        logger.info(f"总进度: {len(processed_data)}/{len(original_data)} ({len(processed_data)/len(original_data)*100:.1f}%)")
        logger.info(f"批次大小: {batch_size}, 可用API: {len(self.clients)}")
        
        # 分批处理
        try:
            with tqdm(total=total_to_process, desc="总体进度") as pbar:
                for batch_start in range(start_index, end_index, batch_size):
                    batch_end = min(batch_start + batch_size, end_index)
                    batch_data = original_data[batch_start:batch_end]
                    batch_id = f"batch_{batch_start}_{batch_end}"
                    
                    logger.info(f"处理 {batch_id}")
                    
                    # 处理当前批次
                    batch_results = self.process_batch_parallel(batch_data, batch_id)
                    
                    # 添加到已处理数据
                    processed_data.extend(batch_results)
                    
                    # 保存断点
                    self.save_checkpoint(processed_data, checkpoint_file)
                    
                    # 更新进度条
                    pbar.update(len(batch_data))
                    
                    # 批次间短暂延迟
                    time.sleep(1)
                    
                    # 在 process_dataset 方法的批次循环中添加
                    if len(processed_data) % 100 == 0:
                        logger.info(f"已处理 {len(processed_data)} 个样本，额外保存checkpoint")
                        self.save_checkpoint(processed_data, checkpoint_file)
                    
        except KeyboardInterrupt:
            logger.info("用户中断，保存当前进度...")
            self.save_checkpoint(processed_data, checkpoint_file)
            return processed_data
        except Exception as e:
            logger.error(f"处理过程中出错: {e}")
            self.save_checkpoint(processed_data, checkpoint_file)
            return processed_data
        
        # 最终保存
        try:
            self.save_checkpoint(processed_data, output_file)
            logger.info(f"处理完成！共生成 {len(processed_data)} 个CoT样本")
            logger.info(f"结果保存到: {output_file}")
            logger.info(f"最终进度: {len(processed_data)}/{len(original_data)} ({len(processed_data)/len(original_data)*100:.1f}%)")
        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")
        
        return processed_data

def main():
    # 配置
    input_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/ohsumed_Train_alpaca_noCoT_updated.json"
    output_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/ohsumed_Train_cot.json"
    checkpoint_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/ohsumed_Train_cot_checkpoint.json"
    
    # 处理参数
    batch_size = 10
    max_samples = 2000  # None表示处理全部
    
    # 创建生成器
    generator = DeepSeekCoTGenerator()
    
    # 处理数据
    cot_data = generator.process_dataset(
        input_file=input_file,
        output_file=output_file,
        checkpoint_file=checkpoint_file,
        batch_size=batch_size,
        max_samples=max_samples
    )
    
    # 显示示例
    if cot_data:
        logger.info("最新生成的CoT数据示例:")
        print(json.dumps(cot_data[-1], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main() 