#!/usr/bin/env python3
"""
通用CoT数据生成器
支持不同数据集和类别映射的即插即用代码
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

class UniversalCoTGenerator:
    def __init__(self, dataset_config):
        """
        初始化通用CoT生成器
        
        Args:
            dataset_config: 数据集配置字典，包含：
                - name: 数据集名称
                - category_mapping: 类别映射字典
                - domain: 领域类型 ('medical', 'news', 'finance', etc.)
                - text_field: 文本字段名 ('text', 'input', 'content', etc.)
                - label_field: 标签字段名 ('label', 'output', 'category', etc.)
        """
        self.dataset_config = dataset_config
        self.dataset_name = dataset_config['name']
        self.category_mapping = dataset_config['category_mapping']
        self.domain = dataset_config['domain']
        self.text_field = dataset_config['text_field']
        self.label_field = dataset_config['label_field']
        
        # API配置
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
    
    def get_next_api(self):
        """轮换获取下一个可用的API"""
        with self.api_lock:
            if not self.clients:
                raise Exception("没有可用的API客户端")
            
            client, config = self.clients[self.api_index % len(self.clients)]
            self.api_index = (self.api_index + 1) % len(self.clients)
            return client, config
    
    def _build_system_prompt(self):
        """根据数据集领域构建系统提示词"""
        if self.domain == 'medical':
            prompt = (
                "You are a medical text classification expert. Given a medical text and its correct classification label, "
                "your task is to provide a rigorous, step-by-step reasoning process that logically leads to the correct classification. "
                "Your explanation should be detailed, professional, and reference key information from the text.\n\n"
                "Available categories:\n"
            )
        elif self.domain == 'news':
            prompt = (
                "You are a news text classification expert. Given a news text and its correct classification label, "
                "your task is to provide a detailed, step-by-step reasoning process that logically leads to the correct classification. "
                "Your explanation should analyze the content, context, and key topics mentioned in the text.\n\n"
                "Available categories:\n"
            )
        elif self.domain == 'finance':
            prompt = (
                "You are a financial text classification expert. Given a financial text and its correct classification label, "
                "your task is to provide a comprehensive, step-by-step reasoning process that logically leads to the correct classification. "
                "Your explanation should analyze financial terms, market indicators, and economic concepts mentioned in the text.\n\n"
                "Available categories:\n"
            )
        else:
            prompt = (
                "You are a text classification expert. Given a text and its correct classification label, "
                "your task is to provide a detailed, step-by-step reasoning process that logically leads to the correct classification. "
                "Your explanation should be thorough and reference key information from the text.\n\n"
                "Available categories:\n"
            )
        
        for code, name in self.category_mapping.items():
            prompt += f"{code} - {name}\n"
        
        prompt += (
            "\nPlease strictly follow this format:\n"
            "Step 1: Identify and list key terms, concepts, or topics mentioned in the text.\n"
            "Step 2: Analyze the main subject matter and context of the text.\n"
            "Step 3: Consider the type of content and its characteristics.\n"
            "Step 4: Explain in detail why the text should be classified under the given category, referencing evidence from the text.\n"
            "Step 5: State your final classification with a confidence statement.\n\n"
            "End your response with: 'Final classification: [CATEGORY_CODE]'\n"
            "Important: The final classification code must exactly match the provided correct label."
        )
        
        return prompt
    
    def _build_user_prompt(self, text, true_label):
        """构建用户提示词"""
        return (
            "Please provide a detailed, step-by-step reasoning process for the following text classification task:\n\n"
            f"Text: {text}\n"
            f"Correct classification label: {true_label}\n\n"
            f"Explain, step by step, why this text belongs to category {true_label}, following the required format."
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
        
        # 过滤有效数据
        valid_batch_data = []
        for i, item in enumerate(batch_data):
            if (item.get(self.text_field) and item.get(self.text_field).strip() and 
                item.get(self.label_field) and item.get(self.label_field).strip()):
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
                    item[self.text_field], 
                    item[self.label_field], 
                    sample_id
                ): (item, sample_id) 
                for item, sample_id in valid_batch_data
            }
            
            # 收集结果
            with tqdm(total=len(future_to_item), desc=f"批次 {batch_id}", leave=False) as pbar:
                for future in as_completed(future_to_item):
                    item, sample_id = future_to_item[future]
                    try:
                        cot_response = future.result(timeout=60)
                        if cot_response:
                            cot_item = {
                                "instruction": (
                                    f"You are a {self.domain} text classification expert. Carefully analyze the given text, "
                                    "provide a detailed step-by-step reasoning process, and then give the final classification result."
                                ),
                                "input": item[self.text_field],
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
        """计算下一个开始索引 - 按顺序检查"""
        if not processed_data:
            return 0
        
        # 创建已处理输入的集合，用于快速查找
        processed_inputs = {item['input'] for item in processed_data if item.get('input')}
        
        # 按顺序检查原始数据，找到第一个未处理的索引
        for i, item in enumerate(original_data):
            if item.get(self.text_field) and item[self.text_field] not in processed_inputs:
                return i
        
        return len(original_data)
    
    def get_processed_count(self, original_data, processed_data):
        """获取已处理的数量"""
        if not processed_data:
            return 0
        
        processed_inputs = {item['input'] for item in processed_data if item.get('input')}
        count = 0
        
        for item in original_data:
            if item.get(self.text_field) and item[self.text_field] in processed_inputs:
                count += 1
        
        return count
    
    def process_dataset(self, input_file, output_file, checkpoint_file, batch_size=5, max_samples=None):
        """处理数据集 - 支持断点续传和定期保存"""
        
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
        initial_processed_count = len(processed_data)
        
        # 计算开始索引
        start_index = self.get_next_start_index(original_data, processed_data)
        processed_count = self.get_processed_count(original_data, processed_data)
        
        logger.info(f"断点位置: {start_index}")
        logger.info(f"已处理数量: {processed_count}/{len(original_data)} ({processed_count/len(original_data)*100:.1f}%)")
        
        if start_index >= len(original_data):
            logger.info("所有数据已处理完成！")
            self.save_final_result(processed_data, output_file)
            return processed_data
        
        # 确定处理范围
        if max_samples:
            end_index = min(start_index + max_samples, len(original_data))
        else:
            end_index = len(original_data)
        
        total_to_process = end_index - start_index
        logger.info(f"处理范围: {start_index} - {end_index} (共 {total_to_process} 个样本)")
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
                    
                    # 每10个请求保存到最终文件
                    new_processed_count = len(processed_data) - initial_processed_count
                    if new_processed_count % 10 == 0 and new_processed_count > 0:
                        logger.info(f"已新增处理 {new_processed_count} 个样本，保存到最终文件")
                        self.save_final_result(processed_data, output_file)
                    
                    # 更新进度条
                    pbar.update(len(batch_data))
                    
                    # 批次间短暂延迟
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            logger.info("用户中断，保存当前进度...")
            self.save_checkpoint(processed_data, checkpoint_file)
            self.save_final_result(processed_data, output_file)
            return processed_data
        except Exception as e:
            logger.error(f"处理过程中出错: {e}")
            self.save_checkpoint(processed_data, checkpoint_file)
            self.save_final_result(processed_data, output_file)
            return processed_data
        
        # 最终保存
        self.save_final_result(processed_data, output_file)
        final_processed_count = self.get_processed_count(original_data, processed_data)
        logger.info(f"处理完成！共生成 {len(processed_data)} 个CoT样本")
        logger.info(f"结果保存到: {output_file}")
        logger.info(f"最终进度: {final_processed_count}/{len(original_data)} ({final_processed_count/len(original_data)*100:.1f}%)")
        
        return processed_data
    
    def save_final_result(self, data, output_file):
        """保存最终结果到输出文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"保存最终结果: {len(data)} 个样本到 {output_file}")
        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")

# 数据集配置
DATASET_CONFIGS = {
    'ohsumed': {
        'name': 'Ohsumed',
        'domain': 'medical',
        'text_field': 'text',
        'label_field': 'output',
        'category_mapping': {
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
    },
    'R52': {
        'name': 'R52',
        'domain': 'news',
        'text_field': 'input',
        'label_field': 'output',
        'category_mapping': {
            "C01": "cocoa",
            "C02": "earn",
            "C03": "acq",
            "C04": "copper",
            "C05": "housing",
            "C06": "money-supply",
            "C07": "coffee",
            "C08": "sugar",
            "C09": "trade",
            "C10": "reserves",
            "C11": "ship",
            "C12": "cotton",
            "C13": "grain",
            "C14": "crude",
            "C15": "nat-gas",
            "C16": "cpi",
            "C17": "interest",
            "C18": "money-fx",
            "C19": "alum",
            "C20": "tin",
            "C21": "gold",
            "C22": "strategic-metal",
            "C23": "retail",
            "C24": "ipi",
            "C25": "iron-steel",
            "C26": "rubber",
            "C27": "heat",
            "C28": "jobs",
            "C29": "lei",
            "C30": "bop",
            "C31": "gnp",
            "C32": "zinc",
            "C33": "veg-oil",
            "C34": "orange",
            "C35": "carcass",
            "C36": "pet-chem",
            "C37": "gas",
            "C38": "wpi",
            "C39": "livestock",
            "C40": "lumber",
            "C41": "instal-debt",
            "C42": "meal-feed",
            "C43": "lead",
            "C44": "potato",
            "C45": "nickel",
            "C46": "cpu",
            "C47": "fuel",
            "C48": "jet",
            "C49": "income",
            "C50": "platinum",
            "C51": "dlr",
            "C52": "tea"
        }
    }
}

def main():
    """主函数"""
    print("=" * 60)
    print("通用CoT数据生成器")
    print("=" * 60)
    
    # 选择数据集
    print("可用数据集:")
    for i, dataset_name in enumerate(DATASET_CONFIGS.keys(), 1):
        print(f"{i}. {dataset_name}")
    
    choice = input("\n请选择数据集 (输入数字): ").strip()
    try:
        dataset_name = list(DATASET_CONFIGS.keys())[int(choice) - 1]
    except (ValueError, IndexError):
        print("无效选择，使用默认数据集: R52")
        dataset_name = 'R52'
    
    config = DATASET_CONFIGS[dataset_name]
    print(f"✅ 选择数据集: {config['name']}")
    print(f"✅ 领域: {config['domain']}")
    print(f"✅ 文本字段: {config['text_field']}")
    print(f"✅ 标签字段: {config['label_field']}")
    
    # 文件路径
    base_dir = f"/mnt/data1/TC/TextClassDemo/data/{dataset_name}"
    input_file = f"{base_dir}/{dataset_name}_Train.json"
    output_file = f"{base_dir}/{dataset_name}_Train_cot.json"
    checkpoint_file = f"{base_dir}/{dataset_name}_Train_cot_checkpoint.json"
    
    # 处理参数
    batch_size = 5
    max_samples = None  # 处理全部数据
    
    print(f"\n文件路径:")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"断点文件: {checkpoint_file}")
    print(f"批次大小: {batch_size}")
    print(f"最大样本数: {max_samples if max_samples else '全部'}")
    
    # 确认开始处理
    confirm = input("\n确认开始处理? (y/N): ").strip().lower()
    if confirm != 'y':
        print("取消处理")
        return
    
    # 创建生成器
    logger.info("初始化生成器...")
    generator = UniversalCoTGenerator(config)
    
    # 处理数据
    logger.info("开始处理数据...")
    try:
        cot_data = generator.process_dataset(
            input_file=input_file,
            output_file=output_file,
            checkpoint_file=checkpoint_file,
            batch_size=batch_size,
            max_samples=max_samples
        )
        
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        
        if cot_data:
            print(f"✅ 总共生成: {len(cot_data)} 个CoT样本")
            print(f"✅ 结果保存到: {output_file}")
            print(f"✅ 断点保存到: {checkpoint_file}")
            
            # 显示最新样本示例
            print("\n最新生成的CoT样本示例:")
            print("-" * 50)
            latest_sample = cot_data[-1]
            print(f"Instruction: {latest_sample['instruction'][:100]}...")
            print(f"Input: {latest_sample['input'][:100]}...")
            print(f"Output: {latest_sample['output'][:200]}...")
            print("-" * 50)
        else:
            print("❌ 没有生成任何数据")
            
    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 