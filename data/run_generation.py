#!/usr/bin/env python3
"""
简化的CoT数据生成脚本
"""

import json
import os
import logging
from ds_api_v2 import DeepSeekCoTGenerator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    print("=" * 60)
    print("CoT数据生成脚本")
    print("=" * 60)
    
    # 文件路径配置
    input_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/ohsumed_Train.json"
    output_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/ohsumed_Train_cot.json"
    checkpoint_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/ohsumed_Train_cot_checkpoint.json"
    
    # 检查输入文件
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 读取原始数据
    logger.info(f"读取原始数据: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        logger.info(f"原始数据: {len(original_data)} 个样本")
    except Exception as e:
        logger.error(f"读取原始数据失败: {e}")
        return
    
    # 检查数据格式并转换
    if original_data and isinstance(original_data[0], dict):
        sample_keys = list(original_data[0].keys())
        logger.info(f"数据格式: {sample_keys}")
        
        # 确定文本字段和标签字段
        text_field = None
        label_field = None
        
        # 常见的文本字段名
        text_candidates = ['text', 'input', 'content', 'sentence', 'abstract']
        for field in text_candidates:
            if field in sample_keys:
                text_field = field
                break
        
        # 常见的标签字段名
        label_candidates = ['label', 'output', 'category', 'class', 'target']
        for field in label_candidates:
            if field in sample_keys:
                label_field = field
                break
        
        if not text_field:
            logger.error(f"未找到文本字段，可用字段: {sample_keys}")
            return
        
        if not label_field:
            logger.error(f"未找到标签字段，可用字段: {sample_keys}")
            return
        
        logger.info(f"使用文本字段: {text_field}")
        logger.info(f"使用标签字段: {label_field}")
        
        # 转换数据格式为统一格式
        converted_data = []
        for item in original_data:
            converted_item = {
                "text": item[text_field],
                "output": item[label_field]
            }
            converted_data.append(converted_item)
        
        # 保存转换后的数据到临时文件
        temp_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/temp_converted_data.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据转换完成，保存到临时文件: {temp_file}")
        input_file = temp_file
    
    # 处理参数
    print("\n" + "=" * 60)
    print("处理参数设置")
    print("=" * 60)
    
    batch_size_input = input("批次大小 (默认5): ").strip()
    batch_size = int(batch_size_input) if batch_size_input else 5
    
    max_samples_input = input("最大处理样本数 (默认全部，输入数字限制): ").strip()
    max_samples = int(max_samples_input) if max_samples_input else None
    
    print(f"✅ 批次大小: {batch_size}")
    print(f"✅ 最大样本数: {max_samples if max_samples else '全部'}")
    
    # 确认开始处理
    confirm = input("\n确认开始处理? (y/N): ").strip().lower()
    if confirm != 'y':
        print("取消处理")
        return
    
    # 创建生成器
    logger.info("初始化生成器...")
    generator = DeepSeekCoTGenerator()
    
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