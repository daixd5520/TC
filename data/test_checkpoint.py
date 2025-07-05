#!/usr/bin/env python3
"""
测试断点功能
"""

import json
import os
from ds_api_v2 import DeepSeekCoTGenerator

def test_checkpoint_logic():
    """测试断点逻辑"""
    print("测试断点逻辑...")
    
    # 模拟原始数据 - 使用text字段
    original_data = [
        {"text": "text1", "output": "C01"},
        {"text": "text2", "output": "C02"},
        {"text": "text3", "output": "C03"},
        {"text": "text4", "output": "C04"},
        {"text": "text5", "output": "C05"},
    ]
    
    # 模拟已处理数据（乱序）- 使用input字段
    processed_data = [
        {"input": "text3", "output": "CoT response for text3"},
        {"input": "text1", "output": "CoT response for text1"},
        {"input": "text5", "output": "CoT response for text5"},
    ]
    
    generator = DeepSeekCoTGenerator()
    
    # 测试获取下一个开始索引
    start_index = generator.get_next_start_index(original_data, processed_data)
    print(f"下一个开始索引: {start_index}")
    print(f"应该开始处理: {original_data[start_index]['text'] if start_index < len(original_data) else 'None'}")
    
    # 测试获取已处理数量
    processed_count = generator.get_processed_count(original_data, processed_data)
    print(f"已处理数量: {processed_count}/{len(original_data)}")
    
    # 显示未处理的样本
    processed_inputs = {item['input'] for item in processed_data}
    unprocessed = [item for item in original_data if item['text'] not in processed_inputs]
    print(f"未处理样本: {[item['text'] for item in unprocessed]}")

def check_real_data():
    """检查真实数据状态"""
    print("\n检查真实数据状态...")
    
    input_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/ohsumed_Train_alpaca_noCoT_updated.json"
    checkpoint_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/ohsumed_Train_cot_checkpoint.json"
    
    if not os.path.exists(input_file):
        print(f"原始数据文件不存在: {input_file}")
        return
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"原始数据: {len(original_data)} 个样本")
    
    # 读取断点数据
    processed_data = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        print(f"断点数据: {len(processed_data)} 个样本")
    
    generator = DeepSeekCoTGenerator()
    
    # 计算进度
    start_index = generator.get_next_start_index(original_data, processed_data)
    processed_count = generator.get_processed_count(original_data, processed_data)
    
    print(f"下一个开始索引: {start_index}")
    print(f"已处理数量: {processed_count}/{len(original_data)} ({processed_count/len(original_data)*100:.1f}%)")
    
    # 显示前几个未处理的样本
    if start_index < len(original_data):
        print("\n前3个未处理样本:")
        for i in range(start_index, min(start_index + 3, len(original_data))):
            item = original_data[i]
            print(f"  索引 {i}: {item['input'][:100]}... -> {item['output']}")

if __name__ == "__main__":
    test_checkpoint_logic()
    check_real_data() 