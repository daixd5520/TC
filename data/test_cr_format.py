#!/usr/bin/env python3
"""
测试CR数据格式和标签转换
"""

import json

def test_cr_format():
    """测试CR数据格式"""
    
    # 模拟CR数据格式
    test_data = [
        {
            "text": "player works and looks great if you can get the dvd ' s to play",
            "label": "positive"
        },
        {
            "text": "what a junk",
            "label": "negative"
        },
        {
            "text": "this product is amazing and better than expected",
            "label": "positive"
        }
    ]
    
    print("原始数据格式:")
    for i, item in enumerate(test_data):
        print(f"样本 {i+1}:")
        print(f"  text: {item['text']}")
        print(f"  label: {item['label']}")
    
    print("\n标签转换测试:")
    for i, item in enumerate(test_data):
        label_text = item['label']
        if label_text == "positive":
            label_cxx = "C01"
        elif label_text == "negative":
            label_cxx = "C02"
        else:
            label_cxx = "未知"
        print(f"样本 {i+1}: {label_text} -> {label_cxx}")
    
    # 类别映射
    category_mapping = {
        "C01": "positive",
        "C02": "negative"
    }
    
    print("\n类别映射验证:")
    for i, item in enumerate(test_data):
        label_text = item['label']
        if label_text == "positive":
            label_cxx = "C01"
        elif label_text == "negative":
            label_cxx = "C02"
        else:
            label_cxx = "未知"
        category_name = category_mapping.get(label_cxx, "未知")
        print(f"样本 {i+1}: {label_cxx} -> {category_name}")
    
    print("\nCoT输出格式示例:")
    for i, item in enumerate(test_data):
        label_text = item['label']
        if label_text == "positive":
            label_cxx = "C01"
        elif label_text == "negative":
            label_cxx = "C02"
        else:
            label_cxx = "未知"
        
        cot_item = {
            "instruction": (
                "You are a sentiment analysis expert. Carefully analyze the given customer review text, "
                "provide a detailed step-by-step reasoning process, and then give the final sentiment classification result."
            ),
            "input": item['text'],
            "output": f"Step 1: Analyze the text content...\nStep 2: Identify sentiment indicators...\nFinal classification: {label_cxx}"
        }
        
        print(f"样本 {i+1} CoT格式:")
        print(f"  instruction: {cot_item['instruction'][:50]}...")
        print(f"  input: {cot_item['input']}")
        print(f"  output: {cot_item['output'][:50]}...")
        print()

if __name__ == "__main__":
    test_cr_format() 