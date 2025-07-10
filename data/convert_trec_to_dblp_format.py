#!/usr/bin/env python3
"""
将TREC数据转换为dblp格式的脚本
"""

import json
import os

def convert_trec_to_dblp_format(input_file, output_file):
    """
    将TREC格式的数据转换为dblp格式
    
    Args:
        input_file: TREC格式的输入文件路径
        output_file: dblp格式的输出文件路径
    """
    
    # 定义instruction模板
    instruction_template = """You are a question type classification expert. Your task is to classify the given question into one of 6 categories.

Category mapping:
C01 - Questions about entities (e.g., objects, animals, substances)
C02 - Questions about people, professions or groups
C03 - Descriptive or definitional questions
C04 - Questions asking for numbers, amounts, dates or other numeric information
C05 - Questions about places or locations
C06 - Abbreviations or acronyms

Text: {text}

For example, if the question asks "What is caffeine?", it belongs to C03.
The output must be one of: C01, C02, C03, C04, C05, C06.
If you output anything else, you will be penalized."""
    
    # 读取TREC数据
    with open(input_file, 'r', encoding='utf-8') as f:
        trec_data = json.load(f)
    
    # 转换数据格式
    dblp_data = []
    
    for item in trec_data:
        text = item['text']
        label = item['label']
        
        # 创建dblp格式的数据项
        dblp_item = {
            "instruction": instruction_template.format(text=text),
            "input": text,
            "output": label
        }
        
        dblp_data.append(dblp_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dblp_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"转换了 {len(dblp_data)} 条数据")

def main():
    """主函数"""
    
    # 定义文件路径
    trec_dir = "TREC"
    
    # 转换训练集
    train_input = os.path.join(trec_dir, "TREC_Train_Cxx.json")
    train_output = os.path.join(trec_dir, "TREC_Train_Cxx_1.json")
    
    if os.path.exists(train_input):
        convert_trec_to_dblp_format(train_input, train_output)
    else:
        print(f"训练集文件不存在: {train_input}")
    
    # 转换测试集
    test_input = os.path.join(trec_dir, "TREC_Test_Cxx.json")
    test_output = os.path.join(trec_dir, "TREC_Test_dblp_format.json")
    
    if os.path.exists(test_input):
        convert_trec_to_dblp_format(test_input, test_output)
    else:
        print(f"测试集文件不存在: {test_input}")

if __name__ == "__main__":
    main() 