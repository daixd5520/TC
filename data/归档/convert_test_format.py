#!/usr/bin/env python3
"""
转换测试文件格式的脚本
将text/label改为input/output，并将标签转换为Cxx格式
"""

import json
import os

def convert_cr_format(input_file, output_file):
    """转换CR数据集格式"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    for item in data:
        text = item['text']
        label = item['label']
        
        # CR标签映射
        if label == 'positive':
            output = 'C01'
        elif label == 'negative':
            output = 'C02'
        else:
            print(f"警告：未知的CR标签: {label}")
            continue
        
        converted_item = {
            "input": text,
            "output": output
        }
        converted_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"CR数据集转换完成：{len(converted_data)} 条数据")

def convert_numeric_format(input_file, output_file, dataset_name):
    """转换数字标签格式（Biomedical和DBLP）"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    for item in data:
        text = item['text']
        label = item['label']
        
        # 将数字转换为Cxx格式
        try:
            label_num = int(label)
            output = f"C{label_num:02d}"
        except ValueError:
            print(f"警告：{dataset_name}数据集中的非数字标签: {label}")
            continue
        
        converted_item = {
            "input": text,
            "output": output
        }
        converted_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"{dataset_name}数据集转换完成：{len(converted_data)} 条数据")

def convert_trec_format(input_file, output_file):
    """转换TREC数据集格式，text/label转input/output，label直接保留为Cxx格式"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    for item in data:
        text = item['text']
        label = item['label']
        # 直接保留label为Cxx格式
        converted_item = {
            "input": text,
            "output": label
        }
        converted_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    print(f"TREC数据集转换完成：{len(converted_data)} 条数据")

def main():
    """主函数"""
    
    # CR数据集转换
    cr_input = "/mnt/data1/TC/TextClassDemo/data/CR/CR_Test.json"
    cr_output = "/mnt/data1/TC/TextClassDemo/data/CR/CR_Test_Cxx.json"
    if os.path.exists(cr_input):
        convert_cr_format(cr_input, cr_output)
    else:
        print(f"CR测试文件不存在: {cr_input}")
    
    # Biomedical数据集转换
    biomedical_input = "/mnt/data1/TC/TextClassDemo/data/Biomedical/Biomedical_Test.json"
    biomedical_output = "/mnt/data1/TC/TextClassDemo/data/Biomedical/Biomedical_Test_Cxx.json"
    if os.path.exists(biomedical_input):
        convert_numeric_format(biomedical_input, biomedical_output, "Biomedical")
    else:
        print(f"Biomedical测试文件不存在: {biomedical_input}")
    
    # DBLP数据集转换
    dblp_input = "/mnt/data1/TC/TextClassDemo/data/dblp/dblp_Test.json"
    dblp_output = "/mnt/data1/TC/TextClassDemo/data/dblp/dblp_Test_Cxx.json"
    if os.path.exists(dblp_input):
        convert_numeric_format(dblp_input, dblp_output, "DBLP")
    else:
        print(f"DBLP测试文件不存在: {dblp_input}")

    # TREC数据集转换
    trec_input = "/mnt/data1/TC/TextClassDemo/data/TREC/TREC_Test_Cxx.json"
    trec_output = "/mnt/data1/TC/TextClassDemo/data/TREC/TREC_Test_Cxx_new.json"
    if os.path.exists(trec_input):
        convert_trec_format(trec_input, trec_output)
    else:
        print(f"TREC测试文件不存在: {trec_input}")

if __name__ == "__main__":
    main() 