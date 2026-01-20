#!/usr/bin/env python3
"""
将所有数据集转换为统一的instruction/input/output格式
"""

import json
import os

def convert_biomedical_format():
    """转换Biomedical数据集格式"""
    
    input_file = "/mnt/data1/TC/TextClassDemo/data/Biomedical/Biomedical_Train.json"
    output_file = "/mnt/data1/TC/TextClassDemo/data/Biomedical/Biomedical_Train_Cxx.json"
    prompt_file = "/mnt/data1/TC/TextClassDemo/configs/prompts/biomedical_prompt.txt"
    
    if not os.path.exists(input_file):
        print(f"❌ Biomedical输入文件不存在: {input_file}")
        return 0
    

    if os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            instruction = f.read().strip()
    else:
        print(f"⚠️  Biomedical prompt文件不存在，使用默认instruction")
        instruction = (
            "You are a biomedical topic classification expert. Your task is to classify the given medical text into one of 20 biomedical categories.\n"
            "Category mapping:\n"
            "C01 - aging\n"
            "C02 - chemistry\n"
            "C03 - cats\n"
            "C04 - glucose\n"
            "C05 - potassium\n"
            "C06 - lung\n"
            "C07 - erythrocytes\n"
            "C08 - lymphocytes\n"
            "C09 - spleen\n"
            "C10 - mutation\n"
            "C11 - skin\n"
            "C12 - norepinephrine\n"
            "C13 - insulin\n"
            "C14 - prognosis\n"
            "C15 - risk\n"
            "C16 - myocardium\n"
            "C17 - sodium\n"
            "C18 - mathematics\n"
            "C19 - swine\n"
            "C20 - temperature\n"
            "For example, if the text is about insulin regulation, it belongs to C13. The output must be one of C01-C20 categories. If you output a non-existent category, you will be penalized."
        )
    
    print(f"📖 处理Biomedical数据集...")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    converted_data = []
    for item in original_data:
        if 'text' in item and 'label' in item:
            label_num = item['label']
            if label_num.isdigit():
                label_cxx = f"C{int(label_num):02d}"
                converted_item = {
                    "instruction": instruction,
                    "input": item['text'],
                    "output": label_cxx
                }
                converted_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Biomedical转换完成: {len(converted_data)} 个样本")
    return len(converted_data)

def convert_cr_format():
    """转换CR数据集格式"""
    
    input_file = "/mnt/data1/TC/TextClassDemo/data/CR/CR_Train.json"
    output_file = "/mnt/data1/TC/TextClassDemo/data/CR/CR_Train_Cxx.json"
    prompt_file = "/mnt/data1/TC/TextClassDemo/configs/prompts/CR_prompt.txt"
    
    if not os.path.exists(input_file):
        print(f"❌ CR输入文件不存在: {input_file}")
        return 0
    

    if os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            instruction = f.read().strip()
    else:
        print(f"⚠️  CR prompt文件不存在，使用默认instruction")
        instruction = (
            "You are a sentiment analysis expert. Your task is to classify the given customer review text into one of two sentiment categories.\n"
            "Category mapping:\n"
            "C01 - positive\n"
            "C02 - negative\n"
            "For example, if the text expresses satisfaction with a product, it belongs to C01. The output must be either \"C01\" or \"C02\". If you output anything else, you will be penalized."
        )
    
    print(f"📖 处理CR数据集...")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    sentiment_mapping = {
        "positive": "C01",
        "negative": "C02"
    }
    
    converted_data = []
    for item in original_data:
        if 'text' in item and 'label' in item:
            sentiment = item['label'].lower()
            if sentiment in sentiment_mapping:
                label_cxx = sentiment_mapping[sentiment]
                converted_item = {
                    "instruction": instruction,
                    "input": item['text'],
                    "output": label_cxx
                }
                converted_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ CR转换完成: {len(converted_data)} 个样本")
    return len(converted_data)

def convert_dblp_format():
    """转换DBLP数据集格式"""
    
    input_file = "/mnt/data1/TC/TextClassDemo/data/dblp/dblp_Train.json"
    output_file = "/mnt/data1/TC/TextClassDemo/data/dblp/dblp_Train_Cxx.json"
    prompt_file = "/mnt/data1/TC/TextClassDemo/configs/prompts/dblp_prompt.txt"
    
    if not os.path.exists(input_file):
        print(f"❌ DBLP输入文件不存在: {input_file}")
        return 0
    

    if os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            instruction = f.read().strip()
    else:
        print(f"⚠️  DBLP prompt文件不存在，使用默认instruction")
        instruction = (
            "You are a computer science topic classification expert. Your task is to classify the given text into one of 6 computer science research categories.\n"
            "Category mapping:\n"
            "C01 - Database (DB)\n"
            "C02 - Artificial Intelligence (AI)\n"
            "C03 - Software Engineering / Computer Architecture (SE/CA)\n"
            "C04 - Computer Networks (NET)\n"
            "C05 - Data Mining (DM)\n"
            "C06 - Security (SEC)\n"
            "For example, if the text is about relational databases, it belongs to C01. The output must be one of C01-C06 categories. If you output a non-existent category, you will be penalized."
        )
    
    print(f"📖 处理DBLP数据集...")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    converted_data = []
    for item in original_data:
        if 'text' in item and 'label' in item:
            label_num = item['label']
            if label_num.isdigit():
                label_cxx = f"C{int(label_num):02d}"
                converted_item = {
                    "instruction": instruction,
                    "input": item['text'],
                    "output": label_cxx
                }
                converted_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ DBLP转换完成: {len(converted_data)} 个样本")
    return len(converted_data)

def convert_trec_format():
    """转换TREC数据集格式"""
    
    input_file = "/mnt/data1/TC/TextClassDemo/data/TREC/TREC_Train.json"
    output_file = "/mnt/data1/TC/TextClassDemo/data/TREC/TREC_Train_Cxx.json"
    prompt_file = "/mnt/data1/TC/TextClassDemo/configs/prompts/TREC_prompt.txt"
    
    if not os.path.exists(input_file):
        print(f"❌ TREC输入文件不存在: {input_file}")
        return 0
    

    if os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            instruction = f.read().strip()
    else:
        print(f"⚠️  TREC prompt文件不存在，使用默认instruction")
        instruction = (
            "You are a question type classification expert. Your task is to classify the given question into one of 6 categories.\n"
            "Category mapping:\n"
            "abbreviation - Abbreviations or acronyms\n"
            "description - Descriptive or definitional questions\n"
            "entity - Questions about entities (e.g., objects, animals, substances)\n"
            "human - Questions about people, professions or groups\n"
            "location - Questions about places or locations\n"
            "numeric - Questions asking for numbers, amounts, dates or other numeric information\n"
            "For example, if the question asks \"What is caffeine?\", it belongs to description.\n"
            "The output must be one of: abbreviation, description, entity, human, location, numeric.\n"
            "If you output anything else, you will be penalized."
        )
    
    print(f"📖 处理TREC数据集...")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    

    trec_mapping = {
        "abbreviation": "C01",
        "description": "C02", 
        "entity": "C03",
        "human": "C04",
        "location": "C05",
        "numeric": "C06"
    }
    
    converted_data = []
    for item in original_data:
        if 'text' in item and 'label' in item:

            label = item['label']
            if label in trec_mapping:
                label_cxx = trec_mapping[label]
            elif label.startswith('C') and len(label) == 3:
                label_cxx = label
            elif label.isdigit():
                label_cxx = f"C{int(label):02d}"
            else:
                print(f"⚠️  未知的TREC标签: {label}")
                continue
            
            converted_item = {
                "instruction": instruction,
                "input": item['text'],
                "output": label_cxx
            }
            converted_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ TREC转换完成: {len(converted_data)} 个样本")
    return len(converted_data)

def main():
    """主函数"""
    print("=" * 60)
    print("数据集格式转换工具")
    print("=" * 60)
    
    total_samples = 0
    

    total_samples += convert_biomedical_format()
    total_samples += convert_cr_format()
    total_samples += convert_dblp_format()
    total_samples += convert_trec_format()
    
    print("\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)
    print(f"✅ 总共转换: {total_samples} 个样本")
    print("\n生成的文件:")
    print("📁 Biomedical: Biomedical_Train_Cxx.json")
    print("📁 CR: CR_Train_Cxx.json")
    print("📁 DBLP: dblp_Train_Cxx.json")
    print("📁 TREC: TREC_Train_Cxx.json")
    

    print("\n📝 转换格式示例:")
    print("-" * 50)
    example = {
        "instruction": "You are a classification expert...",
        "input": "sample text content",
        "output": "C01"
    }
    print(json.dumps(example, ensure_ascii=False, indent=2))
    print("-" * 50)

if __name__ == "__main__":
    main() 