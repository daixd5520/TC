import json
import os

def update_instruction_to_english():
    """将训练数据中的instruction更新为英文版本"""
    
    input_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Train_alpaca_noCoT_updated.json"
    output_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Train_alpaca_noCoT_english.json"
    
    # 新的英文instruction
    english_instruction = (
        "You are a medical text classification expert. Your task is to classify the given medical text into one of 23 categories.\n\n"
        "Category mapping:\n"
        "C01 - Bacterial Infections and Mycoses\n"
        "C02 - Virus Diseases\n"
        "C03 - Parasitic Diseases\n"
        "C04 - Neoplasms\n"
        "C05 - Musculoskeletal Diseases\n"
        "C06 - Digestive System Diseases\n"
        "C07 - Stomatognathic Diseases\n"
        "C08 - Respiratory Tract Diseases\n"
        "C09 - Otorhinolaryngologic Diseases\n"
        "C10 - Nervous System Diseases\n"
        "C11 - Eye Diseases\n"
        "C12 - Urologic and Male Genital Diseases\n"
        "C13 - Female Genital Diseases and Pregnancy Complications\n"
        "C14 - Cardiovascular Diseases\n"
        "C15 - Hemic and Lymphatic Diseases\n"
        "C16 - Neonatal Diseases and Abnormalities\n"
        "C17 - Skin and Connective Tissue Diseases\n"
        "C18 - Nutritional and Metabolic Diseases\n"
        "C19 - Endocrine Diseases\n"
        "C20 - Immunologic Diseases\n"
        "C21 - Disorders of Environmental Origin\n"
        "C22 - Animal Diseases\n"
        "C23 - Pathological Conditions, Signs and Symptoms\n\n"
        "For example, if the text is related to neoplasms, it belongs to C04. The output must be one of C01-C23 categories. If you output a non-existent category, you will be penalized. Let's classify step by step:"
    )
    
    print(f"读取原始数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据样本数: {len(data)}")
    
    # 更新instruction
    updated_data = []
    for i, item in enumerate(data):
        updated_item = {
            "instruction": english_instruction,
            "input": item['input'],
            "output": item['output']
        }
        updated_data.append(updated_item)
        
        # 显示进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1} 个样本")
    
    # 保存更新后的数据
    print(f"保存更新后的数据: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)
    
    print(f"完成！共更新 {len(updated_data)} 个样本")
    
    # 显示示例
    print("\n更新后的数据示例:")
    print(json.dumps(updated_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    update_instruction_to_english()