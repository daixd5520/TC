import json
import os

def update_training_data():
    """更新训练数据，添加包含类别对应关系的prompt"""
    
    # 新的prompt模板
    new_instruction = (
        "你是一个医疗文本分类专家。你的任务是将给定的医疗文本分类到23个类别中的一个。\n\n"
        "类别对应关系：\n"
        "C01 - Bacterial Infections and Mycoses (细菌感染和真菌病)\n"
        "C02 - Virus Diseases (病毒疾病)\n"
        "C03 - Parasitic Diseases (寄生虫疾病)\n"
        "C04 - Neoplasms (肿瘤)\n"
        "C05 - Musculoskeletal Diseases (肌肉骨骼疾病)\n"
        "C06 - Digestive System Diseases (消化系统疾病)\n"
        "C07 - Stomatognathic Diseases (口腔颌面疾病)\n"
        "C08 - Respiratory Tract Diseases (呼吸道疾病)\n"
        "C09 - Otorhinolaryngologic Diseases (耳鼻喉疾病)\n"
        "C10 - Nervous System Diseases (神经系统疾病)\n"
        "C11 - Eye Diseases (眼部疾病)\n"
        "C12 - Urologic and Male Genital Diseases (泌尿和男性生殖系统疾病)\n"
        "C13 - Female Genital Diseases and Pregnancy Complications (女性生殖系统疾病和妊娠并发症)\n"
        "C14 - Cardiovascular Diseases (心血管疾病)\n"
        "C15 - Hemic and Lymphatic Diseases (血液和淋巴系统疾病)\n"
        "C16 - Neonatal Diseases and Abnormalities (新生儿疾病和异常)\n"
        "C17 - Skin and Connective Tissue Diseases (皮肤和结缔组织疾病)\n"
        "C18 - Nutritional and Metabolic Diseases (营养和代谢疾病)\n"
        "C19 - Endocrine Diseases (内分泌疾病)\n"
        "C20 - Immunologic Diseases (免疫系统疾病)\n"
        "C21 - Disorders of Environmental Origin (环境源性疾病)\n"
        "C22 - Animal Diseases (动物疾病)\n"
        "C23 - Pathological Conditions, Signs and Symptoms (病理状况、体征和症状)\n\n"
        "请直接给出最终分类结果。例如，如果文本和肿瘤相关，则输出C04。"
    )
    
    # 处理训练数据
    train_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Train_alpaca_noCoT.json"
    output_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Train_alpaca_noCoT_updated.json"
    
    print(f"正在处理训练数据: {train_file}")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 更新每个样本的instruction
    updated_data = []
    for item in train_data:
        updated_item = {
            "instruction": new_instruction,
            "input": item["input"],
            "output": item["output"]
        }
        updated_data.append(updated_item)
    
    # 保存更新后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)
    
    print(f"更新后的训练数据已保存到: {output_file}")
    print(f"总共处理了 {len(updated_data)} 个样本")
    
    # 处理测试数据
    test_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Test_alpaca_noCoT.json"
    test_output_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Test_alpaca_noCoT_updated.json"
    
    print(f"\n正在处理测试数据: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 更新每个样本的instruction
    updated_test_data = []
    for item in test_data:
        updated_item = {
            "instruction": new_instruction,
            "input": item["input"],
            "output": item["output"]
        }
        updated_test_data.append(updated_item)
    
    # 保存更新后的测试数据
    with open(test_output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_test_data, f, ensure_ascii=False, indent=2)
    
    print(f"更新后的测试数据已保存到: {test_output_file}")
    print(f"总共处理了 {len(updated_test_data)} 个测试样本")
    
    # 显示一些示例
    print("\n更新后的数据示例:")
    for i in range(min(3, len(updated_data))):
        print(f"\n样本 {i+1}:")
        print(f"Instruction: {updated_data[i]['instruction'][:100]}...")
        print(f"Input: {updated_data[i]['input']}")
        print(f"Output: {updated_data[i]['output']}")

if __name__ == "__main__":
    update_training_data()