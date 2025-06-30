import json

def build_prompt(text):
    return (
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
        f"Text: {text}\n\n"
        "For example, if the text is related to neoplasms, it belongs to C04. The output must be one of C01-C23 categories. If you output a non-existent category, you will be penalized."
    )

# 读取原始文件
file_path = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Train_alpaca_noCoT_updated.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 更新每个样本
for item in data:
    # 使用原始input构建新的instruction
    original_input = item["input"]
    item["instruction"] = build_prompt(original_input)
    # 清空input
    item["input"] = ""

# 保存回原文件
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("文件更新完成！")