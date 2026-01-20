import json

def build_prompt_bio(text):
    return (
        "You are a biomedical topic classification expert. Your task is to classify the given medical text into one of 20 biomedical categories.\n\n"
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
        "C20 - temperature\n\n"
        f"Text: {text}\n\n"
        "For example, if the text is about insulin regulation, it belongs to C13. The output must be one of C01-C20 categories."
    )

def build_prompt_cr(text):
    return (
        "You are a sentiment analysis expert. Your task is to classify the given customer review text into one of two sentiment categories.\n\n"
        "Category mapping:\n"
        "C01 - positive\n"
        "C02 - negative\n\n"
        f"Text: {text}\n\n"
        "For example, if the text expresses satisfaction with a product, it belongs to C01. The output must be either 'C01' or 'C02'."
    )

def build_prompt_dblp(text):
    return (
        "You are a computer science topic classification expert. Your task is to classify the given text into one of 6 computer science research categories.\n\n"
        "Category mapping:\n"
        "C01 - Database (DB)\n"
"C02 - Artificial Intelligence (AI)\n"
"C03 - Software Engineering / Computer Architecture (SE/CA)\n"
"C04 - Computer Networks (NET)\n"
"C05 - Data Mining (DM)\n"
        "C06 - Security (SEC)\n\n"
        f"Text: {text}\n\n"
        "For example, if the text is about relational databases, it belongs to C01. The output must be one of C01–C06 categories."
    )
    
def build_prompt_trec(text):
    return (
        "You are a question type classification expert. Your task is to classify the given question into one of 6 categories.\n\n"
        "Category mapping:\n"
        "C01 - Questions about entities (e.g., objects, animals, substances)\n"
        "C02 - Questions about people, professions or groups\n"
        "C03 - Descriptive or definitional questions\n"
        "C04 - Questions asking for numbers, amounts, dates or other numeric information\n"
        "C05 - Questions about places or locations\n"
        "C06 - Abbreviations or acronyms\n\n"
        f"Text: {text}\n\n"
        "For example, if the question asks 'What is caffeine?', it belongs to C03. The output must be one of C01-C06 categories."
    )


file_path = "/mnt/data1/TC/TextClassDemo/data/TREC/TREC_Train_Cxx.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


for item in data:

    original_input = item["input"]
    item["instruction"] = build_prompt_trec(original_input)

    item["input"] = ""


with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("文件更新完成！")