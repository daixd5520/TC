import os
import json
from statistics import mean
from transformers import AutoTokenizer

# 设置分词器路径（适配 gemma2-2b）
TOKENIZER_PATH = "/mnt/data1/FT/inspire_data_prep/models/gemma2-2b"

# 你要统计的 json 文件列表（可继续添加）
JSON_FILES = [
    "data/TREC/TREC_Test_Cxx.json",
    "data/R52/R52_Test_Cxx.json",
    "data/Ohsumed/ohsumed_test_Cxx.json",
    "data/CR/CR_Test_Cxx.json",
    "data/CR/CR_Train_Cxx.json"
]

def compute_instruction_stats(file_path, tokenizer):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return

    lengths = []
    for i, item in enumerate(data):
        instruction = item.get("input")
        if instruction is None:
            print(f"⚠️ Warning: missing 'input' field in entry {i} of {file_path}")
            continue
        tokens = tokenizer(instruction, add_special_tokens=False)["input_ids"]
        lengths.append(len(tokens))

    if lengths:
        print(f"📊 File: {file_path}")
        print(f"   ➤ Total samples: {len(lengths)}")
        print(f"   ➤ Max token length: {max(lengths)}")
        print(f"   ➤ Min token length: {min(lengths)}")
        print(f"   ➤ Avg token length: {mean(lengths):.2f}")
    else:
        print(f"⚠️ No valid input fields found in {file_path}")

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    for json_file in JSON_FILES:
        compute_instruction_stats(json_file, tokenizer)

if __name__ == "__main__":
    main()
