"""
将 ohsumed_Train_alpaca_noCoT_updated.json 中每条数据的 instruction 字段后拼接
" Let's classify step by step:"，并保存为 ohsumed_Train_zeroshotCoT.json
"""

import json
from typing import List, Dict, Any

INPUT_PATH: str = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Train_alpaca_noCoT_updated.json"
OUTPUT_PATH: str = "/mnt/data1/TC/TextClassDemo/data/ohsumed_Train_zeroshotCoT.json"
APPEND_TEXT: str = " Let's classify step by step:"

def validate_item(item: Dict[str, Any]) -> bool:
    """
    校验每条数据是否包含 instruction 字段且为字符串类型
    """
    return "instruction" in item and isinstance(item["instruction"], str)

def process_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对每条数据的 instruction 字段拼接 APPEND_TEXT
    """
    result: List[Dict[str, Any]] = []
    for item in data:
        if not validate_item(item):
            raise ValueError(f"数据项缺少'instruction'字段或类型错误: {item}")
        # 拼接字符串
        new_item = item.copy()
        new_item["instruction"] = new_item["instruction"] + APPEND_TEXT
        result.append(new_item)
    return result

def main() -> None:
    """
    主函数，读取原始数据，处理后写入新文件
    """
    # 读取原始数据
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    
    # 处理数据
    processed_data: List[Dict[str, Any]] = process_data(data)
    
    # 写入新文件
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，已保存到: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()