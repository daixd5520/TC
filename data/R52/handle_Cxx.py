import json
from typing import List, Dict, Any

INPUT_PATH = "/mnt/data1/TC/TextClassDemo/data/R52/R52_Train.json"
OUTPUT_PATH = "/mnt/data1/TC/TextClassDemo/data/R52/R52_Train_Cxx.json"
MAPPING_TXT_PATH = "/mnt/data1/TC/TextClassDemo/data/R52/R52_category_mapping.txt"

def main() -> None:

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)


    seen = set()
    categories: List[str] = []
    for item in data:
        label = item["output"]
        if label not in seen:
            seen.add(label)
            categories.append(label)


    label2cxx: Dict[str, str] = {
        label: f"C{str(i+1).zfill(2)}" for i, label in enumerate(categories)
    }


    new_data: List[Dict[str, Any]] = []
    for item in data:
        new_item = item.copy()
        new_item["output"] = label2cxx[item["output"]]
        new_data.append(new_item)


    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


    mapping_lines = ["Category mapping:"]
    for i, label in enumerate(categories):
        mapping_lines.append(f"{label2cxx[label]} - {label}")
    mapping_txt = "\n".join(mapping_lines)
    with open(MAPPING_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(mapping_txt)

    print(f"已完成类别映射，映射表保存为: {MAPPING_TXT_PATH}")
    print(f"新数据保存为: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()