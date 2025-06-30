import json
from typing import List, Dict, Any

# 类别映射，按顺序写出
category_list = [
    "cocoa", "earn", "acq", "copper", "housing", "money-supply", "coffee", "sugar", "trade", "reserves",
    "ship", "cotton", "grain", "crude", "nat-gas", "cpi", "interest", "money-fx", "alum", "tin",
    "gold", "strategic-metal", "retail", "ipi", "iron-steel", "rubber", "heat", "jobs", "lei", "bop",
    "gnp", "zinc", "veg-oil", "orange", "carcass", "pet-chem", "gas", "wpi", "livestock", "lumber",
    "instal-debt", "meal-feed", "lead", "potato", "nickel", "cpu", "fuel", "jet", "income", "platinum",
    "dlr", "tea"
]
label2cxx: Dict[str, str] = {name: f"C{str(i+1).zfill(2)}" for i, name in enumerate(category_list)}

INPUT_PATH = "/mnt/data1/TC/TextClassDemo/data/R52/R52_Test.json"
OUTPUT_PATH = "/mnt/data1/TC/TextClassDemo/data/R52/R52_Test_Cxx.json"

def main() -> None:
    # 读取原始数据
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # 替换output为Cxx
    new_data: List[Dict[str, Any]] = []
    for item in data:
        label = item["output"]
        if label not in label2cxx:
            raise ValueError(f"未知类别: {label}")
        new_item = item.copy()
        new_item["output"] = label2cxx[label]
        new_data.append(new_item)

    # 保存新json
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"已完成类别替换，新文件保存为: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()