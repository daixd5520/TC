import os
import json
from typing import List, Dict, Set, Any

datasets: List[str] = ["Biomedical", "CR", "dblp", "ohsumed", "R52", "TREC"]
base_path: str = "/mnt/data1/TC/TextClassDemo/data/data"

def count_json_items_and_labels(filepath: str) -> (int, Set[str]):
    """
    统计JSON文件中数据条数和所有类别名称
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{filepath} 不是一个JSON数组")
    labels: Set[str] = set()
    for item in data:
        label = item.get("label")
        if label is None:
            raise ValueError(f"{filepath} 存在无label字段的数据: {item}")
        # 如果label是数字，转为字符串
        labels.add(str(label))
    return len(data), labels

# 构建表格数据
table: List[List[str]] = [["数据集", "Train条数", "Test条数", "类别数", "类别名称"]]
for name in datasets:
    train_file = os.path.join(base_path, f"{name}_Train.json")
    test_file = os.path.join(base_path, f"{name}_Test.json")
    train_count, train_labels = count_json_items_and_labels(train_file)
    test_count, test_labels = count_json_items_and_labels(test_file)
    # 合并train和test的类别
    all_labels = sorted(train_labels.union(test_labels), key=lambda x: (len(x), x))
    table.append([
        name,
        str(train_count),
        str(test_count),
        str(len(all_labels)),
        ", ".join(all_labels)
    ])

# 打印表格
col_widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
for row in table:
    print(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))