import json

file_path = "/mnt/data1/TC/TextClassDemo/data/bak_ohsumed_Train_alpaca_noCoT.json"

# 读取原始数据
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 修改每条数据的 instruction 字段
for item in data:
    item["instruction"] = ""

# 保存到新文件（或覆盖原文件）
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("所有instruction字段已清空。")