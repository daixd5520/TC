import json

# 读取源文件
with open('ohsumed_Test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换后的数据
converted_data = []

# 转换每条记录
for item in data:
    new_item = {
        "instruction": "你是一位顶尖的医学信息学家，专长于 OHSUMED 数据集分类。请仔细阅读以下文本摘要。你的任务是进行精准分类，并仅返回唯一的类别ID作为结果。例如，如果文本属于心血管疾病，你的输出就应该是 'C14'。现在，请处理以下文本：",
        "input": item["text"],
        "output": item["label"]
    }
    converted_data.append(new_item)

# 写入新文件
with open('ohsumed_Test_alpaca_noCoT.json', 'w', encoding='utf-8') as f:
    # 每个记录单独一行，使用ensure_ascii=False保证中文正确显示
    for item in converted_data:
        json.dump(item, f, ensure_ascii=False)
        f.write(',')