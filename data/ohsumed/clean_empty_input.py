import json
import os

def clean_empty_input_data(file_path):
    """清理文件中input为空的数据"""
    print(f"处理文件: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return 0
    
    # 读取原始数据
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"原始数据数量: {len(data)}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return 0
    
    # 统计空input的数量
    empty_input_count = 0
    for item in data:
        if not item.get('input') or item.get('input').strip() == '':
            empty_input_count += 1
    
    print(f"发现空input数据: {empty_input_count} 个")
    
    # 过滤掉input为空的数据
    cleaned_data = []
    for item in data:
        if item.get('input') and item.get('input').strip():
            cleaned_data.append(item)
    
    print(f"清理后数据数量: {len(cleaned_data)}")
    print(f"删除了 {len(data) - len(cleaned_data)} 个空input数据")
    
    # 保存清理后的数据
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        print(f"已保存清理后的数据到: {file_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")
        return 0
    
    return len(cleaned_data)

def main():
    # 文件路径
    checkpoint_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/ohsumed_Train_cot_checkpoint.json"
    cot_file = "/mnt/data1/TC/TextClassDemo/data/ohsumed/ohsumed_Train_cot.json"
    
    print("开始清理空input数据...")
    print("=" * 50)
    
    # 处理 checkpoint 文件
    checkpoint_count = clean_empty_input_data(checkpoint_file)
    
    print("\n" + "=" * 50)
    
    # 处理 cot 文件
    cot_count = clean_empty_input_data(cot_file)
    
    print("\n" + "=" * 50)
    print("清理完成！")
    print(f"checkpoint文件剩余数据: {checkpoint_count}")
    print(f"cot文件剩余数据: {cot_count}")

if __name__ == "__main__":
    main()