# CoT数据生成脚本说明

本文档说明了为biomedical、CR、dblp数据集创建的CoT（Chain of Thought）数据生成脚本。

## 创建的文件

### 1. Biomedical数据集CoT生成脚本
- **文件路径**: `TC/TextClassDemo/data/generate_biomedical_cot.py`
- **功能**: 为Biomedical数据集生成CoT推理数据
- **类别数量**: 20个类别
- **类别映射**: 
  - C01: aging, C02: chemistry, C03: cats, C04: glucose, C05: potassium
  - C06: lung, C07: erythrocytes, C08: lymphocytes, C09: spleen, C10: mutation
  - C11: skin, C12: norepinephrine, C13: insulin, C14: prognosis, C15: risk
  - C16: myocardium, C17: sodium, C18: mathematics, C19: swine, C20: temperature

### 2. CR数据集CoT生成脚本
- **文件路径**: `TC/TextClassDemo/data/generate_cr_cot.py`
- **功能**: 为CR（Customer Review）数据集生成CoT推理数据
- **类别数量**: 2个类别
- **数据格式**: 使用`text`和`label`字段，标签为字符串格式的情感词（如"positive"、"negative"）
- **标签转换**: 自动将情感标签转换为Cxx格式（如"positive"→"C01"，"negative"→"C02"）
- **类别映射**:
  - C01: positive (正面情感)
  - C02: negative (负面情感)

### 3. DBLP数据集CoT生成脚本
- **文件路径**: `TC/TextClassDemo/data/generate_dblp_cot.py`
- **功能**: 为DBLP数据集生成CoT推理数据
- **类别数量**: 6个类别
- **数据格式**: 使用`text`和`label`字段，标签为字符串格式的数字（如"1"、"2"、"3"）
- **标签转换**: 自动将数字标签转换为Cxx格式（如"2"→"C02"）
- **类别映射**:
  - C01: Database (DB)
  - C02: Artificial Intelligence (AI)
  - C03: Software Engineering / Computer Architecture (SE/CA)
  - C04: Computer Networks (NET)
  - C05: Data Mining (DM)
  - C06: Security (SEC)

## 修改的提示词文件

### 1. Biomedical提示词文件
- **文件路径**: `TC/TextClassDemo/configs/prompts/biomedical_prompt.txt`
- **修改内容**: 将类别编号从1-20改为C01-C20格式

### 2. CR提示词文件
- **文件路径**: `TC/TextClassDemo/configs/prompts/CR_prompt.txt`
- **修改内容**: 将类别从positive/negative改为C01/C02格式

### 3. DBLP提示词文件
- **文件路径**: `TC/TextClassDemo/configs/prompts/dblp_prompt.txt`
- **修改内容**: 将类别编号从1-6改为C01-C06格式

## 脚本特性

所有CoT生成脚本都具备以下特性：

1. **多API轮换**: 支持5个API客户端轮换使用，提高处理效率
2. **断点续传**: 支持从断点继续处理，避免重复工作
3. **并行处理**: 使用线程池并行处理多个样本
4. **错误重试**: 支持API调用失败时的指数退避重试机制
5. **进度监控**: 实时显示处理进度和统计信息
6. **数据验证**: 验证生成的CoT响应质量
7. **定期保存**: 定期保存处理结果，防止数据丢失

## 使用方法

### 运行Biomedical数据集CoT生成
```bash
cd TC/TextClassDemo/data
python generate_biomedical_cot.py
```

### 运行CR数据集CoT生成
```bash
cd TC/TextClassDemo/data
python generate_cr_cot.py
```

### 运行DBLP数据集CoT生成
```bash
cd TC/TextClassDemo/data
python generate_dblp_cot.py
```

## 输出文件

每个脚本会生成以下文件：

1. **CoT数据文件**: `{Dataset}_Train_cot.json` - 包含生成的CoT推理数据
2. **断点文件**: `{Dataset}_Train_cot_checkpoint.json` - 用于断点续传

## 数据格式

### 输入数据格式

不同数据集使用不同的字段名：

1. **TREC数据集**: 使用`text`和`label`字段
2. **R52数据集**: 使用`input`和`output`字段  
3. **Biomedical数据集**: 使用`text`和`label`字段，标签为字符串格式的数字（如"8"、"14"）
4. **CR数据集**: 使用`text`和`label`字段，标签为字符串格式的情感词（如"positive"、"negative"）
5. **DBLP数据集**: 使用`text`和`label`字段，标签为字符串格式的数字

### 输出数据格式

生成的CoT数据格式为：
```json
{
  "instruction": "专家角色说明和任务描述",
  "input": "原始文本",
  "output": "详细的CoT推理过程，包含步骤分析和最终分类"
}
```

### 标签格式转换

- **TREC**: 直接使用原始标签（如"C01"）
- **R52**: 直接使用原始标签（如"C01"）
- **Biomedical**: 将数字标签转换为Cxx格式（如"8"→"C08"）
- **CR**: 将情感标签转换为Cxx格式（如"positive"→"C01"，"negative"→"C02"）
- **DBLP**: 将数字标签转换为Cxx格式（如"2"→"C02"）

## 注意事项

1. 确保输入数据文件存在且格式正确
2. 检查API密钥是否有效
3. 可以根据需要调整批次大小和最大样本数
4. 处理过程中可以随时中断，会自动保存进度
5. 生成的CoT数据会自动验证和修正分类标签 