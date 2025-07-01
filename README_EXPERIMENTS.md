# 医学文本分类实验系统

## 概述

这是一个工程化的医学文本分类实验系统，支持通过配置文件和命令行参数进行多轮实验，支持多数据集和动态提示词管理。

## 文件结构
TextClassDemo/
├── configs/ # 配置文件目录
│ ├── default_config.yaml # 默认配置
│ ├── lora_experiment.yaml # LoRA实验配置
│ ├── base_model_experiment.yaml # 基础模型实验配置
│ └── prompts/ # 提示词模板目录
│ ├── ohsumed_prompt.txt # Ohsumed数据集提示词
│ └── example_dataset_prompt.txt # 示例数据集提示词
├── utils/ # 工具模块
│ └── config_manager.py # 配置管理
├── experiment_runner.py # 主实验运行器
├── run_experiments.py # 实验管理脚本
└── README_EXPERIMENTS.md # 使用说明


## 主要改进

### 1. 动态输出目录
输出目录现在基于adapter名称和数据集名称自动生成：
{base_output_dir}/{experiment_name}{adapter_name}{dataset_name}


### 2. 多数据集支持
- 支持通过配置文件指定数据集名称
- 动态加载对应数据集的提示词模板
- 提示词模板存储在 `configs/prompts/` 目录

### 3. 配置参数统一管理
所有参数都移到配置文件中：
- 生成参数（max_new_tokens, temperature, top_p, do_sample）
- 训练参数（batch_size）
- 模型和数据路径

## 使用方法

### 1. 运行单个实验

使用配置文件：
```bash
python experiment_runner.py --config configs/lora_experiment.yaml --mode eval
```

使用命令行参数：
```bash
python experiment_runner.py \
    --base-model-path "/mnt/data1/TC/TextClassDemo/llama3.1-8b" \
    --adapter-path "/mnt/data1/TC/TextClassDemo/LLaMA-Factory/llama3.1-8b_ohsumed_lora_english_zeroshotCoT" \
    --data-path "/mnt/data1/TC/TextClassDemo/data/ohsumed_Test_alpaca_noCoT_updated.json" \
    --dataset-name "ohsumed" \
    --use-lora \
    --mode eval
```

### 2. 创建新的提示词模板

```bash
python run_experiments.py --create-prompt my_dataset --prompt-content "Your prompt template here"
```

### 3. 创建新实验配置

```bash
python run_experiments.py --create my_experiment \
    --base-model-path "/path/to/model" \
    --adapter-path "/path/to/adapter" \
    --data-path "/path/to/data" \
    --dataset-name "my_dataset" \
    --use-lora
```

### 4. 批量运行实验

```bash
python run_experiments.py --all --mode eval
```

## 配置参数说明

### 模型配置
- `base_model_path`: 基础模型路径
- `adapter_path`: LoRA适配器路径
- `use_lora`: 是否使用LoRA

### 数据配置
- `data_path`: 数据文件路径
- `dataset_name`: 数据集名称（用于选择提示词模板，和configs/prompts/{dataset_name}_prompt.txt对应）

### 生成参数
- `max_new_tokens`: 最大生成token数
- `temperature`: 温度参数
- `top_p`: top_p参数
- `do_sample`: 是否采样(投票的时候要打开，不然一直是贪心解码)

### 训练配置
- `batch_size`: 批处理大小
- `vote_count`: 单样本投票条数（对同一样本多次推理后投票，提升鲁棒性，默认5；不需要投票就设置1）

### 输出配置
- `base_output_dir`: 输出目录
- `experiment_name`: 实验名称

## 提示词模板

### 创建提示词模板
在 `configs/prompts/` 目录下创建 `{dataset_name}_prompt.txt` 文件，内容格式：
```
You are a text classification expert. Your task is to classify the given text into one of the specified categories.
Category mapping:
{category_mapping}
Text: {text}
Please analyze the text and provide your classification step by step:
```

### 模板变量
- `{text}`: 输入文本
- `{category_mapping}`: 类别映射（可选）

## 输出结果

每个实验会在 `outputs/{experiment_name}_{adapter_name}_{dataset_name}/` 目录下生成：

- `eval_results.json`: 评估结果
- `confusion_matrix.png`: 混淆矩阵图
- `error_analysis.json`: 错误样本分析
- `experiment.log`: 实验日志

## 示例配置

### LoRA实验配置
```yaml
model:
  base_model_path: "/mnt/data1/TC/TextClassDemo/llama3.1-8b"
  adapter_path: "/mnt/data1/TC/TextClassDemo/LLaMA-Factory/llama3.1-8b_ohsumed_lora_english_zeroshotCoT"
  use_lora: true

data:
  data_path: "/mnt/data1/TC/TextClassDemo/data/ohsumed_Test_alpaca_noCoT_updated.json"
  dataset_name: "ohsumed"

generation:
  max_new_tokens: 256
  temperature: 0.4
  top_p: 0.9
  do_sample: false

training:
  batch_size: 512
  vote_count: 5

output:
  base_output_dir: "./outputs"
  experiment_name: "lora_experiment"
```

## 注意事项

1. 确保所有路径都是绝对路径或相对于工作目录的正确路径
2. 实验会自动创建输出目录
3. 日志文件会记录详细的实验过程
4. 支持中断后重新运行实验
5. 提示词模板文件使用UTF-8编码
6. 输出目录名称会自动包含adapter名称和数据集名称，便于区分不同实验