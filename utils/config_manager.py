"""
配置管理模块
支持从YAML配置文件和命令行参数加载配置
"""
import os
import yaml
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    base_model_path: str
    adapter_path: str
    use_lora: bool


@dataclass
class DataConfig:
    """数据配置"""
    data_path: str
    dataset_name: str


@dataclass
class GenerationConfig:
    """生成参数配置"""
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int


@dataclass
class OutputConfig:
    """输出配置"""
    base_output_dir: str
    experiment_name: str
    
    def get_output_dir(self, adapter_path: str, dataset_name: str) -> str:
        """获取实验输出目录，基于adapter名称和数据集名称"""
        # 从adapter路径中提取adapter名称
        adapter_name = os.path.basename(adapter_path.rstrip('/'))
        return os.path.join(self.base_output_dir, f"{self.experiment_name}_{adapter_name}_{dataset_name}")


@dataclass
class ExperimentConfig:
    """实验配置总类"""
    model: ModelConfig
    data: DataConfig
    generation: GenerationConfig
    training: TrainingConfig
    output: OutputConfig
    
    def get_output_dir(self) -> str:
        """获取实验输出目录"""
        return self.output.get_output_dir(self.model.adapter_path, self.data.dataset_name)


class PromptManager:
    """提示词管理器"""
    
    def __init__(self, prompts_dir: str = "configs/prompts"):
        self.prompts_dir = Path(prompts_dir)
    
    def load_prompt_template(self, dataset_name: str) -> str:
        """加载数据集对应的提示词模板"""
        prompt_file = self.prompts_dir / f"{dataset_name}_prompt.txt"
        
        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            # 如果找不到特定数据集的提示词，使用默认模板
            default_file = self.prompts_dir / "example_dataset_prompt.txt"
            if default_file.exists():
                with open(default_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                # 如果连默认模板都没有，返回一个简单的模板
                return "Text: {text}\n\nPlease classify this text:"
    
    def build_prompt(self, dataset_name: str, text: str, **kwargs) -> str:
        """构建提示词"""
        template = self.load_prompt_template(dataset_name)
        return template.format(text=text, **kwargs)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config: Optional[ExperimentConfig] = None
        self.prompt_manager = PromptManager()
    
    def load_from_yaml(self, config_path: str) -> ExperimentConfig:
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return self._dict_to_config(config_dict)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """将字典转换为配置对象"""
        model_config = ModelConfig(**config_dict['model'])
        data_config = DataConfig(**config_dict['data'])
        generation_config = GenerationConfig(**config_dict['generation'])
        training_config = TrainingConfig(**config_dict['training'])
        output_config = OutputConfig(**config_dict['output'])
        
        return ExperimentConfig(
            model=model_config,
            data=data_config,
            generation=generation_config,
            training=training_config,
            output=output_config
        )
    
    def create_from_args(self, args: argparse.Namespace) -> ExperimentConfig:
        """从命令行参数创建配置"""
        config_dict = {
            'model': {
                'base_model_path': args.base_model_path,
                'adapter_path': args.adapter_path,
                'use_lora': args.use_lora
            },
            'data': {
                'data_path': args.data_path,
                'dataset_name': args.dataset_name
            },
            'generation': {
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'do_sample': args.do_sample
            },
            'training': {
                'batch_size': args.batch_size
            },
            'output': {
                'base_output_dir': args.output_dir,
                'experiment_name': args.experiment_name
            }
        }
        
        return self._dict_to_config(config_dict)
    
    def merge_configs(self, base_config: ExperimentConfig, override_config: Dict[str, Any]) -> ExperimentConfig:
        """合并配置，允许部分覆盖"""
        # 这里可以实现配置合并逻辑
        # 暂时返回基础配置
        return base_config


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="医学文本分类实验")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 模型参数
    parser.add_argument("--base-model-path", type=str, help="基础模型路径")
    parser.add_argument("--adapter-path", type=str, help="LoRA适配器路径")
    parser.add_argument("--use-lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--no-lora", action="store_true", help="不使用LoRA")
    
    # 数据参数
    parser.add_argument("--data-path", type=str, help="数据文件路径")
    parser.add_argument("--dataset-name", type=str, default="ohsumed", help="数据集名称")
    
    # 生成参数
    parser.add_argument("--max-new-tokens", type=int, default=256, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.4, help="温度参数")
    parser.add_argument("--top-p", type=float, default=0.9, help="top_p参数")
    parser.add_argument("--do-sample", action="store_true", help="是否采样")
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=512, help="批处理大小")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--experiment-name", type=str, default="experiment", help="实验名称")
    
    # 运行模式
    parser.add_argument("--mode", choices=["eval", "test"], default="eval", help="运行模式")
    
    return parser