"""
实验管理脚本
支持批量运行多个实验配置
"""
import os
import subprocess
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime


class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("ExperimentManager")
        logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 文件处理器
        log_file = log_dir / f"experiment_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def list_configs(self) -> List[str]:
        """列出所有配置文件"""
        config_files = list(self.config_dir.glob("*.yaml"))
        return [f.name for f in config_files]
    
    def run_single_experiment(self, config_file: str, mode: str = "eval") -> bool:
        """运行单个实验"""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            self.logger.error(f"配置文件不存在: {config_path}")
            return False
        
        self.logger.info(f"开始运行实验: {config_file}")
        
        try:
            # 运行实验
            cmd = [
                "python", "experiment_runner.py",
                "--config", str(config_path),
                "--mode", mode
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                self.logger.info(f"实验 {config_file} 运行成功")
                return True
            else:
                self.logger.error(f"实验 {config_file} 运行失败")
                self.logger.error(f"错误输出: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"运行实验 {config_file} 时发生异常: {e}")
            return False
    
    def run_batch_experiments(self, config_files: List[str], mode: str = "eval") -> Dict[str, bool]:
        """批量运行实验"""
        results = {}
        
        self.logger.info(f"开始批量运行 {len(config_files)} 个实验")
        
        for config_file in config_files:
            success = self.run_single_experiment(config_file, mode)
            results[config_file] = success
        
        # 统计结果
        successful = sum(results.values())
        total = len(results)
        
        self.logger.info(f"批量实验完成: {successful}/{total} 成功")
        
        return results
    
    def create_experiment_config(self, name: str, base_model_path: str, adapter_path: str, 
                                data_path: str, dataset_name: str = "ohsumed", use_lora: bool = True, **kwargs) -> str:
        """创建新的实验配置"""
        config = {
            "model": {
                "base_model_path": base_model_path,
                "adapter_path": adapter_path,
                "use_lora": use_lora
            },
            "data": {
                "data_path": data_path,
                "dataset_name": dataset_name
            },
            "generation": {
                "max_new_tokens": kwargs.get("max_new_tokens", 256),
                "temperature": kwargs.get("temperature", 0.4),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": kwargs.get("do_sample", False)
            },
            "training": {
                "batch_size": kwargs.get("batch_size", 512)
            },
            "output": {
                "base_output_dir": "./outputs",
                "experiment_name": name
            }
        }
        
        config_file = f"{name}.yaml"
        config_path = self.config_dir / config_file
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"创建配置文件: {config_path}")
        return config_file
    
    def create_prompt_template(self, dataset_name: str, prompt_content: str) -> bool:
        """创建提示词模板文件"""
        prompts_dir = Path("configs/prompts")
        prompts_dir.mkdir(exist_ok=True)
        
        prompt_file = prompts_dir / f"{dataset_name}_prompt.txt"
        
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt_content)
            self.logger.info(f"创建提示词模板: {prompt_file}")
            return True
        except Exception as e:
            self.logger.error(f"创建提示词模板失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实验管理器")
    parser.add_argument("--list", action="store_true", help="列出所有配置文件")
    parser.add_argument("--config", type=str, help="运行指定配置文件")
    parser.add_argument("--all", action="store_true", help="运行所有配置文件")
    parser.add_argument("--mode", choices=["eval", "test"], default="eval", help="运行模式")
    parser.add_argument("--create", type=str, help="创建新配置文件")
    parser.add_argument("--base-model-path", type=str, help="基础模型路径")
    parser.add_argument("--adapter-path", type=str, help="适配器路径")
    parser.add_argument("--data-path", type=str, help="数据路径")
    parser.add_argument("--dataset-name", type=str, default="ohsumed", help="数据集名称")
    parser.add_argument("--use-lora", action="store_true", help="使用LoRA")
    parser.add_argument("--no-lora", action="store_true", help="不使用LoRA")
    parser.add_argument("--create-prompt", type=str, help="创建提示词模板文件")
    parser.add_argument("--prompt-content", type=str, help="提示词内容")
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.list:
        configs = manager.list_configs()
        print("可用的配置文件:")
        for config in configs:
            print(f"  - {config}")
    
    elif args.create_prompt:
        if not args.prompt_content:
            print("创建提示词模板需要提供 --prompt-content")
            return
        
        success = manager.create_prompt_template(args.create_prompt, args.prompt_content)
        if success:
            print(f"提示词模板已创建: {args.create_prompt}_prompt.txt")
        else:
            print("提示词模板创建失败")
    
    elif args.create:
        if not all([args.base_model_path, args.adapter_path, args.data_path]):
            print("创建配置文件需要提供 --base-model-path, --adapter-path, --data-path")
            return
        
        use_lora = args.use_lora and not args.no_lora
        config_file = manager.create_experiment_config(
            name=args.create,
            base_model_path=args.base_model_path,
            adapter_path=args.adapter_path,
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            use_lora=use_lora
        )
        print(f"配置文件已创建: {config_file}")
    
    elif args.config:
        success = manager.run_single_experiment(args.config, args.mode)
        if success:
            print(f"实验 {args.config} 运行成功")
        else:
            print(f"实验 {args.config} 运行失败")
    
    elif args.all:
        configs = manager.list_configs()
        results = manager.run_batch_experiments(configs, args.mode)
        
        print("\n实验结果汇总:")
        for config, success in results.items():
            status = "成功" if success else "失败"
            print(f"  {config}: {status}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()