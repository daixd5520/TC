#!/bin/bash

echo "============================================================"
echo "开始评估Biomedical数据集"
echo "============================================================"

# 直接评估
python experiment_runner.py --config configs/direct/biomedical_lora_direct.yaml --mode eval

# 零样本评估
python experiment_runner.py --config configs/zeroshot/biomedical_lora_zeroshotCoT.yaml --mode eval

# 投票评估
python experiment_runner.py --config configs/vote/biomedical_lora_zeroshotCoT_vote5.yaml --mode eval

echo "============================================================"
echo "Biomedical数据集评估完成"
echo "============================================================" 