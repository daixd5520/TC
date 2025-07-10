#!/bin/bash

echo "============================================================"
echo "开始评估CR数据集"
echo "============================================================"

# 直接评估
python experiment_runner.py --config configs/direct/cr_lora_direct.yaml --mode eval

# 零样本评估
python experiment_runner.py --config configs/zeroshot/cr_lora_zeroshotCoT.yaml --mode eval

# 投票评估
python experiment_runner.py --config configs/vote/cr_lora_zeroshotCoT_vote5.yaml --mode eval

echo "============================================================"
echo "CR数据集评估完成"
echo "============================================================" 