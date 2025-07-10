#!/bin/bash

echo "============================================================"
echo "开始评估TREC数据集"
echo "============================================================"

# 直接评估
python experiment_runner.py --config configs/direct/trec_lora_direct.yaml --mode eval

# 零样本评估
python experiment_runner.py --config configs/zeroshot/trec_lora_zeroshotCoT.yaml --mode eval

# 投票评估
python experiment_runner.py --config configs/vote/trec_lora_zeroshotCoT_vote5.yaml --mode eval

echo "============================================================"
echo "TREC数据集评估完成"
echo "============================================================" 