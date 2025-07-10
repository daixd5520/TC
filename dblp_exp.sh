#!/bin/bash

echo "============================================================"
echo "开始评估DBLP数据集"
echo "============================================================"

# 直接评估
python experiment_runner.py --config configs/direct/dblp_lora_direct.yaml --mode eval

# 零样本评估
python experiment_runner.py --config configs/zeroshot/dblp_lora_zeroshotCoT.yaml --mode eval

# 投票评估
python experiment_runner.py --config configs/vote/dblp_lora_zeroshotCoT_vote5.yaml --mode eval

echo "============================================================"
echo "DBLP数据集评估完成"
echo "============================================================" 