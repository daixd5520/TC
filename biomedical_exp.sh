#!/bin/bash

echo "============================================================"
echo "开始评估Biomedical数据集"
echo "============================================================"

# # Direct evaluation
python experiment_runner.py --config configs/direct/biomedical_lora_direct.yaml --mode eval

# # Zero-shot evaluation
# python experiment_runner.py --config configs/zeroshot/biomedical_lora_zeroshotCoT.yaml --mode eval
python experiment_runner.py --config /mnt/data1/TC/TextClassDemo/configs/zeroshot/biomedical_lora_zeroshotCoT.yaml --mode eval

# Voting evaluation
python experiment_runner.py --config confgs/vote/biomedical_lora_zeroshotCoT_vote5.yaml --mode eval

echo "============================================================"
echo "Biomedical数据集评估完成"
echo "============================================================" 