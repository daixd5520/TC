#!/bin/bash

echo "============================================================"
echo "开始评估所有新数据集"
echo "============================================================"

# Biomedical数据集评估
echo "正在评估Biomedical数据集..."
python experiment_runner.py --config configs/direct/biomedical_lora_direct.yaml --mode eval
# python experiment_runner.py --config configs/zeroshot/biomedical_lora_zeroshotCoT.yaml --mode eval
# python experiment_runner.py --config configs/vote/biomedical_lora_zeroshotCoT_vote5.yaml --mode eval

# CR数据集评估
echo "正在评估CR数据集..."
python experiment_runner.py --config configs/direct/cr_lora_direct.yaml --mode eval
# python experiment_runner.py --config configs/zeroshot/cr_lora_zeroshotCoT.yaml --mode eval
# python experiment_runner.py --config configs/vote/cr_lora_zeroshotCoT_vote5.yaml --mode eval

# DBLP数据集评估
echo "正在评估DBLP数据集..."
python experiment_runner.py --config configs/direct/dblp_lora_direct.yaml --mode eval
# python experiment_runner.py --config configs/zeroshot/dblp_lora_zeroshotCoT.yaml --mode eval
# python experiment_runner.py --config configs/vote/dblp_lora_zeroshotCoT_vote5.yaml --mode eval

# TREC数据集评估
echo "正在评估TREC数据集..."
python experiment_runner.py --config configs/direct/trec_lora_direct.yaml --mode eval
# python experiment_runner.py --config configs/zeroshot/trec_lora_zeroshotCoT.yaml --mode eval
# python experiment_runner.py --config configs/vote/trec_lora_zeroshotCoT_vote5.yaml --mode eval

echo "============================================================"
echo "所有新数据集评估完成"
echo "============================================================" 