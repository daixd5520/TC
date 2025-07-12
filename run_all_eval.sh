#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

echo "开始运行所有评估配置..."

# Direct评估配置
echo "=== 运行Direct评估配置 ==="
python experiment_runner.py --config configs/direct/biomedical_lora_direct.yaml --mode eval
echo "biomedical_lora_direct.yaml 评估完成，暂停5秒..."
sleep 5

python experiment_runner.py --config configs/direct/cr_lora_direct.yaml --mode eval
echo "cr_lora_direct.yaml 评估完成，暂停5秒..."
sleep 5

python experiment_runner.py --config configs/direct/dblp_lora_direct.yaml --mode eval
echo "dblp_lora_direct.yaml 评估完成，暂停5秒..."
sleep 5

python experiment_runner.py --config configs/direct/trec_lora_direct.yaml --mode eval
echo "trec_lora_direct.yaml 评估完成，暂停5秒..."
sleep 5

# Vote评估配置
echo "=== 运行Vote评估配置 ==="
python experiment_runner.py --config configs/vote/biomedical_lora_zeroshotCoT_vote5.yaml --mode eval
echo "biomedical_lora_zeroshotCoT_vote5.yaml 评估完成，暂停5秒..."
sleep 5

python experiment_runner.py --config configs/vote/cr_lora_zeroshotCoT_vote5.yaml --mode eval
echo "cr_lora_zeroshotCoT_vote5.yaml 评估完成，暂停5秒..."
sleep 5

python experiment_runner.py --config configs/vote/dblp_lora_zeroshotCoT_vote5.yaml --mode eval
echo "dblp_lora_zeroshotCoT_vote5.yaml 评估完成，暂停5秒..."
sleep 5

python experiment_runner.py --config configs/vote/trec_lora_zeroshotCoT_vote5.yaml --mode eval
echo "trec_lora_zeroshotCoT_vote5.yaml 评估完成，暂停5秒..."
sleep 5

# Zeroshot评估配置
echo "=== 运行Zeroshot评估配置 ==="
python experiment_runner.py --config configs/zeroshot/biomedical_lora_zeroshotCoT.yaml --mode eval
echo "biomedical_lora_zeroshotCoT.yaml 评估完成，暂停5秒..."
sleep 5

python experiment_runner.py --config configs/zeroshot/cr_lora_zeroshotCoT.yaml --mode eval
echo "cr_lora_zeroshotCoT.yaml 评估完成，暂停5秒..."
sleep 5

python experiment_runner.py --config configs/zeroshot/dblp_lora_zeroshotCoT.yaml --mode eval
echo "dblp_lora_zeroshotCoT.yaml 评估完成，暂停5秒..."
sleep 5

python experiment_runner.py --config configs/zeroshot/trec_lora_zeroshotCoT.yaml --mode eval
echo "trec_lora_zeroshotCoT.yaml 评估完成，暂停5秒..."
sleep 5

echo "所有评估配置运行完成！" 