#!/bin/bash

echo "============================================================"
echo "开始训练所有数据集"
echo "============================================================"

# Train Biomedical dataset
echo "🚀 第1步: 训练Biomedical数据集"
bash run_train_biomedical.sh

echo "⏳ 等待20秒..."
sleep 20

# Train CR dataset
# echo "🚀 Step 2: Train CR dataset"
# bash run_train_cr.sh

echo "⏳ 等待20秒..."
sleep 20

# Train DBLP dataset
echo "🚀 第3步: 训练DBLP数据集"
bash run_train_dblp.sh

echo "⏳ 等待20秒..."
sleep 20

# Train TREC dataset
echo "🚀 第4步: 训练TREC数据集"
bash run_train_trec.sh

echo "============================================================"
echo "所有数据集训练完成!"
echo "============================================================"
echo "✅ Biomedical: llama3.1-8b_biomedical_direct_lora"
echo "✅ CR: llama3.1-8b_cr_direct_lora"
echo "✅ DBLP: llama3.1-8b_dblp_direct_lora"
echo "✅ TREC: llama3.1-8b_trec_direct_lora"
echo "============================================================" 
