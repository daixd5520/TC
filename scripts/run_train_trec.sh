#!/bin/bash

echo "============================================================"
echo "开始训练TREC数据集"
echo "============================================================"

llamafactory-cli train \
    --model_name_or_path /mnt/data1/TC/TextClassDemo/llama3.1-8b \
    --stage sft \
    --do_train True \
    --dataset trec_direct \
    --finetuning_type lora \
    --lora_target all \
    --template default \
    --output_dir llama3.1-8b_trec_direct_lora \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy epoch \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss True \
    --fp16 True \
    --save_only_model True

echo "============================================================"
echo "TREC数据集训练完成"
echo "============================================================" 