#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

echo "开始运行Direct实验配置..."

# Direct评估配置
echo "=== 运行Direct评估配置 ==="

# echo "运行 cr_base_direct.yaml..."
# python experiment_runner.py --config configs/direct/cr_base_direct.yaml --mode eval
# echo "cr_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

# echo "运行 biomedical_base_direct.yaml..."
# python experiment_runner.py --config configs/direct/biomedical_base_direct.yaml --mode eval
# echo "biomedical_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

# echo "运行 r52_base_direct.yaml..."
# python experiment_runner.py --config configs/direct/r52_base_direct.yaml --mode eval
# echo "r52_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

# echo "运行 dblp_base_direct.yaml..."
# python experiment_runner.py --config configs/direct/dblp_base_direct.yaml --mode eval
# echo "dblp_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

echo "运行 ohsumed_base_direct.yaml..."
python experiment_runner.py --config configs/direct/ohsumed_base_direct.yaml --mode eval
echo "ohsumed_base_direct.yaml 评估完成，暂停5秒..."
sleep 5

# echo "运行 trec_base_direct.yaml..."
# python experiment_runner.py --config configs/direct/trec_base_direct.yaml --mode eval
# echo "trec_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

echo "所有Direct实验配置运行完成！" 


# #!/bin/bash

# # 设置工作目录
# cd "$(dirname "$0")"

# echo "开始运行Direct实验配置..."

# # Direct评估配置
# echo "=== 运行Direct评估配置 ==="

# echo "运行 cr_base_direct.yaml..."
# accelerate launch --num_processes 1 experiment_runner.py --config configs/direct/cr_base_direct.yaml --mode eval
# echo "cr_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

# echo "运行 biomedical_base_direct.yaml..."
# accelerate launch --num_processes 1 experiment_runner.py --config configs/direct/biomedical_base_direct.yaml --mode eval
# echo "biomedical_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

# echo "运行 r52_base_direct.yaml..."
# accelerate launch --num_processes 1 experiment_runner.py --config configs/direct/r52_base_direct.yaml --mode eval
# echo "r52_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

# echo "运行 dblp_base_direct.yaml..."
# accelerate launch --num_processes 1 experiment_runner.py --config configs/direct/dblp_base_direct.yaml --mode eval
# echo "dblp_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

# echo "运行 ohsumed_base_direct.yaml..."
# accelerate launch --num_processes 1 experiment_runner.py --config configs/direct/ohsumed_base_direct.yaml --mode eval
# echo "ohsumed_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

# echo "运行 trec_base_direct.yaml..."
# accelerate launch --num_processes 1 experiment_runner.py --config configs/direct/trec_base_direct.yaml --mode eval
# echo "trec_base_direct.yaml 评估完成，暂停5秒..."
# sleep 5

# echo "所有Direct实验配置运行完成！" 