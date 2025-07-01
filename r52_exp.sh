# python experiment_runner.py --config configs/r52_base_direct.yaml --mode eval
# python experiment_runner.py --config configs/r52_base_zeroshotCoT.yaml --mode eval
# python experiment_runner.py --config configs/r52_lora_direct.yaml --mode eval
# python experiment_runner.py --config configs/r52_lora_zeroshotCoT.yaml --mode eval
python experiment_runner.py --config configs/vote/r52_base_zeroshotCoT_vote5.yaml --mode eval
python experiment_runner.py --config configs/vote/r52_lora_zeroshotCoT_vote5.yaml --mode eval
python experiment_runner.py --config configs/vote/ohsumed_base_zeroshotCoT_vote5.yaml --mode eval
python experiment_runner.py --config configs/vote/ohsumed_lora_zeroshotCoT_vote5.yaml --mode eval