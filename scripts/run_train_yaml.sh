#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# llamafactory-cli train alpaca_swap_config.yaml

# sleep 20

llamafactory-cli train dolly_swap_config.yaml

sleep 20

llamafactory-cli train wizard_swap_config.yaml 