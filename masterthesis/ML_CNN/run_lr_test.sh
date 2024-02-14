#!/bin/bash

lr=(0.00001 0.0001 0.001 0.01 0.1)

for lr in "${lr[@]}"
do
    echo "Running CNN with learning rate: $lr"
    python3 run_controller.py testing/varying_lr/v_lr_${lr}.yaml
done
