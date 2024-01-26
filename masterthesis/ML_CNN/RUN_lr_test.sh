#!/bin/bash

# Run the CNN model with different learning rates
learning_rates=(0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0)
for lr in "${learning_rates[@]}"
do
    echo "Running CNN with learning rate: $lr"
    python3 lr_test_RACOON.py $lr
    python3 lr_test_RACOON_relu.py $lr
done