#!/bin/bash

# varying parameters:
newton_augmentations=(0.0 0.25 0.5 0.75 1.0)
dropout=(0.0 0.25 0.5 0.75)
learning_rates=(0.00001 0.0001 0.001 0.01 0.1 1.0)

for na in "${newton_augmentations[@]}"
do
    for d in "${dropout[@]}"
    do
        for lr in "${learning_rates[@]}"
        do
            echo "Running CNN with newton_augmentation: $na, dropout: $d, learning rate: $lr"
            python3 systematic_tests.py $na $d $lr
        done
    done
done
