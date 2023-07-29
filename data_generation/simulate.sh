#!/bin/bash

# Input file with seeds
seeds="$1"

# Source the functions needed
source func_lib.sh

while IFS= read -r seed; do 
    execute_simulation $seed

    # To be able to track progress from afar
    git add README.md
    git commit -m"Simulated seed $seed"
    git pull
    git push
done < "$seeds"
    