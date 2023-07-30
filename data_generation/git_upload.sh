#!/bin/bash

# hours
hours="$1"

total_iterations=$((hours * 2))

for ((i = 1; i<= total_iterations; i++)); do 
    git add README.md
    git commit -m"Automatic commit - Iteration $i"
    git pull
    git push

    echo "Waiting for 30 mins..."
    sleep 1800
done 
