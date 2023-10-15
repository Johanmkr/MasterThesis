#!/bin/bash

# Input file with seeds
seeds="$1"

# Source the functions needed
source func_lib.sh

#while IFS= read -r seed; do 
#   echo "Executing simulation for seed: $seed"
#   execute_simulation "$seed"
#   echo "Execution completed for seed: $seed"
#done < "$seeds"

# Command to execute the simulation for each seed
for line in $(cat "$seeds"); do
	execute_simulation $line
done    
