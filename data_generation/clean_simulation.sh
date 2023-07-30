#!/bin/bash

#input file with seeds
seeds="$1"

# Source the functions needed
source func_lib.sh

while IFS= read -r seed; do 
	clean_up_from_seed $seed
done < "$seeds"

