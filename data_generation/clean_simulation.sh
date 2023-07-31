#!/bin/bash

#input file with seeds
seeds="$1"

# Source the functions needed
source func_lib.sh

echo "Are you sure you want to delete simulations? [y/n]"
read sure

if [ "$sure" = "y" ]; then
	while IFS= read -r seed; do 
		clean_up_from_seed $seed
	done < "$seeds"
else
	echo "Aborting..."
fi
