#!/bin/bash

# Input file with seeds
seeds="$1"

# Source the functions needed
source func_lib.sh

while IFS= read -r seed; do 
   execute_simulation $seed
done < "$seeds"
    
