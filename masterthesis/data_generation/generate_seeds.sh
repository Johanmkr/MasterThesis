#!/bin/bash

#> seeds.txt

#for ((seed=1; seed<=200; seed++)) do
#	echo "$seed" >> seeds.txt
#done

from_number="$1"
to_number="$2"

# Output file to save the numbers
output_file="$3"

> "$output_file"

# Generate numbers in the desired format and save to the output file
for ((i = (($from_number)); i <= (($to_number)); i++)); do
    printf -v formatted_number "%04d" "$i"  # Format the number with leading zeros
    echo "$formatted_number" >> "$output_file"
done
