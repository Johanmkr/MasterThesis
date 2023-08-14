#!/bin/bash

file="README.md"


# Set the start and end subheadings
start_subheading="## Tree"
end_subheading="##"

# Temp file
temp_file=$(mktemp)

# Flags
keep=true
remove=false

while IFS= read -r line; do
    # Control the keep and remove flag
    if [[ $line == *"$start_subheading"* ]]; then 
        keep=false  # Stops keeping if ## Tree is encountered
    elif [[ $line == *"$end_subheading"* ]]; then
        remove=false # If any other subheading occurs
        keep=true
    fi
    # Update or keep content
    if ! $remove; then
        if $keep; then 
            echo "$line" >> "$temp_file"    # simply copy line
        else    #   write tree output
            echo "$start_subheading" >> "$temp_file" 
            # echo "" >> "$temp_file"
            tree . | sed -e 's/^/    /' >> "$temp_file" 
            echo "Updated on $(date +%F)" >> "$temp_file"
            remove=true
        fi
    fi

    
done < "$file"

#TODO: Fix the files below initialisation in the tree

# Loop through the temp file and remove lines below initialisaiton in the tree


mv "$temp_file" "$file"
    

