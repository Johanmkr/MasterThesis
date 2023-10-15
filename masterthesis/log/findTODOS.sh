#!/bin/bash

# Directory
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the output markdown file
output_file="todos.md"

# Remove the output file if it already exists
rm -f "$output_file"

# Start the markdown table
echo "| File | Line | TODO Comment |" >> "$output_file"
echo "|------|------|--------------|" >> "$output_file"

# Recursively search for Python files in subdirectories
while IFS= read -r -d '' file; do
    # Check for TODO occurrences in each line of the file
    line_num=1
    while IFS= read -r line; do
        if [[ "$line" == *"###TODO"* || "$line" == *"### TODO"* ]]; then
            # Extract the TODO comment and the following line
            todo_comment=$(sed -n "${line_num}p" "$file" | sed -e 's/###TODO\|### TODO//')

            # Extract only the filename from the file path
            filename=$(basename "$file")
            
            # Append the file path, line number, and todo comment as a row in the markdown table
            echo "| \`$filename\` | $line_num | \`$todo_comment\` |" >> "$output_file"
        fi
        ((line_num++))
    done < "$file"
done < <(find "$root_dir" -type f -name "*.py" -print0)

# Print completion message
echo "TODO search completed. Results saved to $output_file."



