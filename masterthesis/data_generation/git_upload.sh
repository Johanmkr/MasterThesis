#!/bin/bash

# hours
hours="$1"

total_iterations=$((hours *4))

for ((i = 1; i<= total_iterations; i++)); do 
    git add README.md
    git commit -m"Automatic commit - Iteration $i"
    git pull
    git push
    
    # Calculate some times
    current_time=$(date +%H:%M)
    current_minutes=$(date -d "$current_time" +%s)
    minutes_to_next_update=15
    new_minutes=$((current_minutes + minutes_to_next_update * 60))
    new_time=$(date -d "@$new_minutes" +%H:%M)

    echo "Last updated at $current_time, next update will be at $new_time!"
    for ((i = "$minutes_to_next_update"; i>= 1;  i--)); do
	echo "Update in: $i minute(s)..."
        sleep 60	
    done
done 
