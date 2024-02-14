#!/bin/bash

ts=(100 200 400 800 1500)

for ts in "${ts[@]}"
do
    echo "Running CNN with training seeds: $ts"
    python3 run_controller.py varying_ts/v_ts_${ts}.yaml
done
