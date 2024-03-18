#!/bin/bash

# Locate simulation directory
SIMULATIONdir="/uio/hume/student-u00/johanmkr/Documents/NbodySimulation/gevolution-1.2"
# Locate data generation directory
DataGENERATIONdir="$(dirname "$(readlink -f "$0")")"

# Move to simulation directory
cd "$SIMULATIONdir"

# Check if executable
if [ ! -x "gevolution" ]; then
	make
fi 

# Execute gr simulation
# first command line argument
ini_file="$1"

start_time=$(date +%s)
#Testing
# echo "mpirun -np 16 ./gevolution -n 4 -m 4 -s $newton_ini"
# echo "mpirun -np 16 ./gevolution -n 4 -m 4 -s $gr_ini"

# # Execution
mpirun -np 64 ./gevolution -n 8 -m 8 -s $ini_file
end_time=$(date +%s)
elapsed_time=$(echo "$end_time - $start_time" | bc)

# go back to data generation directory
cd "$DataGENERATIONdir"
