#!/bin/bash

# Paths
DataGENERATIONdir="$(dirname "$(readlink -f "$0")")"
DataSTORAGEdir="/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs"
SIMULATIONdir="/uio/hume/student-u00/johanmkr/Documents/NbodySimulation/gevolution-1.2"

# Generate newton and gr directories to store data for given seed. 
# Args:
#   seed:int
generate_storage_directories(){
    local seed="$1"
    cd "$DataSTORAGEdir"
    mkdir "seed${seed}"
    cd "seed${seed}"
    mkdir "newton"
    mkdir "gr"
    cd "$DataGENERATIONdir"
}

# Generates folder and makes both newton and gr .ini files. 
# Args:
#   seed:int
generate_initialisation(){
    local seed="$1"
    cd "$DataGENERATIONdir/initialisations"
    mkdir "seed${seed}"
    cp "tmp.ini" "seed${seed}/seed${seed}newton.ini"
    cp "tmp.ini" "seed${seed}/seed${seed}gr.ini"
    cd "seed${seed}"

    # Change seed
    sed -i "s/^\(seed\s*=\s*\).*/\1$seed/" seed${seed}newton.ini
    sed -i "s/^\(seed\s*=\s*\).*/\1$seed/" seed${seed}gr.ini

    # Change gravity theory
    sed -i "s/^\(gravity theory\s*=\s*\).*/\1Newton/" seed${seed}newton.ini
    sed -i "s/^\(gravity theory\s*=\s*\).*/\1GR/" seed${seed}gr.ini

    # Change file bases
    sed -i "s/^\(generic file base\s*=\s*\).*/\1newton/" seed${seed}newton.ini
    sed -i "s/^\(generic file base\s*=\s*\).*/\1gr/" seed${seed}gr.ini
    sed -i "s/^\(snapshot file base\s*=\s*\).*/\1newton_snap/" seed${seed}newton.ini
    sed -i "s/^\(snapshot file base\s*=\s*\).*/\1gr_snap/" seed${seed}gr.ini
    sed -i "s/^\(Pk file base\s*=\s*\).*/\1newton_pk/" seed${seed}newton.ini
    sed -i "s/^\(Pk file base\s*=\s*\).*/\1gr_pk/" seed${seed}gr.ini


    # Change output path
    local newtonpath="$DataSTORAGEdir/seed${seed}/newton/"
    local grpath="$DataSTORAGEdir/seed${seed}/gr/"
    sed -i "s|^\(output path\s*=\s*\).*|\1$newtonpath|" seed${seed}newton.ini
    sed -i "s|^\(output path\s*=\s*\).*|\1$grpath|" seed${seed}gr.ini

    echo "$seed" >> ../log_ini.txt
    cd "$DataGENERATIONdir"
}

# Executes the simulation for a given seed
# Args:
#   seed:int
execute_simulation(){
    local seed="$1"
    
    # Check if simulation is already run
    if grep -q "\<$seed\>" "$DataGENERATIONdir/simulations_run.txt"; then
        echo "Simulation with seed: $seed is already run"
        echo ""
    else
        echo "$seed" >> $DataGENERATIONdir/simulations_run.txt
        # Check if initialised 
        if ! grep -q "\<$seed\>" "$DataGENERATIONdir/initialisations/log_ini.txt"; then
            echo "Seed ($seed) not yet initialised, initialising now..."
            generate_initialisation "$seed"
        fi 

        # Check if output directories exists
        if [ ! -d "$DataSTORAGEdir/seed${seed}" ]; then
            echo "Seed ($seed): Output directores do not exist. Creating them now..."
            generate_storage_directories "$seed"
        fi

        cd "$SIMULATIONdir"

        # Check if executable
        if [ ! -x "gevolution" ]; then
            make
        fi 

        local newton_ini="$DataGENERATIONdir/initialisations/seed${seed}/seed${seed}newton.ini"
        local gr_ini="$DataGENERATIONdir/initialisations/seed${seed}/seed${seed}gr.ini"

        start_time=$(date +%s)
        # Testing
        #echo "mpirun -np 16 ./gevolution -n 4 -m 4 -s $newton_ini"
        #echo "mpirun -np 16 ./gevolution -n 4 -m 4 -s $gr_ini"

        # Execution
        mpirun -np 64 ./gevolution -n 8 -m 8 -s $newton_ini
        mpirun -np 64 ./gevolution -n 8 -m 8 -s $gr_ini
        end_time=$(date +%s)
        elapsed_time=$(echo "$end_time - $start_time" | bc)

        cd "$DataSTORAGEdir"
        echo "Successfully ran seed${seed} on $(date)" >> log.txt

        cd "$DataGENERATIONdir"
        formatted_date=$(date +"%d-%m-%Y at %H:%M")
       	echo "|$seed|$formatted_date|$elapsed_time|" >> README.md
       	# echo "$seed" >> simulations_run.txt
        echo ""
    fi
}

# Removes all data related to a given seed
# Args:
#   seed:int
clean_up_from_seed(){
	local seed="$1"
	
	# Delete initialisation folders
	cd "$DataGENERATIONdir/initialisations"
	rm -r seed${seed}
	grep -vi "$seed" "log_ini.txt" >> tmp.txt
	mv tmp.txt log_ini.txt
	
	# Delete output folder
	cd "$DataSTORAGEdir"
	rm -r seed${seed}

	# Remove from log.txt in data storage dir
	grep -vi "seed${seed}" "log.txt" >> tmp.txt
	mv tmp.txt log.txt

	# Remove from readme file if present
	cd "$DataGENERATIONdir"	
	grep -vi "|$seed|" "README.md" >> tmp.md
	mv tmp.md README.md
	
	# Remove from simulations_run.txt
	grep -vi "$seed" "simulations_run.txt" >> tmp.txt
	mv tmp.txt simulations_run.txt
	
	echo "Removed directories and information related to seed ($seed)"
}
