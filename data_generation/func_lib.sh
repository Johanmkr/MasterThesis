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

    # Change output path
    local newtonpath="$DataSTORAGEdir/seed${seed}/newton/"
    local grpath="$DataSTORAGEdir/seed${seed}/gr/"
    sed -i "s|^\(output path\s*=\s*\).*|\1$newtonpath|" seed${seed}newton.ini
    sed -i "s|^\(output path\s*=\s*\).*|\1$grpath|" seed${seed}gr.ini
    
    cd "$DataGENERATIONdir"
}


generate_initialisation 3
