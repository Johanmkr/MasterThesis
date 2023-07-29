# Data generation

> Johan Mylius Kroken

Data generation routines in the `bash` language for generating N-body simulations to be used for my master thesis. 

The data itself will be stored here: `/mn/stornext/d10/data/johanmkr/simulations` with the initial runs in the folder `gevolution_first_runs`.

1 simulation consist of two runs, with both general and newtonian gravitites, using the same seed. Different simulations uses different seeds. 


## Todo
* [] Provide a default .ini file template to be modified.
* [] Library script with functions:
    * Paths: current directory, data directory, gevolution directory. 
    * Create directories to store the data for a simulation, separated into `gr` and `newton` parts. Both folders should be within `gevolution_first_runs`: `seedXXXXX/gr` and `seedXXXX/newton`. Args: seed.
    * Create folder within `initialisations` based on seed, and make two copies of the initiation file: `seedXXXXXnewton.ini` and `seedXXXXXgr.ini` with correct gravity theory, and output path according to the seed. Should also write to a log file in initialisations, recording the seeds used. Args: seed. 
    * Function that takes a file (of seeds) and a seed as input and return 0 if seed is found in the file. 
* [] Script that generates new seed, check if it already exists (new seed if yes), creates the folder for data storage and the folder for initialisation and then creates the two initiations scripts. 
* Execution file. Should execute one simulation given a seed, or all simulations not yet executed. 


## Simulations run

| Seed | Date | Duration |
|------|------|----------|
