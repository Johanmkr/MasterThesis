# Data generation

> Johan Mylius Kroken

## Storage and structure

Data generation routines in the `bash` language for generating N-body simulations to be used for my master thesis. 

The data itself will be stored here: `/mn/stornext/d10/data/johanmkr/simulations` with the initial runs in the folder `gevolution_first_runs`.

1 simulation consist of two runs, with both general and newtonian gravitites, using the same seed. Different simulations uses different seeds. Therefore, a run with seed XXXX will have the following data storage hierarchy. The amound of output depends on the initiation file for the given seed. 

    /mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs
    ├── log.txt
    ├── seedXXXX
    │   ├── gr
    │   │   ├── lcdm_background.dat
    │   │   ├── lcdm_pk000_phi.dat
    │   │   ├── lcdm_pk001_phi.dat
    │   │   ├── lcdm_pk002_phi.dat
    │   │   ├── lcdm_pk003_phi.dat
    │   │   ├── lcdm_pk004_phi.dat
    │   │   ├── lcdm_settings_used.ini
    │   │   ├── lcdm_snap000_phi.h5
    │   │   ├── lcdm_snap001_phi.h5
    │   │   ├── lcdm_snap002_phi.h5
    │   │   ├── lcdm_snap003_phi.h5
    │   │   └── lcdm_snap004_phi.h5
    │   └── newton
    │       ├── lcdm_background.dat
    │       ├── lcdm_pk000_phi.dat
    │       ├── lcdm_pk001_phi.dat
    │       ├── lcdm_pk002_phi.dat
    │       ├── lcdm_pk003_phi.dat
    │       ├── lcdm_pk004_phi.dat
    │       ├── lcdm_settings_used.ini
    │       ├── lcdm_snap000_phi.h5
    │       ├── lcdm_snap001_phi.h5
    │       ├── lcdm_snap002_phi.h5
    │       ├── lcdm_snap003_phi.h5
    │       └── lcdm_snap004_phi.h5
    ├── seedXXXY
    ...
    └── seedYYYY


## Generate the data

### Library of functions:

The file `func_lib.sh` contain som functions to ease the running of simulations, all of which must be provided with a seed as argument when executed:

* `generate_storage_directories` $\to$ generates the directory structure where the data is to be stored. 
* `generate_initialisation` $\to$ generates a subfolder `initialisations/seedXXXX`. Copies `tmp.ini` and creates two new files: 
    * `seedXXXXnewton.ini`
    * `seedXXXXgr.ini`

    where both files are updated with the correct seed, gravity theory and the output path created for the specific seed by the above function. This function also fill in the current seed to the file `log_ini.txt` to keep track of the initialised seeds.
* `execute_simulation` $\to$ checks if the simulation is already run by comparing the seed to the `simulations_run.txt` file. If not, check if initialised, if not initialises. Same for the storage directories. It then executes the simulation by performing the following commands (the executabel is automatically created):

        mpirun -np 16 ./gevolution -n 4 -m 4 -s NEWTON.ini
        mpirun -np 16 ./gevolution -n 4 -m 4 -s GR.ini

    where `NEWTON.ini` and `GR.ini` are seed-specific initiation files. In addition it writes a log to the `log.txt` in the data storage directory and also place the seed used in the `simulations_run.txt` and updates the `README.md` file the seed used, date and time of simulation and its duration in seconds. 

* `clean_up_from_seed` $\to$ deletes any trace of simulations run for a specific seed. It does the following for a given seed:
    * Deletes the initialisation folder and its contents.
    * Deletes the output folder and its contents.
    * Remove information from `log.txt` in storage folder.
    * Remove information from `README.md` file.
    * Remove seed from `simulations_run.txt`

### Seeds
There are files with seeds in them (4 digit numbers starting from 0001 and ascends). A seedfile can be generated with the bash script `generate_seeds.sh` in a following way:

    ./generate_seeds.sh FROM TO filename.txt

where `FROM` and `TO` are integers on the interval [1, 9999]. The outputfile are one seed per line in ascending order of increments 1 from `FROM` to `TO`. 

### Running simulations
Simulations are most easily run with a seed file in the following way:

    ./simulate.sh somefile.txt

where `somefile.txt` contain seeds. Simulations are then performed for each seed in order. Simulation of a single seed can also be done directly with:

    execute_simulation X

where `X` is the seed number. The functions must then be made available in the terminal by `source func_lib.sh`. 

### Cleaning up 
If a seed is simulated by accident or just testing, all information about a seed can be removed, both directories and log information. This can be done for a seed-file by:

    ./clean_simulations.sh seedfile.txt

or manually:

    clean_up_from_seed X

after sourcing the library. **NBNBNB** this also deletes the output data from a given simulation. 

### Keeping track
Simulations left for a long time updates the `README.md` after each individual simulation. Running the script:

    ./git_upload X

will add the `README.md` file and push it to git every 15 minutes for `X` hours. In this way, the progress of the simulations can be tracked online on github. 


## Simulations run

| Seed | Date | Duration (s) |
|------|------|--------------|
|0001|30-07-2023 at 09:56|794|
|0002|30-07-2023 at 10:14|792|
|0003|30-07-2023 at 10:29|794|
|0004|30-07-2023 at 11:21|787|
|0005|30-07-2023 at 11:41|788|
|0006|30-07-2023 at 11:54|793|
|0007|30-07-2023 at 12:08|794|
|0008|30-07-2023 at 12:21|795|
|0009|30-07-2023 at 12:34|790|
|0010|30-07-2023 at 12:47|792|
|0011|30-07-2023 at 13:45|789|
|0012|30-07-2023 at 13:58|794|
|0013|30-07-2023 at 14:11|791|
|0014|30-07-2023 at 14:24|788|
|0015|30-07-2023 at 14:37|793|
|0016|30-07-2023 at 14:51|791|
|0017|30-07-2023 at 15:04|798|
|0018|30-07-2023 at 15:17|786|
|0019|30-07-2023 at 15:30|793|
|0020|30-07-2023 at 15:43|790|
|0021|30-07-2023 at 15:57|787|
