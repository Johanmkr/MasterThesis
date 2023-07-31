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
|0000|31-07-2023 at 10:03|913|
|0001|31-07-2023 at 10:19|913|
|0002|31-07-2023 at 10:34|910|
|0003|31-07-2023 at 10:49|907|
|0004|31-07-2023 at 11:04|915|
|0005|31-07-2023 at 11:19|914|
|0006|31-07-2023 at 11:35|910|
|0007|31-07-2023 at 11:50|911|
|0008|31-07-2023 at 12:05|911|
|0009|31-07-2023 at 12:20|909|
|0010|31-07-2023 at 12:41|913|
|0030|31-07-2023 at 12:48|735|
|0020|31-07-2023 at 12:52|1011|
|0040|31-07-2023 at 12:53|1024|
|0011|31-07-2023 at 12:57|914|
|0031|31-07-2023 at 13:00|739|
|0021|31-07-2023 at 13:08|1008|
|0041|31-07-2023 at 13:10|1027|
|0012|31-07-2023 at 13:12|913|
|0032|31-07-2023 at 13:12|743|
|0033|31-07-2023 at 13:25|741|
|0022|31-07-2023 at 13:25|1004|
|0013|31-07-2023 at 13:27|913|
|0042|31-07-2023 at 13:27|1025|
|0034|31-07-2023 at 13:37|746|
|0023|31-07-2023 at 13:42|1009|
|0014|31-07-2023 at 13:42|909|
|0043|31-07-2023 at 13:44|1020|
|0035|31-07-2023 at 13:50|742|
|0015|31-07-2023 at 13:57|908|
|0024|31-07-2023 at 13:59|1004|
|0044|31-07-2023 at 14:01|1021|
|0036|31-07-2023 at 14:02|746|
|0016|31-07-2023 at 14:12|908|
|0037|31-07-2023 at 14:14|743|
|0025|31-07-2023 at 14:15|1006|
|0045|31-07-2023 at 14:18|1026|
|0038|31-07-2023 at 14:27|745|
|0017|31-07-2023 at 14:28|907|
|0026|31-07-2023 at 14:32|1007|
|0046|31-07-2023 at 14:35|1025|
|0039|31-07-2023 at 14:39|747|
|0018|31-07-2023 at 14:43|911|
|0027|31-07-2023 at 14:49|1004|
|0047|31-07-2023 at 14:52|1019|
|0019|31-07-2023 at 14:58|911|
|0050|31-07-2023 at 15:02|742|
|0028|31-07-2023 at 15:06|1006|
|0150|31-07-2023 at 15:09|747|
|0048|31-07-2023 at 15:09|1023|
|0250|31-07-2023 at 15:11|741|
|0051|31-07-2023 at 15:14|740|
|0350|31-07-2023 at 15:17|910|
|0151|31-07-2023 at 15:22|749|
|0029|31-07-2023 at 15:23|1004|
|0251|31-07-2023 at 15:23|746|
|0049|31-07-2023 at 15:27|1029|
|0052|31-07-2023 at 15:27|745|
|0351|31-07-2023 at 15:32|912|
|0152|31-07-2023 at 15:34|747|
|0252|31-07-2023 at 15:35|743|
|0450|31-07-2023 at 15:38|751|
|0053|31-07-2023 at 15:39|742|
|0550|31-07-2023 at 15:41|743|
|0153|31-07-2023 at 15:46|736|
|0352|31-07-2023 at 15:47|912|
|0253|31-07-2023 at 15:48|740|
|0451|31-07-2023 at 15:50|754|
|0054|31-07-2023 at 15:51|742|
|0650|31-07-2023 at 15:52|745|
|0551|31-07-2023 at 15:53|742|
|0154|31-07-2023 at 15:59|747|
|0254|31-07-2023 at 16:00|738|
|0353|31-07-2023 at 16:02|911|
|0452|31-07-2023 at 16:03|745|
|0055|31-07-2023 at 16:04|744|
|0651|31-07-2023 at 16:04|741|
|0552|31-07-2023 at 16:05|745|
|0155|31-07-2023 at 16:11|741|
