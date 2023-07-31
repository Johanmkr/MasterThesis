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
|0255|31-07-2023 at 16:12|739|
|0453|31-07-2023 at 16:15|743|
|0056|31-07-2023 at 16:16|745|
|0652|31-07-2023 at 16:16|741|
|0354|31-07-2023 at 16:18|913|
|0553|31-07-2023 at 16:18|744|
|0156|31-07-2023 at 16:23|741|
|0256|31-07-2023 at 16:25|732|
|0454|31-07-2023 at 16:28|748|
|0057|31-07-2023 at 16:29|740|
|0653|31-07-2023 at 16:29|739|
|0554|31-07-2023 at 16:30|750|
|0355|31-07-2023 at 16:33|907|
|0157|31-07-2023 at 16:36|744|
|0257|31-07-2023 at 16:37|741|
|0455|31-07-2023 at 16:40|752|
|0058|31-07-2023 at 16:41|739|
|0654|31-07-2023 at 16:41|741|
|0555|31-07-2023 at 16:43|738|
|0356|31-07-2023 at 16:48|910|
|0158|31-07-2023 at 16:48|739|
|0258|31-07-2023 at 16:49|740|
|0456|31-07-2023 at 16:53|749|
|0059|31-07-2023 at 16:53|739|
|0655|31-07-2023 at 16:53|745|
|0556|31-07-2023 at 16:55|743|
|0159|31-07-2023 at 17:00|741|
|0259|31-07-2023 at 17:02|742|
|0357|31-07-2023 at 17:03|913|
|0457|31-07-2023 at 17:05|752|
|0060|31-07-2023 at 17:06|740|
|0656|31-07-2023 at 17:06|746|
|0557|31-07-2023 at 17:07|742|
|0160|31-07-2023 at 17:13|752|
|0260|31-07-2023 at 17:14|739|
|0458|31-07-2023 at 17:18|752|
|0061|31-07-2023 at 17:18|743|
|0657|31-07-2023 at 17:18|747|
|0358|31-07-2023 at 17:18|910|
|0558|31-07-2023 at 17:20|744|
|0161|31-07-2023 at 17:25|745|
|0261|31-07-2023 at 17:26|743|
|0459|31-07-2023 at 17:30|751|
|0062|31-07-2023 at 17:30|740|
|0658|31-07-2023 at 17:31|739|
|0559|31-07-2023 at 17:32|741|
|0359|31-07-2023 at 17:33|912|
|0162|31-07-2023 at 17:38|743|
|0262|31-07-2023 at 17:39|740|
|0460|31-07-2023 at 17:43|748|
|0063|31-07-2023 at 17:43|744|
|0659|31-07-2023 at 17:43|741|
|0560|31-07-2023 at 17:45|742|
|0360|31-07-2023 at 17:49|912|
|0163|31-07-2023 at 17:50|744|
|0263|31-07-2023 at 17:51|738|
|0064|31-07-2023 at 17:55|740|
|0461|31-07-2023 at 17:55|743|
|0660|31-07-2023 at 17:55|746|
|0561|31-07-2023 at 17:57|742|
|0164|31-07-2023 at 18:03|739|
|0264|31-07-2023 at 18:03|742|
|0361|31-07-2023 at 18:04|912|
|0065|31-07-2023 at 18:07|744|
|0462|31-07-2023 at 18:07|750|
|0661|31-07-2023 at 18:08|742|
|0562|31-07-2023 at 18:09|747|
|0165|31-07-2023 at 18:15|742|
|0265|31-07-2023 at 18:16|739|
|0362|31-07-2023 at 18:19|911|
|0066|31-07-2023 at 18:20|743|
|0463|31-07-2023 at 18:20|747|
|0662|31-07-2023 at 18:20|737|
|0563|31-07-2023 at 18:22|745|
|0166|31-07-2023 at 18:27|751|
|0266|31-07-2023 at 18:28|745|
|0067|31-07-2023 at 18:32|743|
|0663|31-07-2023 at 18:32|745|
|0464|31-07-2023 at 18:32|750|
|0564|31-07-2023 at 18:34|741|
|0363|31-07-2023 at 18:34|910|
|0167|31-07-2023 at 18:40|747|
|0267|31-07-2023 at 18:41|742|
|0068|31-07-2023 at 18:45|749|
|0664|31-07-2023 at 18:45|745|
|0465|31-07-2023 at 18:45|750|
|0565|31-07-2023 at 18:47|751|
|0364|31-07-2023 at 18:49|908|
|0168|31-07-2023 at 18:52|741|
|0268|31-07-2023 at 18:53|742|
|0069|31-07-2023 at 18:57|740|
|0665|31-07-2023 at 18:57|745|
|0466|31-07-2023 at 18:58|759|
|0566|31-07-2023 at 18:59|742|
|0365|31-07-2023 at 19:05|913|
|0169|31-07-2023 at 19:05|751|
|0269|31-07-2023 at 19:05|751|
|0070|31-07-2023 at 19:09|745|
|0666|31-07-2023 at 19:10|742|
|0467|31-07-2023 at 19:10|753|
|0567|31-07-2023 at 19:11|738|
|0170|31-07-2023 at 19:17|750|
|0270|31-07-2023 at 19:18|742|
|0366|31-07-2023 at 19:20|910|
|0071|31-07-2023 at 19:22|741|
|0667|31-07-2023 at 19:22|747|
|0468|31-07-2023 at 19:23|752|
|0568|31-07-2023 at 19:24|743|
|0171|31-07-2023 at 19:30|744|
|0271|31-07-2023 at 19:30|737|
|0072|31-07-2023 at 19:34|746|
|0668|31-07-2023 at 19:34|741|
|0367|31-07-2023 at 19:35|912|
|0469|31-07-2023 at 19:35|746|
|0569|31-07-2023 at 19:36|743|
|0172|31-07-2023 at 19:42|752|
|0272|31-07-2023 at 19:42|745|
|0073|31-07-2023 at 19:47|741|
|0669|31-07-2023 at 19:47|747|
|0470|31-07-2023 at 19:48|748|
|0570|31-07-2023 at 19:49|748|
|0368|31-07-2023 at 19:50|908|
|0173|31-07-2023 at 19:55|741|
|0273|31-07-2023 at 19:55|743|
|0074|31-07-2023 at 19:59|744|
|0670|31-07-2023 at 19:59|745|
|0471|31-07-2023 at 20:00|750|
|0571|31-07-2023 at 20:01|744|
|0369|31-07-2023 at 20:05|908|
|0174|31-07-2023 at 20:07|747|
|0274|31-07-2023 at 20:07|742|
|0075|31-07-2023 at 20:11|738|
|0671|31-07-2023 at 20:12|746|
|0472|31-07-2023 at 20:13|750|
|0572|31-07-2023 at 20:13|742|
|0175|31-07-2023 at 20:20|748|
|0275|31-07-2023 at 20:20|748|
|0370|31-07-2023 at 20:20|910|
|0076|31-07-2023 at 20:24|737|
|0672|31-07-2023 at 20:24|737|
|0473|31-07-2023 at 20:25|750|
|0573|31-07-2023 at 20:26|740|
|0176|31-07-2023 at 20:32|747|
|0276|31-07-2023 at 20:32|739|
|0371|31-07-2023 at 20:36|908|
|0077|31-07-2023 at 20:36|744|
|0673|31-07-2023 at 20:36|744|
|0474|31-07-2023 at 20:38|749|
|0574|31-07-2023 at 20:38|744|
|0177|31-07-2023 at 20:44|740|
|0277|31-07-2023 at 20:44|738|
|0078|31-07-2023 at 20:48|743|
|0674|31-07-2023 at 20:49|735|
|0475|31-07-2023 at 20:50|745|
|0575|31-07-2023 at 20:51|746|
|0372|31-07-2023 at 20:51|911|
|0278|31-07-2023 at 20:57|739|
|0178|31-07-2023 at 20:57|750|
|0079|31-07-2023 at 21:01|738|
|0675|31-07-2023 at 21:01|739|
|0476|31-07-2023 at 21:03|753|
|0576|31-07-2023 at 21:03|744|
|0373|31-07-2023 at 21:06|913|
|0279|31-07-2023 at 21:09|744|
|0179|31-07-2023 at 21:09|744|
|0080|31-07-2023 at 21:13|744|
|0676|31-07-2023 at 21:13|747|
|0477|31-07-2023 at 21:15|749|
|0577|31-07-2023 at 21:15|744|
|0374|31-07-2023 at 21:21|913|
|0280|31-07-2023 at 21:21|741|
|0180|31-07-2023 at 21:22|744|
|0081|31-07-2023 at 21:25|742|
|0677|31-07-2023 at 21:26|738|
|0478|31-07-2023 at 21:28|751|
|0578|31-07-2023 at 21:28|742|
|0281|31-07-2023 at 21:34|741|
|0181|31-07-2023 at 21:34|743|
|0375|31-07-2023 at 21:36|910|
|0082|31-07-2023 at 21:38|752|
|0678|31-07-2023 at 21:38|739|
|0479|31-07-2023 at 21:40|746|
|0579|31-07-2023 at 21:40|745|
|0282|31-07-2023 at 21:46|742|
|0182|31-07-2023 at 21:46|746|
|0083|31-07-2023 at 21:50|740|
|0679|31-07-2023 at 21:50|743|
|0376|31-07-2023 at 21:51|907|
|0480|31-07-2023 at 21:52|743|
|0580|31-07-2023 at 21:52|740|
|0283|31-07-2023 at 21:58|741|
|0183|31-07-2023 at 21:59|743|
|0084|31-07-2023 at 22:03|741|
|0680|31-07-2023 at 22:03|743|
|0581|31-07-2023 at 22:05|744|
|0481|31-07-2023 at 22:05|755|
|0377|31-07-2023 at 22:07|911|
|0284|31-07-2023 at 22:11|739|
|0184|31-07-2023 at 22:11|743|
|0085|31-07-2023 at 22:15|741|
|0681|31-07-2023 at 22:15|743|
|0582|31-07-2023 at 22:17|747|
|0482|31-07-2023 at 22:17|749|
|0378|31-07-2023 at 22:22|911|
|0285|31-07-2023 at 22:23|744|
|0185|31-07-2023 at 22:24|745|
|0086|31-07-2023 at 22:27|739|
|0682|31-07-2023 at 22:27|737|
|0583|31-07-2023 at 22:30|743|
|0483|31-07-2023 at 22:30|746|
|0286|31-07-2023 at 22:35|739|
|0186|31-07-2023 at 22:36|747|
|0379|31-07-2023 at 22:37|906|
|0683|31-07-2023 at 22:40|737|
|0087|31-07-2023 at 22:40|747|
|0584|31-07-2023 at 22:42|744|
|0484|31-07-2023 at 22:42|753|
