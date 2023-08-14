# Day-by-day log

## Monday 19.06.23
Getting familiar with gevolution, attempting to make simple test runs in order to analyse the data output. Exploring PyTorch, and set up the necessary SSH connections (not done with this yet). 

## Tuesday 20.06.23
Work on skeleton ML pipeline. Work with output from gevolution, but primarily focus on the ML part today.  

## Wednesday 21.06.23
Worked on some elementary ML theory. Looked at the first successfull output from gevolution, visualised this. 

## Thursday 22.06.23
Look more specifically on the outputs of gevolution runs, how to tweak settings.
### Meeting with Paco and Julian
* Generate dummy data with default values from gevolution. Slice it into different slabs and generate dataset. Create model based on this.

## Friday 23.06.23
* Managed to run two simulations with and without GR for small cube 64x64 length 320 Mpc
* Investigated data (see ptpg/testing_data.ipynb)
* Almost done with creating a dataloader object, then a simple ML pipeline can follow in order to see whether this works. 


## Monday 26.06.23
* Continued to build dataset and dataloader objects of the dummy data from two small slices of gr and newtonian. 
* How to best normalise the data? Zero centred and unit variance?
* Have pipeline, generalising (further also)
* Run the longer simulations, and look at architecture.
* Need to understand more theory.

## Tuesday 27.06.23 
* Meetings with Julian and Daivd, getting clearity about what the project entails and possible end goals.
* Linked CLASS with gevolutino, run with slightly larger cubes (128, 128, 128) box dith dims 5120 Mpc. 

## Wednesday 28.06.23
* Started with generalising the data treatment, now with larger .h5 files. Need a pipeline to generate a dataloader from potentially more datacubes. 

## Thursday 29.06.23
* Finalised the SSH login remote on the UiO machines to better work from home. 
* Worked on the data treatment from `.h5`-files to a pytorch Dataloader object.
* Decide on some folder/project structure for the main

## Friday 30.06.23
* Brushed up the main `README.md` file so that the repository is better documented. Still need to be refined and cleaned in a lot of places, but works well for now. 
* Focusing on streamlining the data treatment process. Needs to be generalised for a folder with arbitrary many snapshots. The only requirements must be that gr and newtonian snapshots are kept in two separate folders. Should be able to handle snapshots taken at different redshifts and distinguish between snapshots when several redshifts are provided. 
* Next: Update cube handler object to handle several h5 cubes.


## Tuesday 25.07.23
* Wrote and brushed up the main cube, collection and dataset files, based on the tentative data pipeline.
* Missing: Still some optimalisation to make it work with arbitrary number of cubes etc.
* Missing indexing in the dataloader. Need a way to convert from single-dim-index to various multi-dim-indices.

## Wednesday 26.07.23
* Finished the tentative data pipeline. Things seems to work, neeed to write some documentation and tests.
* Started looking at the networks, and how to best implement them for later use, i.e. when the shape of the test simulation cubes are changing. 
* Architecture of network?

## Thursday 27.07.23
* Get network to work with new data pipeline.
* Returning question of architecture, should read more perhaps?

## Friday 28.07.23
* Fixed issue with main ssh connection from personal computer to uio server. 
* Set up and ran new simulations. Gevolution with Ngrid=(128, 256, 512) with boxsize = 5120 Mpc and tiling factor = Ngrid/4

## Saturday 29.07.23
* Working on an automated way of generating the data sets:
    * Create .ini files with correct parameters, but varying seeds (both for gr and newton)
    * Execute the simulations in correct order, making sure the output is put in the right directory. 
* Managed to create some data generation pipeline. Testing this for seeds 1,2,3 in order. If successfull we need to check the initiation files and be sure they are correct, then run 1000 simulations. 


## Sunday 30.07.23
* Fixed an issue with the simulations stopping after one run, the loop exited. Wrote new loop and things seem to work. 
* Added functionality to delete seed information from the data_generation folder.
* Started a run of 200 simulations.
* Fixed the automatic pushing to github, setting it to every 15 minutes. Can be changed manually.

## Monday 31.07.23
* Simulations works, consulted Julian and made some changes to the initial tmp.ini file. Made some test runs during the night. 
* Reading theory about how the gevolution code works.

## Tuesday 01.08.23
* Got som feedback on tmp.ini. Will probably make some changes. 
* Re-investigates the time it takes to execute 256 and 512 runs on different UiO-nodes. 
* Have now run from seed 0000-0749, 750 simulations in total. Will run seed 1000-1999 with additional output and matter power spectra. THe distribution of work will be as follows:
    |seeds|node|
    |-----|----|
    |1000-1249| hyades7|
    |1250-1499| hyades9|
    |1500-1749| hyades10|
    |1750-1999| hyades3|


NB: Forgot to write up daily summaries, but the overall work done in this period is included in `weekly_updates/README.md`.
