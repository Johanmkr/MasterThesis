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


