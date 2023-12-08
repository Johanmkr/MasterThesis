# Questions/discussion points to adress supervisors

## To David:

## To Paco

* Architectures
    * Number of layers (are my two models okay?)
    * Activation: ReLU or LeakyReLU?
    * Dropout/MaxPooling: .25 or .5 or any? Dropout on input layer to sample subset of data?
    * Same on all layers?
    * Bias/BatchNorm?
    * Capacity - which layer_param value to use basically
    * Optimizer -> just go with ADAM?
    * Loss function -> Binary cross-entropy should be fine. 
    * Weight initialisation?

* Data
    * Plan on using 1750 of the seeds for training, testing and validation, and then have 250 seeds remaining for inference and analysis: (good/bad?)
    * Normalisation. Zero mean, unit variance across the entire dataset. 
    * Which redshift to train for?
    * Train, test, val split (now .7, .2, .1)
    * Shuffling -> taken care of by Dataloader
    

* Training
    * Train both models, which one is the most interesting?
    * Plan on doing distributed parallel training on the two GPUs I have access to. 


## To Julian

## To Farbod 
[] - Why is the matter power spectra from CLASS different? It appears that the power spectrum file output (inferred from gravitational potential) does not differ much between synchronous and newtonian gauge. Is this because it is inferred from the potential?
[] - Gauge stuff, explanantion about how power spectra in the different gauges appear. 