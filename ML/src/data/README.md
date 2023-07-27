# Explanation of the tentative data pipeline

The data consist of snapshot-outputs from the `gevolution` code, which consist of 3-dimensional data cubes in the `.h5` format. 

Thought steps in the pipeline

 1) **h5cube.Cube** Object of single cube. Read one file with `.h5` ending and create object with the data cube as main attribute, but also options to attribute redshift and gr/newton flags and other relevant information to the object.

 2) **h5collection.Collection** Object with a collection of **Cube** objects, generated when a path to a directory containing several `.h5` files is provided. The **Cube** objects need to be initialised with the correct gr/newton flag and redshift. Should find all `.h5` recursively from a starting folder. 

3) **h5dataset.DatasetOfCubes** Create a dataset object of **Collection** objects. 

4) $\longrightarrow$ DataLoader object for training the neural networks.

# Pipeline

gevolution $\to$ `.h5`-files $\to$ `h5cube.Cube` object $\to$ `h5collection.Collection` object $\to$ `DatasetOfCubes`object to be interpreted by a network.