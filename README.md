# Summer-Sandbox23
> Exploration of the N-body simulations made by *gevolution* and how to best treat the data in order to feed it into a convolutional neural network binary classifier in order to distinguish relativistic and newtonian simulations.

# Content

## [log](/log/)
This [folder](/log) contains the log of the work, among other relevant `.md` files. 
* [dbd.md](/log/dbd.md) contains short descriptions of the day-by-day work. 
* [questions.md](/log/questions.md), self-explanatory.
* [todo.md](/log/todo.md), self-explanatory.

## [ML](/ML/)
This folder contains the Machine Learning part of project and is the main folder. For its subdivision see below. The idea is place the script used for execution in the [scripts](/ML/scripts/) folder and all the source code in the [src](/ML/src) folder. This directory also contains a [todo](/ML/todos.md) of the *###TODO*s found in the source codes. The requirements are found in [requirements.txt](/ML/requirements.txt).

### [scripts](/ML/scripts/)
Scripts able to perform the major executions of the project (not ready for a while, need to sort out data treatment first)
### [src/data](/ML/src/data/)
The data files itself (output from *gevolution*) is (or will be for large simulations) stored at UiO and need to be accessed. This file contains the data processing pipeline from the `.h5`-files provided by *gevolution* to a fully functioning dataset. More info [here](/ML/src/data/README.md).

### [src/network](/ML/src/network/)
Contains the source code the neural network and other machine learning software. Not yet optimised nor generalised in any way. A lot of hard coding present.

## [ptpg](/ptpg/)
PyTorchPlayGround $\to$ basically just a folder of testing material. Format is mostly `.ipynb` for testing and visualisation. No longer in use. Some dummy datasets and small *gevolution* datasets are also located here. Will perhaps be cleaned at some point.  

## Something

## Tree
    .
    ├── data_generation
    │   ├── func_lib.sh
    │   ├── generate_seeds.sh
    │   ├── initialisations
    │   │   ├── log_ini.txt
    │   │   └── tmp.ini
    │   ├── README.md
    │   ├── seeds.txt
    │   ├── simulate.sh
    │   ├── simulations_run.txt
    │   └── test_seeds.txt
    ├── log
    │   ├── dbd.md
    │   ├── HowToMarkdown.md
    │   ├── ideas.md
    │   ├── important.md
    │   ├── mlstuff.md
    │   ├── questions.md
    │   └── todo.md
    ├── ML
    │   ├── findTODOS.sh
    │   ├── requirements.txt
    │   ├── src
    │   │   ├── data
    │   │   │   ├── cubehandler.py
    │   │   │   ├── DataHandler.py
    │   │   │   ├── h5collection.py
    │   │   │   ├── h5cube.py
    │   │   │   ├── h5dataset.py
    │   │   │   ├── __init__.py
    │   │   │   ├── __pycache__
    │   │   │   │   ├── h5collection.cpython-38.pyc
    │   │   │   │   ├── h5cube.cpython-38.pyc
    │   │   │   │   ├── h5dataset.cpython-38.pyc
    │   │   │   │   └── __init__.cpython-38.pyc
    │   │   │   └── README.md
    │   │   ├── __init__.py
    │   │   ├── mainTest.py
    │   │   ├── network
    │   │   │   ├── COW.py
    │   │   │   ├── __init__.py
    │   │   │   ├── model.py
    │   │   │   ├── network.py
    │   │   │   └── __pycache__
    │   │   │       ├── COW.cpython-38.pyc
    │   │   │       ├── __init__.cpython-38.pyc
    │   │   │       ├── model.cpython-38.pyc
    │   │   │       └── network.cpython-38.pyc
    │   │   ├── powerspectra.py
    │   │   └── __pycache__
    │   │       ├── COW.cpython-38.pyc
    │   │       ├── DataHandler.cpython-38.pyc
    │   │       ├── h5pySTUFF.cpython-38.pyc
    │   │       ├── MLutils.cpython-36.pyc
    │   │       ├── MLutils.cpython-38.pyc
    │   │       ├── model.cpython-38.pyc
    │   │       └── network.cpython-38.pyc
    │   └── todos.md
    ├── ptpg
    │   ├── cnn.ipynb
    │   ├── data
    │   │   ├── FashionMNIST
    │   │   │   └── raw
    │   │   │       ├── t10k-images-idx3-ubyte
    │   │   │       ├── t10k-images-idx3-ubyte.gz
    │   │   │       ├── t10k-labels-idx1-ubyte
    │   │   │       ├── t10k-labels-idx1-ubyte.gz
    │   │   │       ├── train-images-idx3-ubyte
    │   │   │       ├── train-images-idx3-ubyte.gz
    │   │   │       ├── train-labels-idx1-ubyte
    │   │   │       └── train-labels-idx1-ubyte.gz
    │   │   ├── lcdm_snap000_B.h5
    │   │   ├── lcdm_snap000_phi.h5
    │   │   ├── lcdm_snap001_B.h5
    │   │   ├── lcdm_snap001_phi.h5
    │   │   ├── lcdm_snap002_B.h5
    │   │   ├── lcdm_snap002_phi.h5
    │   │   ├── lcdm_snap003_B.h5
    │   │   └── lcdm_snap003_phi.h5
    │   ├── introduction.ipynb
    │   ├── readme.md
    │   ├── testing_data.ipynb
    │   └── test.py
    ├── README.md
    └── readmeupdate.sh
    
    14 directories, 71 files
Updated on 2023-07-29
## Subheading about something
sometext testing

