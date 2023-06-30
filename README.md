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
PyTorchPlayGround $\to$ basically just a folder of testing material. Format is mostly `.ipynb` for testing and visualisation. No longer in use. 

## Something

## Tree
    .
    ├── log
    │   ├── dbd.md
    │   ├── HowToMarkdown.md
    │   ├── important.md
    │   ├── mlstuff.md
    │   ├── questions.md
    │   └── todo.md
    ├── ML
    │   ├── findTODOS.sh
    │   ├── requirements.txt
    │   ├── scripts
    │   │   └── mainTest.py
    │   ├── src
    │   │   ├── COW.py
    │   │   ├── data
    │   │   │   ├── DataHandler.py
    │   │   │   ├── h5pySTUFF.py
    │   │   │   ├── __init__.py
    │   │   │   └── README.md
    │   │   ├── MLutils.py
    │   │   ├── model.py
    │   │   ├── network
    │   │   ├── network.py
    │   │   └── __pycache__
    │   │       ├── COW.cpython-310.pyc
    │   │       ├── DataHandler.cpython-310.pyc
    │   │       ├── h5pySTUFF.cpython-310.pyc
    │   │       ├── MLutils.cpython-310.pyc
    │   │       ├── model.cpython-310.pyc
    │   │       └── network.cpython-310.pyc
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
    
    11 directories, 47 files
Updated on 2023-06-30
## Subheading about something
sometext testing

