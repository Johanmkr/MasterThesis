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
    │   ├── clean_simulation.sh
    │   ├── func_lib.sh
    │   ├── generate_seeds.sh
    │   ├── git_upload.sh
    │   ├── initialisations
    │   │   ├── log_ini.txt
    │   │   ├── log.txt
    │   │   ├── seed0000
    │   │   │   ├── seed0000gr.ini
    │   │   │   └── seed0000newton.ini
    │   │   ├── seed0001
    │   │   │   ├── seed0001gr.ini
    │   │   │   └── seed0001newton.ini
    │   │   ├── seed0002
    │   │   │   ├── seed0002gr.ini
    │   │   │   └── seed0002newton.ini
    │   │   ├── seed0003
    │   │   │   ├── seed0003gr.ini
    │   │   │   └── seed0003newton.ini
    │   │   ├── seed0004
    │   │   │   ├── seed0004gr.ini
    │   │   │   └── seed0004newton.ini
    │   │   ├── seed0005
    │   │   │   ├── seed0005gr.ini
    │   │   │   └── seed0005newton.ini
    │   │   ├── seed0006
    │   │   │   ├── seed0006gr.ini
    │   │   │   └── seed0006newton.ini
    │   │   ├── seed0007
    │   │   │   ├── seed0007gr.ini
    │   │   │   └── seed0007newton.ini
    │   │   ├── seed0008
    │   │   │   ├── seed0008gr.ini
    │   │   │   └── seed0008newton.ini
    │   │   ├── seed0009
    │   │   │   ├── seed0009gr.ini
    │   │   │   └── seed0009newton.ini
    │   │   ├── seed0010
    │   │   │   ├── seed0010gr.ini
    │   │   │   └── seed0010newton.ini
    │   │   ├── seed0011
    │   │   │   ├── seed0011gr.ini
    │   │   │   └── seed0011newton.ini
    │   │   ├── seed0012
    │   │   │   ├── seed0012gr.ini
    │   │   │   └── seed0012newton.ini
    │   │   ├── seed0013
    │   │   │   ├── seed0013gr.ini
    │   │   │   └── seed0013newton.ini
    │   │   ├── seed0014
    │   │   │   ├── seed0014gr.ini
    │   │   │   └── seed0014newton.ini
    │   │   ├── seed0015
    │   │   │   ├── seed0015gr.ini
    │   │   │   └── seed0015newton.ini
    │   │   ├── seed0016
    │   │   │   ├── seed0016gr.ini
    │   │   │   └── seed0016newton.ini
    │   │   ├── seed0017
    │   │   │   ├── seed0017gr.ini
    │   │   │   └── seed0017newton.ini
    │   │   ├── seed0018
    │   │   │   ├── seed0018gr.ini
    │   │   │   └── seed0018newton.ini
    │   │   ├── seed0019
    │   │   │   ├── seed0019gr.ini
    │   │   │   └── seed0019newton.ini
    │   │   ├── seed0020
    │   │   │   ├── seed0020gr.ini
    │   │   │   └── seed0020newton.ini
    │   │   ├── seed0021
    │   │   │   ├── seed0021gr.ini
    │   │   │   └── seed0021newton.ini
    │   │   ├── seed0022
    │   │   │   ├── seed0022gr.ini
    │   │   │   └── seed0022newton.ini
    │   │   ├── seed0023
    │   │   │   ├── seed0023gr.ini
    │   │   │   └── seed0023newton.ini
    │   │   ├── seed0024
    │   │   │   ├── seed0024gr.ini
    │   │   │   └── seed0024newton.ini
    │   │   ├── seed0025
    │   │   │   ├── seed0025gr.ini
    │   │   │   └── seed0025newton.ini
    │   │   ├── seed0026
    │   │   │   ├── seed0026gr.ini
    │   │   │   └── seed0026newton.ini
    │   │   ├── seed0027
    │   │   │   ├── seed0027gr.ini
    │   │   │   └── seed0027newton.ini
    │   │   ├── seed0028
    │   │   │   ├── seed0028gr.ini
    │   │   │   └── seed0028newton.ini
    │   │   ├── seed0029
    │   │   │   ├── seed0029gr.ini
    │   │   │   └── seed0029newton.ini
    │   │   ├── seed0030
    │   │   │   ├── seed0030gr.ini
    │   │   │   └── seed0030newton.ini
    │   │   ├── seed0031
    │   │   │   ├── seed0031gr.ini
    │   │   │   └── seed0031newton.ini
    │   │   ├── seed0032
    │   │   │   ├── seed0032gr.ini
    │   │   │   └── seed0032newton.ini
    │   │   ├── seed0033
    │   │   │   ├── seed0033gr.ini
    │   │   │   └── seed0033newton.ini
    │   │   ├── seed0034
    │   │   │   ├── seed0034gr.ini
    │   │   │   └── seed0034newton.ini
    │   │   ├── seed0035
    │   │   │   ├── seed0035gr.ini
    │   │   │   └── seed0035newton.ini
    │   │   ├── seed0036
    │   │   │   ├── seed0036gr.ini
    │   │   │   └── seed0036newton.ini
    │   │   ├── seed0037
    │   │   │   ├── seed0037gr.ini
    │   │   │   └── seed0037newton.ini
    │   │   ├── seed0038
    │   │   │   ├── seed0038gr.ini
    │   │   │   └── seed0038newton.ini
    │   │   ├── seed0039
    │   │   │   ├── seed0039gr.ini
    │   │   │   └── seed0039newton.ini
    │   │   ├── seed0040
    │   │   │   ├── seed0040gr.ini
    │   │   │   └── seed0040newton.ini
    │   │   ├── seed0041
    │   │   │   ├── seed0041gr.ini
    │   │   │   └── seed0041newton.ini
    │   │   ├── seed0042
    │   │   │   ├── seed0042gr.ini
    │   │   │   └── seed0042newton.ini
    │   │   ├── seed0043
    │   │   │   ├── seed0043gr.ini
    │   │   │   └── seed0043newton.ini
    │   │   ├── seed0044
    │   │   │   ├── seed0044gr.ini
    │   │   │   └── seed0044newton.ini
    │   │   ├── seed0045
    │   │   │   ├── seed0045gr.ini
    │   │   │   └── seed0045newton.ini
    │   │   ├── seed0046
    │   │   │   ├── seed0046gr.ini
    │   │   │   └── seed0046newton.ini
    │   │   ├── seed0047
    │   │   │   ├── seed0047gr.ini
    │   │   │   └── seed0047newton.ini
    │   │   ├── seed0048
    │   │   │   ├── seed0048gr.ini
    │   │   │   └── seed0048newton.ini
    │   │   ├── seed0049
    │   │   │   ├── seed0049gr.ini
    │   │   │   └── seed0049newton.ini
    │   │   ├── seed0050
    │   │   │   ├── seed0050gr.ini
    │   │   │   └── seed0050newton.ini
    │   │   ├── seed0051
    │   │   │   ├── seed0051gr.ini
    │   │   │   └── seed0051newton.ini
    │   │   ├── seed0052
    │   │   │   ├── seed0052gr.ini
    │   │   │   └── seed0052newton.ini
    │   │   ├── seed0053
    │   │   │   ├── seed0053gr.ini
    │   │   │   └── seed0053newton.ini
    │   │   ├── seed0054
    │   │   │   ├── seed0054gr.ini
    │   │   │   └── seed0054newton.ini
    │   │   ├── seed0055
    │   │   │   ├── seed0055gr.ini
    │   │   │   └── seed0055newton.ini
    │   │   ├── seed0056
    │   │   │   ├── seed0056gr.ini
    │   │   │   └── seed0056newton.ini
    │   │   ├── seed0057
    │   │   │   ├── seed0057gr.ini
    │   │   │   └── seed0057newton.ini
    │   │   ├── seed0058
    │   │   │   ├── seed0058gr.ini
    │   │   │   └── seed0058newton.ini
    │   │   ├── seed0059
    │   │   │   ├── seed0059gr.ini
    │   │   │   └── seed0059newton.ini
    │   │   ├── seed0060
    │   │   │   ├── seed0060gr.ini
    │   │   │   └── seed0060newton.ini
    │   │   ├── seed0061
    │   │   │   ├── seed0061gr.ini
    │   │   │   └── seed0061newton.ini
    │   │   ├── seed0062
    │   │   │   ├── seed0062gr.ini
    │   │   │   └── seed0062newton.ini
    │   │   ├── seed0063
    │   │   │   ├── seed0063gr.ini
    │   │   │   └── seed0063newton.ini
    │   │   ├── seed0064
    │   │   │   ├── seed0064gr.ini
    │   │   │   └── seed0064newton.ini
    │   │   ├── seed0065
    │   │   │   ├── seed0065gr.ini
    │   │   │   └── seed0065newton.ini
    │   │   ├── seed0066
    │   │   │   ├── seed0066gr.ini
    │   │   │   └── seed0066newton.ini
    │   │   ├── seed0067
    │   │   │   ├── seed0067gr.ini
    │   │   │   └── seed0067newton.ini
    │   │   ├── seed0068
    │   │   │   ├── seed0068gr.ini
    │   │   │   └── seed0068newton.ini
    │   │   ├── seed0069
    │   │   │   ├── seed0069gr.ini
    │   │   │   └── seed0069newton.ini
    │   │   ├── seed0070
    │   │   │   ├── seed0070gr.ini
    │   │   │   └── seed0070newton.ini
    │   │   ├── seed0071
    │   │   │   ├── seed0071gr.ini
    │   │   │   └── seed0071newton.ini
    │   │   ├── seed0072
    │   │   │   ├── seed0072gr.ini
    │   │   │   └── seed0072newton.ini
    │   │   ├── seed0073
    │   │   │   ├── seed0073gr.ini
    │   │   │   └── seed0073newton.ini
    │   │   ├── seed0074
    │   │   │   ├── seed0074gr.ini
    │   │   │   └── seed0074newton.ini
    │   │   ├── seed0075
    │   │   │   ├── seed0075gr.ini
    │   │   │   └── seed0075newton.ini
    │   │   ├── seed0076
    │   │   │   ├── seed0076gr.ini
    │   │   │   └── seed0076newton.ini
    │   │   ├── seed0077
    │   │   │   ├── seed0077gr.ini
    │   │   │   └── seed0077newton.ini
    │   │   ├── seed0078
    │   │   │   ├── seed0078gr.ini
    │   │   │   └── seed0078newton.ini
    │   │   ├── seed0079
    │   │   │   ├── seed0079gr.ini
    │   │   │   └── seed0079newton.ini
    │   │   ├── seed0080
    │   │   │   ├── seed0080gr.ini
    │   │   │   └── seed0080newton.ini
    │   │   ├── seed0081
    │   │   │   ├── seed0081gr.ini
    │   │   │   └── seed0081newton.ini
    │   │   ├── seed0082
    │   │   │   ├── seed0082gr.ini
    │   │   │   └── seed0082newton.ini
    │   │   ├── seed0083
    │   │   │   ├── seed0083gr.ini
    │   │   │   └── seed0083newton.ini
    │   │   ├── seed0084
    │   │   │   ├── seed0084gr.ini
    │   │   │   └── seed0084newton.ini
    │   │   ├── seed0085
    │   │   │   ├── seed0085gr.ini
    │   │   │   └── seed0085newton.ini
    │   │   ├── seed0086
    │   │   │   ├── seed0086gr.ini
    │   │   │   └── seed0086newton.ini
    │   │   ├── seed0087
    │   │   │   ├── seed0087gr.ini
    │   │   │   └── seed0087newton.ini
    │   │   ├── seed0088
    │   │   │   ├── seed0088gr.ini
    │   │   │   └── seed0088newton.ini
    │   │   ├── seed0089
    │   │   │   ├── seed0089gr.ini
    │   │   │   └── seed0089newton.ini
    │   │   ├── seed0090
    │   │   │   ├── seed0090gr.ini
    │   │   │   └── seed0090newton.ini
    │   │   ├── seed0091
    │   │   │   ├── seed0091gr.ini
    │   │   │   └── seed0091newton.ini
    │   │   ├── seed0092
    │   │   │   ├── seed0092gr.ini
    │   │   │   └── seed0092newton.ini
    │   │   ├── seed0093
    │   │   │   ├── seed0093gr.ini
    │   │   │   └── seed0093newton.ini
    │   │   ├── seed0094
    │   │   │   ├── seed0094gr.ini
    │   │   │   └── seed0094newton.ini
    │   │   ├── seed0095
    │   │   │   ├── seed0095gr.ini
    │   │   │   └── seed0095newton.ini
    │   │   ├── seed0096
    │   │   │   ├── seed0096gr.ini
    │   │   │   └── seed0096newton.ini
    │   │   ├── seed0097
    │   │   │   ├── seed0097gr.ini
    │   │   │   └── seed0097newton.ini
    │   │   ├── seed0098
    │   │   │   ├── seed0098gr.ini
    │   │   │   └── seed0098newton.ini
    │   │   ├── seed0099
    │   │   │   ├── seed0099gr.ini
    │   │   │   └── seed0099newton.ini
    │   │   ├── seed0100
    │   │   │   ├── seed0100gr.ini
    │   │   │   └── seed0100newton.ini
    │   │   ├── seed0101
    │   │   │   ├── seed0101gr.ini
    │   │   │   └── seed0101newton.ini
    │   │   ├── seed0102
    │   │   │   ├── seed0102gr.ini
    │   │   │   └── seed0102newton.ini
    │   │   ├── seed0103
    │   │   │   ├── seed0103gr.ini
    │   │   │   └── seed0103newton.ini
    │   │   ├── seed0104
    │   │   │   ├── seed0104gr.ini
    │   │   │   └── seed0104newton.ini
    │   │   ├── seed0105
    │   │   │   ├── seed0105gr.ini
    │   │   │   └── seed0105newton.ini
    │   │   ├── seed0106
    │   │   │   ├── seed0106gr.ini
    │   │   │   └── seed0106newton.ini
    │   │   ├── seed0107
    │   │   │   ├── seed0107gr.ini
    │   │   │   └── seed0107newton.ini
    │   │   ├── seed0108
    │   │   │   ├── seed0108gr.ini
    │   │   │   └── seed0108newton.ini
    │   │   ├── seed0109
    │   │   │   ├── seed0109gr.ini
    │   │   │   └── seed0109newton.ini
    │   │   ├── seed0110
    │   │   │   ├── seed0110gr.ini
    │   │   │   └── seed0110newton.ini
    │   │   ├── seed0111
    │   │   │   ├── seed0111gr.ini
    │   │   │   └── seed0111newton.ini
    │   │   ├── seed0112
    │   │   │   ├── seed0112gr.ini
    │   │   │   └── seed0112newton.ini
    │   │   ├── seed0113
    │   │   │   ├── seed0113gr.ini
    │   │   │   └── seed0113newton.ini
    │   │   ├── seed0114
    │   │   │   ├── seed0114gr.ini
    │   │   │   └── seed0114newton.ini
    │   │   ├── seed0115
    │   │   │   ├── seed0115gr.ini
    │   │   │   └── seed0115newton.ini
    │   │   ├── seed0116
    │   │   │   ├── seed0116gr.ini
    │   │   │   └── seed0116newton.ini
    │   │   ├── seed0117
    │   │   │   ├── seed0117gr.ini
    │   │   │   └── seed0117newton.ini
    │   │   ├── seed0118
    │   │   │   ├── seed0118gr.ini
    │   │   │   └── seed0118newton.ini
    │   │   ├── seed0119
    │   │   │   ├── seed0119gr.ini
    │   │   │   └── seed0119newton.ini
    │   │   ├── seed0120
    │   │   │   ├── seed0120gr.ini
    │   │   │   └── seed0120newton.ini
    │   │   ├── seed0121
    │   │   │   ├── seed0121gr.ini
    │   │   │   └── seed0121newton.ini
    │   │   ├── seed0122
    │   │   │   ├── seed0122gr.ini
    │   │   │   └── seed0122newton.ini
    │   │   ├── seed0123
    │   │   │   ├── seed0123gr.ini
    │   │   │   └── seed0123newton.ini
    │   │   ├── seed0124
    │   │   │   ├── seed0124gr.ini
    │   │   │   └── seed0124newton.ini
    │   │   ├── seed0125
    │   │   │   ├── seed0125gr.ini
    │   │   │   └── seed0125newton.ini
    │   │   ├── seed0126
    │   │   │   ├── seed0126gr.ini
    │   │   │   └── seed0126newton.ini
    │   │   ├── seed0127
    │   │   │   ├── seed0127gr.ini
    │   │   │   └── seed0127newton.ini
    │   │   ├── seed0128
    │   │   │   ├── seed0128gr.ini
    │   │   │   └── seed0128newton.ini
    │   │   ├── seed0129
    │   │   │   ├── seed0129gr.ini
    │   │   │   └── seed0129newton.ini
    │   │   ├── seed0130
    │   │   │   ├── seed0130gr.ini
    │   │   │   └── seed0130newton.ini
    │   │   ├── seed0131
    │   │   │   ├── seed0131gr.ini
    │   │   │   └── seed0131newton.ini
    │   │   ├── seed0132
    │   │   │   ├── seed0132gr.ini
    │   │   │   └── seed0132newton.ini
    │   │   ├── seed0133
    │   │   │   ├── seed0133gr.ini
    │   │   │   └── seed0133newton.ini
    │   │   ├── seed0134
    │   │   │   ├── seed0134gr.ini
    │   │   │   └── seed0134newton.ini
    │   │   ├── seed0135
    │   │   │   ├── seed0135gr.ini
    │   │   │   └── seed0135newton.ini
    │   │   ├── seed0136
    │   │   │   ├── seed0136gr.ini
    │   │   │   └── seed0136newton.ini
    │   │   ├── seed0137
    │   │   │   ├── seed0137gr.ini
    │   │   │   └── seed0137newton.ini
    │   │   ├── seed0138
    │   │   │   ├── seed0138gr.ini
    │   │   │   └── seed0138newton.ini
    │   │   ├── seed0139
    │   │   │   ├── seed0139gr.ini
    │   │   │   └── seed0139newton.ini
    │   │   ├── seed0140
    │   │   │   ├── seed0140gr.ini
    │   │   │   └── seed0140newton.ini
    │   │   ├── seed0141
    │   │   │   ├── seed0141gr.ini
    │   │   │   └── seed0141newton.ini
    │   │   ├── seed0142
    │   │   │   ├── seed0142gr.ini
    │   │   │   └── seed0142newton.ini
    │   │   ├── seed0143
    │   │   │   ├── seed0143gr.ini
    │   │   │   └── seed0143newton.ini
    │   │   ├── seed0144
    │   │   │   ├── seed0144gr.ini
    │   │   │   └── seed0144newton.ini
    │   │   ├── seed0145
    │   │   │   ├── seed0145gr.ini
    │   │   │   └── seed0145newton.ini
    │   │   ├── seed0146
    │   │   │   ├── seed0146gr.ini
    │   │   │   └── seed0146newton.ini
    │   │   ├── seed0147
    │   │   │   ├── seed0147gr.ini
    │   │   │   └── seed0147newton.ini
    │   │   ├── seed0148
    │   │   │   ├── seed0148gr.ini
    │   │   │   └── seed0148newton.ini
    │   │   ├── seed0149
    │   │   │   ├── seed0149gr.ini
    │   │   │   └── seed0149newton.ini
    │   │   ├── seed0150
    │   │   │   ├── seed0150gr.ini
    │   │   │   └── seed0150newton.ini
    │   │   ├── seed0151
    │   │   │   ├── seed0151gr.ini
    │   │   │   └── seed0151newton.ini
    │   │   ├── seed0152
    │   │   │   ├── seed0152gr.ini
    │   │   │   └── seed0152newton.ini
    │   │   ├── seed0153
    │   │   │   ├── seed0153gr.ini
    │   │   │   └── seed0153newton.ini
    │   │   ├── seed0154
    │   │   │   ├── seed0154gr.ini
    │   │   │   └── seed0154newton.ini
    │   │   ├── seed0155
    │   │   │   ├── seed0155gr.ini
    │   │   │   └── seed0155newton.ini
    │   │   ├── seed0156
    │   │   │   ├── seed0156gr.ini
    │   │   │   └── seed0156newton.ini
    │   │   ├── seed0157
    │   │   │   ├── seed0157gr.ini
    │   │   │   └── seed0157newton.ini
    │   │   ├── seed0158
    │   │   │   ├── seed0158gr.ini
    │   │   │   └── seed0158newton.ini
    │   │   ├── seed0159
    │   │   │   ├── seed0159gr.ini
    │   │   │   └── seed0159newton.ini
    │   │   ├── seed0160
    │   │   │   ├── seed0160gr.ini
    │   │   │   └── seed0160newton.ini
    │   │   ├── seed0161
    │   │   │   ├── seed0161gr.ini
    │   │   │   └── seed0161newton.ini
    │   │   ├── seed0162
    │   │   │   ├── seed0162gr.ini
    │   │   │   └── seed0162newton.ini
    │   │   ├── seed0163
    │   │   │   ├── seed0163gr.ini
    │   │   │   └── seed0163newton.ini
    │   │   ├── seed0164
    │   │   │   ├── seed0164gr.ini
    │   │   │   └── seed0164newton.ini
    │   │   ├── seed0165
    │   │   │   ├── seed0165gr.ini
    │   │   │   └── seed0165newton.ini
    │   │   ├── seed0166
    │   │   │   ├── seed0166gr.ini
    │   │   │   └── seed0166newton.ini
    │   │   ├── seed0167
    │   │   │   ├── seed0167gr.ini
    │   │   │   └── seed0167newton.ini
    │   │   ├── seed0168
    │   │   │   ├── seed0168gr.ini
    │   │   │   └── seed0168newton.ini
    │   │   ├── seed0169
    │   │   │   ├── seed0169gr.ini
    │   │   │   └── seed0169newton.ini
    │   │   ├── seed0170
    │   │   │   ├── seed0170gr.ini
    │   │   │   └── seed0170newton.ini
    │   │   ├── seed0171
    │   │   │   ├── seed0171gr.ini
    │   │   │   └── seed0171newton.ini
    │   │   ├── seed0172
    │   │   │   ├── seed0172gr.ini
    │   │   │   └── seed0172newton.ini
    │   │   ├── seed0173
    │   │   │   ├── seed0173gr.ini
    │   │   │   └── seed0173newton.ini
    │   │   ├── seed0174
    │   │   │   ├── seed0174gr.ini
    │   │   │   └── seed0174newton.ini
    │   │   ├── seed0175
    │   │   │   ├── seed0175gr.ini
    │   │   │   └── seed0175newton.ini
    │   │   ├── seed0176
    │   │   │   ├── seed0176gr.ini
    │   │   │   └── seed0176newton.ini
    │   │   ├── seed0177
    │   │   │   ├── seed0177gr.ini
    │   │   │   └── seed0177newton.ini
    │   │   ├── seed0178
    │   │   │   ├── seed0178gr.ini
    │   │   │   └── seed0178newton.ini
    │   │   ├── seed0179
    │   │   │   ├── seed0179gr.ini
    │   │   │   └── seed0179newton.ini
    │   │   ├── seed0180
    │   │   │   ├── seed0180gr.ini
    │   │   │   └── seed0180newton.ini
    │   │   ├── seed0181
    │   │   │   ├── seed0181gr.ini
    │   │   │   └── seed0181newton.ini
    │   │   ├── seed0182
    │   │   │   ├── seed0182gr.ini
    │   │   │   └── seed0182newton.ini
    │   │   ├── seed0183
    │   │   │   ├── seed0183gr.ini
    │   │   │   └── seed0183newton.ini
    │   │   ├── seed0184
    │   │   │   ├── seed0184gr.ini
    │   │   │   └── seed0184newton.ini
    │   │   ├── seed0185
    │   │   │   ├── seed0185gr.ini
    │   │   │   └── seed0185newton.ini
    │   │   ├── seed0186
    │   │   │   ├── seed0186gr.ini
    │   │   │   └── seed0186newton.ini
    │   │   ├── seed0187
    │   │   │   ├── seed0187gr.ini
    │   │   │   └── seed0187newton.ini
    │   │   ├── seed0188
    │   │   │   ├── seed0188gr.ini
    │   │   │   └── seed0188newton.ini
    │   │   ├── seed0189
    │   │   │   ├── seed0189gr.ini
    │   │   │   └── seed0189newton.ini
    │   │   ├── seed0190
    │   │   │   ├── seed0190gr.ini
    │   │   │   └── seed0190newton.ini
    │   │   ├── seed0191
    │   │   │   ├── seed0191gr.ini
    │   │   │   └── seed0191newton.ini
    │   │   ├── seed0192
    │   │   │   ├── seed0192gr.ini
    │   │   │   └── seed0192newton.ini
    │   │   ├── seed0193
    │   │   │   ├── seed0193gr.ini
    │   │   │   └── seed0193newton.ini
    │   │   ├── seed0194
    │   │   │   ├── seed0194gr.ini
    │   │   │   └── seed0194newton.ini
    │   │   ├── seed0195
    │   │   │   ├── seed0195gr.ini
    │   │   │   └── seed0195newton.ini
    │   │   ├── seed0196
    │   │   │   ├── seed0196gr.ini
    │   │   │   └── seed0196newton.ini
    │   │   ├── seed0197
    │   │   │   ├── seed0197gr.ini
    │   │   │   └── seed0197newton.ini
    │   │   ├── seed0198
    │   │   │   ├── seed0198gr.ini
    │   │   │   └── seed0198newton.ini
    │   │   ├── seed0199
    │   │   │   ├── seed0199gr.ini
    │   │   │   └── seed0199newton.ini
    │   │   ├── seed0200
    │   │   │   ├── seed0200gr.ini
    │   │   │   └── seed0200newton.ini
    │   │   ├── seed0201
    │   │   │   ├── seed0201gr.ini
    │   │   │   └── seed0201newton.ini
    │   │   ├── seed0202
    │   │   │   ├── seed0202gr.ini
    │   │   │   └── seed0202newton.ini
    │   │   ├── seed0203
    │   │   │   ├── seed0203gr.ini
    │   │   │   └── seed0203newton.ini
    │   │   ├── seed0204
    │   │   │   ├── seed0204gr.ini
    │   │   │   └── seed0204newton.ini
    │   │   ├── seed0205
    │   │   │   ├── seed0205gr.ini
    │   │   │   └── seed0205newton.ini
    │   │   ├── seed0206
    │   │   │   ├── seed0206gr.ini
    │   │   │   └── seed0206newton.ini
    │   │   ├── seed0207
    │   │   │   ├── seed0207gr.ini
    │   │   │   └── seed0207newton.ini
    │   │   ├── seed0208
    │   │   │   ├── seed0208gr.ini
    │   │   │   └── seed0208newton.ini
    │   │   ├── seed0209
    │   │   │   ├── seed0209gr.ini
    │   │   │   └── seed0209newton.ini
    │   │   ├── seed0210
    │   │   │   ├── seed0210gr.ini
    │   │   │   └── seed0210newton.ini
    │   │   ├── seed0211
    │   │   │   ├── seed0211gr.ini
    │   │   │   └── seed0211newton.ini
    │   │   ├── seed0212
    │   │   │   ├── seed0212gr.ini
    │   │   │   └── seed0212newton.ini
    │   │   ├── seed0213
    │   │   │   ├── seed0213gr.ini
    │   │   │   └── seed0213newton.ini
    │   │   ├── seed0214
    │   │   │   ├── seed0214gr.ini
    │   │   │   └── seed0214newton.ini
    │   │   ├── seed0215
    │   │   │   ├── seed0215gr.ini
    │   │   │   └── seed0215newton.ini
    │   │   ├── seed0216
    │   │   │   ├── seed0216gr.ini
    │   │   │   └── seed0216newton.ini
    │   │   ├── seed0217
    │   │   │   ├── seed0217gr.ini
    │   │   │   └── seed0217newton.ini
    │   │   ├── seed0218
    │   │   │   ├── seed0218gr.ini
    │   │   │   └── seed0218newton.ini
    │   │   ├── seed0219
    │   │   │   ├── seed0219gr.ini
    │   │   │   └── seed0219newton.ini
    │   │   ├── seed0220
    │   │   │   ├── seed0220gr.ini
    │   │   │   └── seed0220newton.ini
    │   │   ├── seed0221
    │   │   │   ├── seed0221gr.ini
    │   │   │   └── seed0221newton.ini
    │   │   ├── seed0222
    │   │   │   ├── seed0222gr.ini
    │   │   │   └── seed0222newton.ini
    │   │   ├── seed0223
    │   │   │   ├── seed0223gr.ini
    │   │   │   └── seed0223newton.ini
    │   │   ├── seed0224
    │   │   │   ├── seed0224gr.ini
    │   │   │   └── seed0224newton.ini
    │   │   ├── seed0225
    │   │   │   ├── seed0225gr.ini
    │   │   │   └── seed0225newton.ini
    │   │   ├── seed0226
    │   │   │   ├── seed0226gr.ini
    │   │   │   └── seed0226newton.ini
    │   │   ├── seed0227
    │   │   │   ├── seed0227gr.ini
    │   │   │   └── seed0227newton.ini
    │   │   ├── seed0228
    │   │   │   ├── seed0228gr.ini
    │   │   │   └── seed0228newton.ini
    │   │   ├── seed0229
    │   │   │   ├── seed0229gr.ini
    │   │   │   └── seed0229newton.ini
    │   │   ├── seed0230
    │   │   │   ├── seed0230gr.ini
    │   │   │   └── seed0230newton.ini
    │   │   ├── seed0231
    │   │   │   ├── seed0231gr.ini
    │   │   │   └── seed0231newton.ini
    │   │   ├── seed0232
    │   │   │   ├── seed0232gr.ini
    │   │   │   └── seed0232newton.ini
    │   │   ├── seed0233
    │   │   │   ├── seed0233gr.ini
    │   │   │   └── seed0233newton.ini
    │   │   ├── seed0234
    │   │   │   ├── seed0234gr.ini
    │   │   │   └── seed0234newton.ini
    │   │   ├── seed0235
    │   │   │   ├── seed0235gr.ini
    │   │   │   └── seed0235newton.ini
    │   │   ├── seed0236
    │   │   │   ├── seed0236gr.ini
    │   │   │   └── seed0236newton.ini
    │   │   ├── seed0237
    │   │   │   ├── seed0237gr.ini
    │   │   │   └── seed0237newton.ini
    │   │   ├── seed0238
    │   │   │   ├── seed0238gr.ini
    │   │   │   └── seed0238newton.ini
    │   │   ├── seed0239
    │   │   │   ├── seed0239gr.ini
    │   │   │   └── seed0239newton.ini
    │   │   ├── seed0240
    │   │   │   ├── seed0240gr.ini
    │   │   │   └── seed0240newton.ini
    │   │   ├── seed0241
    │   │   │   ├── seed0241gr.ini
    │   │   │   └── seed0241newton.ini
    │   │   ├── seed0242
    │   │   │   ├── seed0242gr.ini
    │   │   │   └── seed0242newton.ini
    │   │   ├── seed0243
    │   │   │   ├── seed0243gr.ini
    │   │   │   └── seed0243newton.ini
    │   │   ├── seed0244
    │   │   │   ├── seed0244gr.ini
    │   │   │   └── seed0244newton.ini
    │   │   ├── seed0245
    │   │   │   ├── seed0245gr.ini
    │   │   │   └── seed0245newton.ini
    │   │   ├── seed0246
    │   │   │   ├── seed0246gr.ini
    │   │   │   └── seed0246newton.ini
    │   │   ├── seed0247
    │   │   │   ├── seed0247gr.ini
    │   │   │   └── seed0247newton.ini
    │   │   ├── seed0248
    │   │   │   ├── seed0248gr.ini
    │   │   │   └── seed0248newton.ini
    │   │   ├── seed0249
    │   │   │   ├── seed0249gr.ini
    │   │   │   └── seed0249newton.ini
    │   │   ├── seed0250
    │   │   │   ├── seed0250gr.ini
    │   │   │   └── seed0250newton.ini
    │   │   ├── seed0251
    │   │   │   ├── seed0251gr.ini
    │   │   │   └── seed0251newton.ini
    │   │   ├── seed0252
    │   │   │   ├── seed0252gr.ini
    │   │   │   └── seed0252newton.ini
    │   │   ├── seed0253
    │   │   │   ├── seed0253gr.ini
    │   │   │   └── seed0253newton.ini
    │   │   ├── seed0254
    │   │   │   ├── seed0254gr.ini
    │   │   │   └── seed0254newton.ini
    │   │   ├── seed0255
    │   │   │   ├── seed0255gr.ini
    │   │   │   └── seed0255newton.ini
    │   │   ├── seed0256
    │   │   │   ├── seed0256gr.ini
    │   │   │   └── seed0256newton.ini
    │   │   ├── seed0257
    │   │   │   ├── seed0257gr.ini
    │   │   │   └── seed0257newton.ini
    │   │   ├── seed0258
    │   │   │   ├── seed0258gr.ini
    │   │   │   └── seed0258newton.ini
    │   │   ├── seed0259
    │   │   │   ├── seed0259gr.ini
    │   │   │   └── seed0259newton.ini
    │   │   ├── seed0260
    │   │   │   ├── seed0260gr.ini
    │   │   │   └── seed0260newton.ini
    │   │   ├── seed0261
    │   │   │   ├── seed0261gr.ini
    │   │   │   └── seed0261newton.ini
    │   │   ├── seed0262
    │   │   │   ├── seed0262gr.ini
    │   │   │   └── seed0262newton.ini
    │   │   ├── seed0263
    │   │   │   ├── seed0263gr.ini
    │   │   │   └── seed0263newton.ini
    │   │   ├── seed0264
    │   │   │   ├── seed0264gr.ini
    │   │   │   └── seed0264newton.ini
    │   │   ├── seed0265
    │   │   │   ├── seed0265gr.ini
    │   │   │   └── seed0265newton.ini
    │   │   ├── seed0266
    │   │   │   ├── seed0266gr.ini
    │   │   │   └── seed0266newton.ini
    │   │   ├── seed0267
    │   │   │   ├── seed0267gr.ini
    │   │   │   └── seed0267newton.ini
    │   │   ├── seed0268
    │   │   │   ├── seed0268gr.ini
    │   │   │   └── seed0268newton.ini
    │   │   ├── seed0269
    │   │   │   ├── seed0269gr.ini
    │   │   │   └── seed0269newton.ini
    │   │   ├── seed0270
    │   │   │   ├── seed0270gr.ini
    │   │   │   └── seed0270newton.ini
    │   │   ├── seed0271
    │   │   │   ├── seed0271gr.ini
    │   │   │   └── seed0271newton.ini
    │   │   ├── seed0272
    │   │   │   ├── seed0272gr.ini
    │   │   │   └── seed0272newton.ini
    │   │   ├── seed0273
    │   │   │   ├── seed0273gr.ini
    │   │   │   └── seed0273newton.ini
    │   │   ├── seed0274
    │   │   │   ├── seed0274gr.ini
    │   │   │   └── seed0274newton.ini
    │   │   ├── seed0275
    │   │   │   ├── seed0275gr.ini
    │   │   │   └── seed0275newton.ini
    │   │   ├── seed0276
    │   │   │   ├── seed0276gr.ini
    │   │   │   └── seed0276newton.ini
    │   │   ├── seed0277
    │   │   │   ├── seed0277gr.ini
    │   │   │   └── seed0277newton.ini
    │   │   ├── seed0278
    │   │   │   ├── seed0278gr.ini
    │   │   │   └── seed0278newton.ini
    │   │   ├── seed0279
    │   │   │   ├── seed0279gr.ini
    │   │   │   └── seed0279newton.ini
    │   │   ├── seed0280
    │   │   │   ├── seed0280gr.ini
    │   │   │   └── seed0280newton.ini
    │   │   ├── seed0281
    │   │   │   ├── seed0281gr.ini
    │   │   │   └── seed0281newton.ini
    │   │   ├── seed0282
    │   │   │   ├── seed0282gr.ini
    │   │   │   └── seed0282newton.ini
    │   │   ├── seed0283
    │   │   │   ├── seed0283gr.ini
    │   │   │   └── seed0283newton.ini
    │   │   ├── seed0284
    │   │   │   ├── seed0284gr.ini
    │   │   │   └── seed0284newton.ini
    │   │   ├── seed0285
    │   │   │   ├── seed0285gr.ini
    │   │   │   └── seed0285newton.ini
    │   │   ├── seed0286
    │   │   │   ├── seed0286gr.ini
    │   │   │   └── seed0286newton.ini
    │   │   ├── seed0287
    │   │   │   ├── seed0287gr.ini
    │   │   │   └── seed0287newton.ini
    │   │   ├── seed0288
    │   │   │   ├── seed0288gr.ini
    │   │   │   └── seed0288newton.ini
    │   │   ├── seed0289
    │   │   │   ├── seed0289gr.ini
    │   │   │   └── seed0289newton.ini
    │   │   ├── seed0290
    │   │   │   ├── seed0290gr.ini
    │   │   │   └── seed0290newton.ini
    │   │   ├── seed0291
    │   │   │   ├── seed0291gr.ini
    │   │   │   └── seed0291newton.ini
    │   │   ├── seed0292
    │   │   │   ├── seed0292gr.ini
    │   │   │   └── seed0292newton.ini
    │   │   ├── seed0293
    │   │   │   ├── seed0293gr.ini
    │   │   │   └── seed0293newton.ini
    │   │   ├── seed0294
    │   │   │   ├── seed0294gr.ini
    │   │   │   └── seed0294newton.ini
    │   │   ├── seed0295
    │   │   │   ├── seed0295gr.ini
    │   │   │   └── seed0295newton.ini
    │   │   ├── seed0296
    │   │   │   ├── seed0296gr.ini
    │   │   │   └── seed0296newton.ini
    │   │   ├── seed0297
    │   │   │   ├── seed0297gr.ini
    │   │   │   └── seed0297newton.ini
    │   │   ├── seed0298
    │   │   │   ├── seed0298gr.ini
    │   │   │   └── seed0298newton.ini
    │   │   ├── seed0299
    │   │   │   ├── seed0299gr.ini
    │   │   │   └── seed0299newton.ini
    │   │   ├── seed0300
    │   │   │   ├── seed0300gr.ini
    │   │   │   └── seed0300newton.ini
    │   │   ├── seed0301
    │   │   │   ├── seed0301gr.ini
    │   │   │   └── seed0301newton.ini
    │   │   ├── seed0302
    │   │   │   ├── seed0302gr.ini
    │   │   │   └── seed0302newton.ini
    │   │   ├── seed0303
    │   │   │   ├── seed0303gr.ini
    │   │   │   └── seed0303newton.ini
    │   │   ├── seed0304
    │   │   │   ├── seed0304gr.ini
    │   │   │   └── seed0304newton.ini
    │   │   ├── seed0305
    │   │   │   ├── seed0305gr.ini
    │   │   │   └── seed0305newton.ini
    │   │   ├── seed0306
    │   │   │   ├── seed0306gr.ini
    │   │   │   └── seed0306newton.ini
    │   │   ├── seed0307
    │   │   │   ├── seed0307gr.ini
    │   │   │   └── seed0307newton.ini
    │   │   ├── seed0308
    │   │   │   ├── seed0308gr.ini
    │   │   │   └── seed0308newton.ini
    │   │   ├── seed0309
    │   │   │   ├── seed0309gr.ini
    │   │   │   └── seed0309newton.ini
    │   │   ├── seed0310
    │   │   │   ├── seed0310gr.ini
    │   │   │   └── seed0310newton.ini
    │   │   ├── seed0311
    │   │   │   ├── seed0311gr.ini
    │   │   │   └── seed0311newton.ini
    │   │   ├── seed0312
    │   │   │   ├── seed0312gr.ini
    │   │   │   └── seed0312newton.ini
    │   │   ├── seed0313
    │   │   │   ├── seed0313gr.ini
    │   │   │   └── seed0313newton.ini
    │   │   ├── seed0314
    │   │   │   ├── seed0314gr.ini
    │   │   │   └── seed0314newton.ini
    │   │   ├── seed0315
    │   │   │   ├── seed0315gr.ini
    │   │   │   └── seed0315newton.ini
    │   │   ├── seed0316
    │   │   │   ├── seed0316gr.ini
    │   │   │   └── seed0316newton.ini
    │   │   ├── seed0317
    │   │   │   ├── seed0317gr.ini
    │   │   │   └── seed0317newton.ini
    │   │   ├── seed0318
    │   │   │   ├── seed0318gr.ini
    │   │   │   └── seed0318newton.ini
    │   │   ├── seed0319
    │   │   │   ├── seed0319gr.ini
    │   │   │   └── seed0319newton.ini
    │   │   ├── seed0320
    │   │   │   ├── seed0320gr.ini
    │   │   │   └── seed0320newton.ini
    │   │   ├── seed0321
    │   │   │   ├── seed0321gr.ini
    │   │   │   └── seed0321newton.ini
    │   │   ├── seed0322
    │   │   │   ├── seed0322gr.ini
    │   │   │   └── seed0322newton.ini
    │   │   ├── seed0323
    │   │   │   ├── seed0323gr.ini
    │   │   │   └── seed0323newton.ini
    │   │   ├── seed0324
    │   │   │   ├── seed0324gr.ini
    │   │   │   └── seed0324newton.ini
    │   │   ├── seed0325
    │   │   │   ├── seed0325gr.ini
    │   │   │   └── seed0325newton.ini
    │   │   ├── seed0326
    │   │   │   ├── seed0326gr.ini
    │   │   │   └── seed0326newton.ini
    │   │   ├── seed0327
    │   │   │   ├── seed0327gr.ini
    │   │   │   └── seed0327newton.ini
    │   │   ├── seed0328
    │   │   │   ├── seed0328gr.ini
    │   │   │   └── seed0328newton.ini
    │   │   ├── seed0329
    │   │   │   ├── seed0329gr.ini
    │   │   │   └── seed0329newton.ini
    │   │   ├── seed0330
    │   │   │   ├── seed0330gr.ini
    │   │   │   └── seed0330newton.ini
    │   │   ├── seed0331
    │   │   │   ├── seed0331gr.ini
    │   │   │   └── seed0331newton.ini
    │   │   ├── seed0332
    │   │   │   ├── seed0332gr.ini
    │   │   │   └── seed0332newton.ini
    │   │   ├── seed0333
    │   │   │   ├── seed0333gr.ini
    │   │   │   └── seed0333newton.ini
    │   │   ├── seed0334
    │   │   │   ├── seed0334gr.ini
    │   │   │   └── seed0334newton.ini
    │   │   ├── seed0335
    │   │   │   ├── seed0335gr.ini
    │   │   │   └── seed0335newton.ini
    │   │   ├── seed0336
    │   │   │   ├── seed0336gr.ini
    │   │   │   └── seed0336newton.ini
    │   │   ├── seed0337
    │   │   │   ├── seed0337gr.ini
    │   │   │   └── seed0337newton.ini
    │   │   ├── seed0338
    │   │   │   ├── seed0338gr.ini
    │   │   │   └── seed0338newton.ini
    │   │   ├── seed0339
    │   │   │   ├── seed0339gr.ini
    │   │   │   └── seed0339newton.ini
    │   │   ├── seed0340
    │   │   │   ├── seed0340gr.ini
    │   │   │   └── seed0340newton.ini
    │   │   ├── seed0341
    │   │   │   ├── seed0341gr.ini
    │   │   │   └── seed0341newton.ini
    │   │   ├── seed0342
    │   │   │   ├── seed0342gr.ini
    │   │   │   └── seed0342newton.ini
    │   │   ├── seed0343
    │   │   │   ├── seed0343gr.ini
    │   │   │   └── seed0343newton.ini
    │   │   ├── seed0344
    │   │   │   ├── seed0344gr.ini
    │   │   │   └── seed0344newton.ini
    │   │   ├── seed0345
    │   │   │   ├── seed0345gr.ini
    │   │   │   └── seed0345newton.ini
    │   │   ├── seed0346
    │   │   │   ├── seed0346gr.ini
    │   │   │   └── seed0346newton.ini
    │   │   ├── seed0347
    │   │   │   ├── seed0347gr.ini
    │   │   │   └── seed0347newton.ini
    │   │   ├── seed0348
    │   │   │   ├── seed0348gr.ini
    │   │   │   └── seed0348newton.ini
    │   │   ├── seed0349
    │   │   │   ├── seed0349gr.ini
    │   │   │   └── seed0349newton.ini
    │   │   ├── seed0350
    │   │   │   ├── seed0350gr.ini
    │   │   │   └── seed0350newton.ini
    │   │   ├── seed0351
    │   │   │   ├── seed0351gr.ini
    │   │   │   └── seed0351newton.ini
    │   │   ├── seed0352
    │   │   │   ├── seed0352gr.ini
    │   │   │   └── seed0352newton.ini
    │   │   ├── seed0353
    │   │   │   ├── seed0353gr.ini
    │   │   │   └── seed0353newton.ini
    │   │   ├── seed0354
    │   │   │   ├── seed0354gr.ini
    │   │   │   └── seed0354newton.ini
    │   │   ├── seed0355
    │   │   │   ├── seed0355gr.ini
    │   │   │   └── seed0355newton.ini
    │   │   ├── seed0356
    │   │   │   ├── seed0356gr.ini
    │   │   │   └── seed0356newton.ini
    │   │   ├── seed0357
    │   │   │   ├── seed0357gr.ini
    │   │   │   └── seed0357newton.ini
    │   │   ├── seed0358
    │   │   │   ├── seed0358gr.ini
    │   │   │   └── seed0358newton.ini
    │   │   ├── seed0359
    │   │   │   ├── seed0359gr.ini
    │   │   │   └── seed0359newton.ini
    │   │   ├── seed0360
    │   │   │   ├── seed0360gr.ini
    │   │   │   └── seed0360newton.ini
    │   │   ├── seed0361
    │   │   │   ├── seed0361gr.ini
    │   │   │   └── seed0361newton.ini
    │   │   ├── seed0362
    │   │   │   ├── seed0362gr.ini
    │   │   │   └── seed0362newton.ini
    │   │   ├── seed0363
    │   │   │   ├── seed0363gr.ini
    │   │   │   └── seed0363newton.ini
    │   │   ├── seed0364
    │   │   │   ├── seed0364gr.ini
    │   │   │   └── seed0364newton.ini
    │   │   ├── seed0365
    │   │   │   ├── seed0365gr.ini
    │   │   │   └── seed0365newton.ini
    │   │   ├── seed0366
    │   │   │   ├── seed0366gr.ini
    │   │   │   └── seed0366newton.ini
    │   │   ├── seed0367
    │   │   │   ├── seed0367gr.ini
    │   │   │   └── seed0367newton.ini
    │   │   ├── seed0368
    │   │   │   ├── seed0368gr.ini
    │   │   │   └── seed0368newton.ini
    │   │   ├── seed0369
    │   │   │   ├── seed0369gr.ini
    │   │   │   └── seed0369newton.ini
    │   │   ├── seed0370
    │   │   │   ├── seed0370gr.ini
    │   │   │   └── seed0370newton.ini
    │   │   ├── seed0371
    │   │   │   ├── seed0371gr.ini
    │   │   │   └── seed0371newton.ini
    │   │   ├── seed0372
    │   │   │   ├── seed0372gr.ini
    │   │   │   └── seed0372newton.ini
    │   │   ├── seed0373
    │   │   │   ├── seed0373gr.ini
    │   │   │   └── seed0373newton.ini
    │   │   ├── seed0374
    │   │   │   ├── seed0374gr.ini
    │   │   │   └── seed0374newton.ini
    │   │   ├── seed0375
    │   │   │   ├── seed0375gr.ini
    │   │   │   └── seed0375newton.ini
    │   │   ├── seed0376
    │   │   │   ├── seed0376gr.ini
    │   │   │   └── seed0376newton.ini
    │   │   ├── seed0377
    │   │   │   ├── seed0377gr.ini
    │   │   │   └── seed0377newton.ini
    │   │   ├── seed0378
    │   │   │   ├── seed0378gr.ini
    │   │   │   └── seed0378newton.ini
    │   │   ├── seed0379
    │   │   │   ├── seed0379gr.ini
    │   │   │   └── seed0379newton.ini
    │   │   ├── seed0380
    │   │   │   ├── seed0380gr.ini
    │   │   │   └── seed0380newton.ini
    │   │   ├── seed0381
    │   │   │   ├── seed0381gr.ini
    │   │   │   └── seed0381newton.ini
    │   │   ├── seed0382
    │   │   │   ├── seed0382gr.ini
    │   │   │   └── seed0382newton.ini
    │   │   ├── seed0383
    │   │   │   ├── seed0383gr.ini
    │   │   │   └── seed0383newton.ini
    │   │   ├── seed0384
    │   │   │   ├── seed0384gr.ini
    │   │   │   └── seed0384newton.ini
    │   │   ├── seed0385
    │   │   │   ├── seed0385gr.ini
    │   │   │   └── seed0385newton.ini
    │   │   ├── seed0386
    │   │   │   ├── seed0386gr.ini
    │   │   │   └── seed0386newton.ini
    │   │   ├── seed0387
    │   │   │   ├── seed0387gr.ini
    │   │   │   └── seed0387newton.ini
    │   │   ├── seed0388
    │   │   │   ├── seed0388gr.ini
    │   │   │   └── seed0388newton.ini
    │   │   ├── seed0389
    │   │   │   ├── seed0389gr.ini
    │   │   │   └── seed0389newton.ini
    │   │   ├── seed0390
    │   │   │   ├── seed0390gr.ini
    │   │   │   └── seed0390newton.ini
    │   │   ├── seed0391
    │   │   │   ├── seed0391gr.ini
    │   │   │   └── seed0391newton.ini
    │   │   ├── seed0392
    │   │   │   ├── seed0392gr.ini
    │   │   │   └── seed0392newton.ini
    │   │   ├── seed0393
    │   │   │   ├── seed0393gr.ini
    │   │   │   └── seed0393newton.ini
    │   │   ├── seed0394
    │   │   │   ├── seed0394gr.ini
    │   │   │   └── seed0394newton.ini
    │   │   ├── seed0395
    │   │   │   ├── seed0395gr.ini
    │   │   │   └── seed0395newton.ini
    │   │   ├── seed0396
    │   │   │   ├── seed0396gr.ini
    │   │   │   └── seed0396newton.ini
    │   │   ├── seed0397
    │   │   │   ├── seed0397gr.ini
    │   │   │   └── seed0397newton.ini
    │   │   ├── seed0398
    │   │   │   ├── seed0398gr.ini
    │   │   │   └── seed0398newton.ini
    │   │   ├── seed0399
    │   │   │   ├── seed0399gr.ini
    │   │   │   └── seed0399newton.ini
    │   │   ├── seed0400
    │   │   │   ├── seed0400gr.ini
    │   │   │   └── seed0400newton.ini
    │   │   ├── seed0401
    │   │   │   ├── seed0401gr.ini
    │   │   │   └── seed0401newton.ini
    │   │   ├── seed0402
    │   │   │   ├── seed0402gr.ini
    │   │   │   └── seed0402newton.ini
    │   │   ├── seed0403
    │   │   │   ├── seed0403gr.ini
    │   │   │   └── seed0403newton.ini
    │   │   ├── seed0404
    │   │   │   ├── seed0404gr.ini
    │   │   │   └── seed0404newton.ini
    │   │   ├── seed0405
    │   │   │   ├── seed0405gr.ini
    │   │   │   └── seed0405newton.ini
    │   │   ├── seed0406
    │   │   │   ├── seed0406gr.ini
    │   │   │   └── seed0406newton.ini
    │   │   ├── seed0407
    │   │   │   ├── seed0407gr.ini
    │   │   │   └── seed0407newton.ini
    │   │   ├── seed0408
    │   │   │   ├── seed0408gr.ini
    │   │   │   └── seed0408newton.ini
    │   │   ├── seed0409
    │   │   │   ├── seed0409gr.ini
    │   │   │   └── seed0409newton.ini
    │   │   ├── seed0410
    │   │   │   ├── seed0410gr.ini
    │   │   │   └── seed0410newton.ini
    │   │   ├── seed0411
    │   │   │   ├── seed0411gr.ini
    │   │   │   └── seed0411newton.ini
    │   │   ├── seed0412
    │   │   │   ├── seed0412gr.ini
    │   │   │   └── seed0412newton.ini
    │   │   ├── seed0413
    │   │   │   ├── seed0413gr.ini
    │   │   │   └── seed0413newton.ini
    │   │   ├── seed0414
    │   │   │   ├── seed0414gr.ini
    │   │   │   └── seed0414newton.ini
    │   │   ├── seed0415
    │   │   │   ├── seed0415gr.ini
    │   │   │   └── seed0415newton.ini
    │   │   ├── seed0416
    │   │   │   ├── seed0416gr.ini
    │   │   │   └── seed0416newton.ini
    │   │   ├── seed0417
    │   │   │   ├── seed0417gr.ini
    │   │   │   └── seed0417newton.ini
    │   │   ├── seed0418
    │   │   │   ├── seed0418gr.ini
    │   │   │   └── seed0418newton.ini
    │   │   ├── seed0419
    │   │   │   ├── seed0419gr.ini
    │   │   │   └── seed0419newton.ini
    │   │   ├── seed0420
    │   │   │   ├── seed0420gr.ini
    │   │   │   └── seed0420newton.ini
    │   │   ├── seed0421
    │   │   │   ├── seed0421gr.ini
    │   │   │   └── seed0421newton.ini
    │   │   ├── seed0422
    │   │   │   ├── seed0422gr.ini
    │   │   │   └── seed0422newton.ini
    │   │   ├── seed0423
    │   │   │   ├── seed0423gr.ini
    │   │   │   └── seed0423newton.ini
    │   │   ├── seed0424
    │   │   │   ├── seed0424gr.ini
    │   │   │   └── seed0424newton.ini
    │   │   ├── seed0425
    │   │   │   ├── seed0425gr.ini
    │   │   │   └── seed0425newton.ini
    │   │   ├── seed0426
    │   │   │   ├── seed0426gr.ini
    │   │   │   └── seed0426newton.ini
    │   │   ├── seed0427
    │   │   │   ├── seed0427gr.ini
    │   │   │   └── seed0427newton.ini
    │   │   ├── seed0428
    │   │   │   ├── seed0428gr.ini
    │   │   │   └── seed0428newton.ini
    │   │   ├── seed0429
    │   │   │   ├── seed0429gr.ini
    │   │   │   └── seed0429newton.ini
    │   │   ├── seed0430
    │   │   │   ├── seed0430gr.ini
    │   │   │   └── seed0430newton.ini
    │   │   ├── seed0431
    │   │   │   ├── seed0431gr.ini
    │   │   │   └── seed0431newton.ini
    │   │   ├── seed0432
    │   │   │   ├── seed0432gr.ini
    │   │   │   └── seed0432newton.ini
    │   │   ├── seed0433
    │   │   │   ├── seed0433gr.ini
    │   │   │   └── seed0433newton.ini
    │   │   ├── seed0434
    │   │   │   ├── seed0434gr.ini
    │   │   │   └── seed0434newton.ini
    │   │   ├── seed0435
    │   │   │   ├── seed0435gr.ini
    │   │   │   └── seed0435newton.ini
    │   │   ├── seed0436
    │   │   │   ├── seed0436gr.ini
    │   │   │   └── seed0436newton.ini
    │   │   ├── seed0437
    │   │   │   ├── seed0437gr.ini
    │   │   │   └── seed0437newton.ini
    │   │   ├── seed0438
    │   │   │   ├── seed0438gr.ini
    │   │   │   └── seed0438newton.ini
    │   │   ├── seed0439
    │   │   │   ├── seed0439gr.ini
    │   │   │   └── seed0439newton.ini
    │   │   ├── seed0440
    │   │   │   ├── seed0440gr.ini
    │   │   │   └── seed0440newton.ini
    │   │   ├── seed0441
    │   │   │   ├── seed0441gr.ini
    │   │   │   └── seed0441newton.ini
    │   │   ├── seed0442
    │   │   │   ├── seed0442gr.ini
    │   │   │   └── seed0442newton.ini
    │   │   ├── seed0443
    │   │   │   ├── seed0443gr.ini
    │   │   │   └── seed0443newton.ini
    │   │   ├── seed0444
    │   │   │   ├── seed0444gr.ini
    │   │   │   └── seed0444newton.ini
    │   │   ├── seed0445
    │   │   │   ├── seed0445gr.ini
    │   │   │   └── seed0445newton.ini
    │   │   ├── seed0446
    │   │   │   ├── seed0446gr.ini
    │   │   │   └── seed0446newton.ini
    │   │   ├── seed0447
    │   │   │   ├── seed0447gr.ini
    │   │   │   └── seed0447newton.ini
    │   │   ├── seed0448
    │   │   │   ├── seed0448gr.ini
    │   │   │   └── seed0448newton.ini
    │   │   ├── seed0449
    │   │   │   ├── seed0449gr.ini
    │   │   │   └── seed0449newton.ini
    │   │   ├── seed0450
    │   │   │   ├── seed0450gr.ini
    │   │   │   └── seed0450newton.ini
    │   │   ├── seed0451
    │   │   │   ├── seed0451gr.ini
    │   │   │   └── seed0451newton.ini
    │   │   ├── seed0452
    │   │   │   ├── seed0452gr.ini
    │   │   │   └── seed0452newton.ini
    │   │   ├── seed0453
    │   │   │   ├── seed0453gr.ini
    │   │   │   └── seed0453newton.ini
    │   │   ├── seed0454
    │   │   │   ├── seed0454gr.ini
    │   │   │   └── seed0454newton.ini
    │   │   ├── seed0455
    │   │   │   ├── seed0455gr.ini
    │   │   │   └── seed0455newton.ini
    │   │   ├── seed0456
    │   │   │   ├── seed0456gr.ini
    │   │   │   └── seed0456newton.ini
    │   │   ├── seed0457
    │   │   │   ├── seed0457gr.ini
    │   │   │   └── seed0457newton.ini
    │   │   ├── seed0458
    │   │   │   ├── seed0458gr.ini
    │   │   │   └── seed0458newton.ini
    │   │   ├── seed0459
    │   │   │   ├── seed0459gr.ini
    │   │   │   └── seed0459newton.ini
    │   │   ├── seed0460
    │   │   │   ├── seed0460gr.ini
    │   │   │   └── seed0460newton.ini
    │   │   ├── seed0461
    │   │   │   ├── seed0461gr.ini
    │   │   │   └── seed0461newton.ini
    │   │   ├── seed0462
    │   │   │   ├── seed0462gr.ini
    │   │   │   └── seed0462newton.ini
    │   │   ├── seed0463
    │   │   │   ├── seed0463gr.ini
    │   │   │   └── seed0463newton.ini
    │   │   ├── seed0464
    │   │   │   ├── seed0464gr.ini
    │   │   │   └── seed0464newton.ini
    │   │   ├── seed0465
    │   │   │   ├── seed0465gr.ini
    │   │   │   └── seed0465newton.ini
    │   │   ├── seed0466
    │   │   │   ├── seed0466gr.ini
    │   │   │   └── seed0466newton.ini
    │   │   ├── seed0467
    │   │   │   ├── seed0467gr.ini
    │   │   │   └── seed0467newton.ini
    │   │   ├── seed0468
    │   │   │   ├── seed0468gr.ini
    │   │   │   └── seed0468newton.ini
    │   │   ├── seed0469
    │   │   │   ├── seed0469gr.ini
    │   │   │   └── seed0469newton.ini
    │   │   ├── seed0470
    │   │   │   ├── seed0470gr.ini
    │   │   │   └── seed0470newton.ini
    │   │   ├── seed0471
    │   │   │   ├── seed0471gr.ini
    │   │   │   └── seed0471newton.ini
    │   │   ├── seed0472
    │   │   │   ├── seed0472gr.ini
    │   │   │   └── seed0472newton.ini
    │   │   ├── seed0473
    │   │   │   ├── seed0473gr.ini
    │   │   │   └── seed0473newton.ini
    │   │   ├── seed0474
    │   │   │   ├── seed0474gr.ini
    │   │   │   └── seed0474newton.ini
    │   │   ├── seed0475
    │   │   │   ├── seed0475gr.ini
    │   │   │   └── seed0475newton.ini
    │   │   ├── seed0476
    │   │   │   ├── seed0476gr.ini
    │   │   │   └── seed0476newton.ini
    │   │   ├── seed0477
    │   │   │   ├── seed0477gr.ini
    │   │   │   └── seed0477newton.ini
    │   │   ├── seed0478
    │   │   │   ├── seed0478gr.ini
    │   │   │   └── seed0478newton.ini
    │   │   ├── seed0479
    │   │   │   ├── seed0479gr.ini
    │   │   │   └── seed0479newton.ini
    │   │   ├── seed0480
    │   │   │   ├── seed0480gr.ini
    │   │   │   └── seed0480newton.ini
    │   │   ├── seed0481
    │   │   │   ├── seed0481gr.ini
    │   │   │   └── seed0481newton.ini
    │   │   ├── seed0482
    │   │   │   ├── seed0482gr.ini
    │   │   │   └── seed0482newton.ini
    │   │   ├── seed0483
    │   │   │   ├── seed0483gr.ini
    │   │   │   └── seed0483newton.ini
    │   │   ├── seed0484
    │   │   │   ├── seed0484gr.ini
    │   │   │   └── seed0484newton.ini
    │   │   ├── seed0485
    │   │   │   ├── seed0485gr.ini
    │   │   │   └── seed0485newton.ini
    │   │   ├── seed0486
    │   │   │   ├── seed0486gr.ini
    │   │   │   └── seed0486newton.ini
    │   │   ├── seed0487
    │   │   │   ├── seed0487gr.ini
    │   │   │   └── seed0487newton.ini
    │   │   ├── seed0488
    │   │   │   ├── seed0488gr.ini
    │   │   │   └── seed0488newton.ini
    │   │   ├── seed0489
    │   │   │   ├── seed0489gr.ini
    │   │   │   └── seed0489newton.ini
    │   │   ├── seed0490
    │   │   │   ├── seed0490gr.ini
    │   │   │   └── seed0490newton.ini
    │   │   ├── seed0491
    │   │   │   ├── seed0491gr.ini
    │   │   │   └── seed0491newton.ini
    │   │   ├── seed0492
    │   │   │   ├── seed0492gr.ini
    │   │   │   └── seed0492newton.ini
    │   │   ├── seed0493
    │   │   │   ├── seed0493gr.ini
    │   │   │   └── seed0493newton.ini
    │   │   ├── seed0494
    │   │   │   ├── seed0494gr.ini
    │   │   │   └── seed0494newton.ini
    │   │   ├── seed0495
    │   │   │   ├── seed0495gr.ini
    │   │   │   └── seed0495newton.ini
    │   │   ├── seed0496
    │   │   │   ├── seed0496gr.ini
    │   │   │   └── seed0496newton.ini
    │   │   ├── seed0497
    │   │   │   ├── seed0497gr.ini
    │   │   │   └── seed0497newton.ini
    │   │   ├── seed0498
    │   │   │   ├── seed0498gr.ini
    │   │   │   └── seed0498newton.ini
    │   │   ├── seed0499
    │   │   │   ├── seed0499gr.ini
    │   │   │   └── seed0499newton.ini
    │   │   ├── seed0500
    │   │   │   ├── seed0500gr.ini
    │   │   │   └── seed0500newton.ini
    │   │   ├── seed0501
    │   │   │   ├── seed0501gr.ini
    │   │   │   └── seed0501newton.ini
    │   │   ├── seed0502
    │   │   │   ├── seed0502gr.ini
    │   │   │   └── seed0502newton.ini
    │   │   ├── seed0503
    │   │   │   ├── seed0503gr.ini
    │   │   │   └── seed0503newton.ini
    │   │   ├── seed0504
    │   │   │   ├── seed0504gr.ini
    │   │   │   └── seed0504newton.ini
    │   │   ├── seed0505
    │   │   │   ├── seed0505gr.ini
    │   │   │   └── seed0505newton.ini
    │   │   ├── seed0506
    │   │   │   ├── seed0506gr.ini
    │   │   │   └── seed0506newton.ini
    │   │   ├── seed0507
    │   │   │   ├── seed0507gr.ini
    │   │   │   └── seed0507newton.ini
    │   │   ├── seed0508
    │   │   │   ├── seed0508gr.ini
    │   │   │   └── seed0508newton.ini
    │   │   ├── seed0509
    │   │   │   ├── seed0509gr.ini
    │   │   │   └── seed0509newton.ini
    │   │   ├── seed0510
    │   │   │   ├── seed0510gr.ini
    │   │   │   └── seed0510newton.ini
    │   │   ├── seed0511
    │   │   │   ├── seed0511gr.ini
    │   │   │   └── seed0511newton.ini
    │   │   ├── seed0512
    │   │   │   ├── seed0512gr.ini
    │   │   │   └── seed0512newton.ini
    │   │   ├── seed0513
    │   │   │   ├── seed0513gr.ini
    │   │   │   └── seed0513newton.ini
    │   │   ├── seed0514
    │   │   │   ├── seed0514gr.ini
    │   │   │   └── seed0514newton.ini
    │   │   ├── seed0515
    │   │   │   ├── seed0515gr.ini
    │   │   │   └── seed0515newton.ini
    │   │   ├── seed0516
    │   │   │   ├── seed0516gr.ini
    │   │   │   └── seed0516newton.ini
    │   │   ├── seed0517
    │   │   │   ├── seed0517gr.ini
    │   │   │   └── seed0517newton.ini
    │   │   ├── seed0518
    │   │   │   ├── seed0518gr.ini
    │   │   │   └── seed0518newton.ini
    │   │   ├── seed0519
    │   │   │   ├── seed0519gr.ini
    │   │   │   └── seed0519newton.ini
    │   │   ├── seed0520
    │   │   │   ├── seed0520gr.ini
    │   │   │   └── seed0520newton.ini
    │   │   ├── seed0521
    │   │   │   ├── seed0521gr.ini
    │   │   │   └── seed0521newton.ini
    │   │   ├── seed0522
    │   │   │   ├── seed0522gr.ini
    │   │   │   └── seed0522newton.ini
    │   │   ├── seed0523
    │   │   │   ├── seed0523gr.ini
    │   │   │   └── seed0523newton.ini
    │   │   ├── seed0524
    │   │   │   ├── seed0524gr.ini
    │   │   │   └── seed0524newton.ini
    │   │   ├── seed0525
    │   │   │   ├── seed0525gr.ini
    │   │   │   └── seed0525newton.ini
    │   │   ├── seed0526
    │   │   │   ├── seed0526gr.ini
    │   │   │   └── seed0526newton.ini
    │   │   ├── seed0527
    │   │   │   ├── seed0527gr.ini
    │   │   │   └── seed0527newton.ini
    │   │   ├── seed0528
    │   │   │   ├── seed0528gr.ini
    │   │   │   └── seed0528newton.ini
    │   │   ├── seed0529
    │   │   │   ├── seed0529gr.ini
    │   │   │   └── seed0529newton.ini
    │   │   ├── seed0530
    │   │   │   ├── seed0530gr.ini
    │   │   │   └── seed0530newton.ini
    │   │   ├── seed0531
    │   │   │   ├── seed0531gr.ini
    │   │   │   └── seed0531newton.ini
    │   │   ├── seed0532
    │   │   │   ├── seed0532gr.ini
    │   │   │   └── seed0532newton.ini
    │   │   ├── seed0533
    │   │   │   ├── seed0533gr.ini
    │   │   │   └── seed0533newton.ini
    │   │   ├── seed0534
    │   │   │   ├── seed0534gr.ini
    │   │   │   └── seed0534newton.ini
    │   │   ├── seed0535
    │   │   │   ├── seed0535gr.ini
    │   │   │   └── seed0535newton.ini
    │   │   ├── seed0536
    │   │   │   ├── seed0536gr.ini
    │   │   │   └── seed0536newton.ini
    │   │   ├── seed0537
    │   │   │   ├── seed0537gr.ini
    │   │   │   └── seed0537newton.ini
    │   │   ├── seed0538
    │   │   │   ├── seed0538gr.ini
    │   │   │   └── seed0538newton.ini
    │   │   ├── seed0539
    │   │   │   ├── seed0539gr.ini
    │   │   │   └── seed0539newton.ini
    │   │   ├── seed0540
    │   │   │   ├── seed0540gr.ini
    │   │   │   └── seed0540newton.ini
    │   │   ├── seed0541
    │   │   │   ├── seed0541gr.ini
    │   │   │   └── seed0541newton.ini
    │   │   ├── seed0542
    │   │   │   ├── seed0542gr.ini
    │   │   │   └── seed0542newton.ini
    │   │   ├── seed0543
    │   │   │   ├── seed0543gr.ini
    │   │   │   └── seed0543newton.ini
    │   │   ├── seed0544
    │   │   │   ├── seed0544gr.ini
    │   │   │   └── seed0544newton.ini
    │   │   ├── seed0545
    │   │   │   ├── seed0545gr.ini
    │   │   │   └── seed0545newton.ini
    │   │   ├── seed0546
    │   │   │   ├── seed0546gr.ini
    │   │   │   └── seed0546newton.ini
    │   │   ├── seed0547
    │   │   │   ├── seed0547gr.ini
    │   │   │   └── seed0547newton.ini
    │   │   ├── seed0548
    │   │   │   ├── seed0548gr.ini
    │   │   │   └── seed0548newton.ini
    │   │   ├── seed0549
    │   │   │   ├── seed0549gr.ini
    │   │   │   └── seed0549newton.ini
    │   │   ├── seed0550
    │   │   │   ├── seed0550gr.ini
    │   │   │   └── seed0550newton.ini
    │   │   ├── seed0551
    │   │   │   ├── seed0551gr.ini
    │   │   │   └── seed0551newton.ini
    │   │   ├── seed0552
    │   │   │   ├── seed0552gr.ini
    │   │   │   └── seed0552newton.ini
    │   │   ├── seed0553
    │   │   │   ├── seed0553gr.ini
    │   │   │   └── seed0553newton.ini
    │   │   ├── seed0554
    │   │   │   ├── seed0554gr.ini
    │   │   │   └── seed0554newton.ini
    │   │   ├── seed0555
    │   │   │   ├── seed0555gr.ini
    │   │   │   └── seed0555newton.ini
    │   │   ├── seed0556
    │   │   │   ├── seed0556gr.ini
    │   │   │   └── seed0556newton.ini
    │   │   ├── seed0557
    │   │   │   ├── seed0557gr.ini
    │   │   │   └── seed0557newton.ini
    │   │   ├── seed0558
    │   │   │   ├── seed0558gr.ini
    │   │   │   └── seed0558newton.ini
    │   │   ├── seed0559
    │   │   │   ├── seed0559gr.ini
    │   │   │   └── seed0559newton.ini
    │   │   ├── seed0560
    │   │   │   ├── seed0560gr.ini
    │   │   │   └── seed0560newton.ini
    │   │   ├── seed0561
    │   │   │   ├── seed0561gr.ini
    │   │   │   └── seed0561newton.ini
    │   │   ├── seed0562
    │   │   │   ├── seed0562gr.ini
    │   │   │   └── seed0562newton.ini
    │   │   ├── seed0563
    │   │   │   ├── seed0563gr.ini
    │   │   │   └── seed0563newton.ini
    │   │   ├── seed0564
    │   │   │   ├── seed0564gr.ini
    │   │   │   └── seed0564newton.ini
    │   │   ├── seed0565
    │   │   │   ├── seed0565gr.ini
    │   │   │   └── seed0565newton.ini
    │   │   ├── seed0566
    │   │   │   ├── seed0566gr.ini
    │   │   │   └── seed0566newton.ini
    │   │   ├── seed0567
    │   │   │   ├── seed0567gr.ini
    │   │   │   └── seed0567newton.ini
    │   │   ├── seed0568
    │   │   │   ├── seed0568gr.ini
    │   │   │   └── seed0568newton.ini
    │   │   ├── seed0569
    │   │   │   ├── seed0569gr.ini
    │   │   │   └── seed0569newton.ini
    │   │   ├── seed0570
    │   │   │   ├── seed0570gr.ini
    │   │   │   └── seed0570newton.ini
    │   │   ├── seed0571
    │   │   │   ├── seed0571gr.ini
    │   │   │   └── seed0571newton.ini
    │   │   ├── seed0572
    │   │   │   ├── seed0572gr.ini
    │   │   │   └── seed0572newton.ini
    │   │   ├── seed0573
    │   │   │   ├── seed0573gr.ini
    │   │   │   └── seed0573newton.ini
    │   │   ├── seed0574
    │   │   │   ├── seed0574gr.ini
    │   │   │   └── seed0574newton.ini
    │   │   ├── seed0575
    │   │   │   ├── seed0575gr.ini
    │   │   │   └── seed0575newton.ini
    │   │   ├── seed0576
    │   │   │   ├── seed0576gr.ini
    │   │   │   └── seed0576newton.ini
    │   │   ├── seed0577
    │   │   │   ├── seed0577gr.ini
    │   │   │   └── seed0577newton.ini
    │   │   ├── seed0578
    │   │   │   ├── seed0578gr.ini
    │   │   │   └── seed0578newton.ini
    │   │   ├── seed0579
    │   │   │   ├── seed0579gr.ini
    │   │   │   └── seed0579newton.ini
    │   │   ├── seed0580
    │   │   │   ├── seed0580gr.ini
    │   │   │   └── seed0580newton.ini
    │   │   ├── seed0581
    │   │   │   ├── seed0581gr.ini
    │   │   │   └── seed0581newton.ini
    │   │   ├── seed0582
    │   │   │   ├── seed0582gr.ini
    │   │   │   └── seed0582newton.ini
    │   │   ├── seed0583
    │   │   │   ├── seed0583gr.ini
    │   │   │   └── seed0583newton.ini
    │   │   ├── seed0584
    │   │   │   ├── seed0584gr.ini
    │   │   │   └── seed0584newton.ini
    │   │   ├── seed0585
    │   │   │   ├── seed0585gr.ini
    │   │   │   └── seed0585newton.ini
    │   │   ├── seed0586
    │   │   │   ├── seed0586gr.ini
    │   │   │   └── seed0586newton.ini
    │   │   ├── seed0587
    │   │   │   ├── seed0587gr.ini
    │   │   │   └── seed0587newton.ini
    │   │   ├── seed0588
    │   │   │   ├── seed0588gr.ini
    │   │   │   └── seed0588newton.ini
    │   │   ├── seed0589
    │   │   │   ├── seed0589gr.ini
    │   │   │   └── seed0589newton.ini
    │   │   ├── seed0590
    │   │   │   ├── seed0590gr.ini
    │   │   │   └── seed0590newton.ini
    │   │   ├── seed0591
    │   │   │   ├── seed0591gr.ini
    │   │   │   └── seed0591newton.ini
    │   │   ├── seed0592
    │   │   │   ├── seed0592gr.ini
    │   │   │   └── seed0592newton.ini
    │   │   ├── seed0593
    │   │   │   ├── seed0593gr.ini
    │   │   │   └── seed0593newton.ini
    │   │   ├── seed0594
    │   │   │   ├── seed0594gr.ini
    │   │   │   └── seed0594newton.ini
    │   │   ├── seed0595
    │   │   │   ├── seed0595gr.ini
    │   │   │   └── seed0595newton.ini
    │   │   ├── seed0596
    │   │   │   ├── seed0596gr.ini
    │   │   │   └── seed0596newton.ini
    │   │   ├── seed0597
    │   │   │   ├── seed0597gr.ini
    │   │   │   └── seed0597newton.ini
    │   │   ├── seed0598
    │   │   │   ├── seed0598gr.ini
    │   │   │   └── seed0598newton.ini
    │   │   ├── seed0599
    │   │   │   ├── seed0599gr.ini
    │   │   │   └── seed0599newton.ini
    │   │   ├── seed0600
    │   │   │   ├── seed0600gr.ini
    │   │   │   └── seed0600newton.ini
    │   │   ├── seed0601
    │   │   │   ├── seed0601gr.ini
    │   │   │   └── seed0601newton.ini
    │   │   ├── seed0602
    │   │   │   ├── seed0602gr.ini
    │   │   │   └── seed0602newton.ini
    │   │   ├── seed0603
    │   │   │   ├── seed0603gr.ini
    │   │   │   └── seed0603newton.ini
    │   │   ├── seed0604
    │   │   │   ├── seed0604gr.ini
    │   │   │   └── seed0604newton.ini
    │   │   ├── seed0605
    │   │   │   ├── seed0605gr.ini
    │   │   │   └── seed0605newton.ini
    │   │   ├── seed0606
    │   │   │   ├── seed0606gr.ini
    │   │   │   └── seed0606newton.ini
    │   │   ├── seed0607
    │   │   │   ├── seed0607gr.ini
    │   │   │   └── seed0607newton.ini
    │   │   ├── seed0608
    │   │   │   ├── seed0608gr.ini
    │   │   │   └── seed0608newton.ini
    │   │   ├── seed0609
    │   │   │   ├── seed0609gr.ini
    │   │   │   └── seed0609newton.ini
    │   │   ├── seed0610
    │   │   │   ├── seed0610gr.ini
    │   │   │   └── seed0610newton.ini
    │   │   ├── seed0611
    │   │   │   ├── seed0611gr.ini
    │   │   │   └── seed0611newton.ini
    │   │   ├── seed0612
    │   │   │   ├── seed0612gr.ini
    │   │   │   └── seed0612newton.ini
    │   │   ├── seed0613
    │   │   │   ├── seed0613gr.ini
    │   │   │   └── seed0613newton.ini
    │   │   ├── seed0614
    │   │   │   ├── seed0614gr.ini
    │   │   │   └── seed0614newton.ini
    │   │   ├── seed0615
    │   │   │   ├── seed0615gr.ini
    │   │   │   └── seed0615newton.ini
    │   │   ├── seed0616
    │   │   │   ├── seed0616gr.ini
    │   │   │   └── seed0616newton.ini
    │   │   ├── seed0617
    │   │   │   ├── seed0617gr.ini
    │   │   │   └── seed0617newton.ini
    │   │   ├── seed0618
    │   │   │   ├── seed0618gr.ini
    │   │   │   └── seed0618newton.ini
    │   │   ├── seed0619
    │   │   │   ├── seed0619gr.ini
    │   │   │   └── seed0619newton.ini
    │   │   ├── seed0620
    │   │   │   ├── seed0620gr.ini
    │   │   │   └── seed0620newton.ini
    │   │   ├── seed0621
    │   │   │   ├── seed0621gr.ini
    │   │   │   └── seed0621newton.ini
    │   │   ├── seed0622
    │   │   │   ├── seed0622gr.ini
    │   │   │   └── seed0622newton.ini
    │   │   ├── seed0623
    │   │   │   ├── seed0623gr.ini
    │   │   │   └── seed0623newton.ini
    │   │   ├── seed0624
    │   │   │   ├── seed0624gr.ini
    │   │   │   └── seed0624newton.ini
    │   │   ├── seed0625
    │   │   │   ├── seed0625gr.ini
    │   │   │   └── seed0625newton.ini
    │   │   ├── seed0626
    │   │   │   ├── seed0626gr.ini
    │   │   │   └── seed0626newton.ini
    │   │   ├── seed0627
    │   │   │   ├── seed0627gr.ini
    │   │   │   └── seed0627newton.ini
    │   │   ├── seed0628
    │   │   │   ├── seed0628gr.ini
    │   │   │   └── seed0628newton.ini
    │   │   ├── seed0629
    │   │   │   ├── seed0629gr.ini
    │   │   │   └── seed0629newton.ini
    │   │   ├── seed0630
    │   │   │   ├── seed0630gr.ini
    │   │   │   └── seed0630newton.ini
    │   │   ├── seed0631
    │   │   │   ├── seed0631gr.ini
    │   │   │   └── seed0631newton.ini
    │   │   ├── seed0632
    │   │   │   ├── seed0632gr.ini
    │   │   │   └── seed0632newton.ini
    │   │   ├── seed0633
    │   │   │   ├── seed0633gr.ini
    │   │   │   └── seed0633newton.ini
    │   │   ├── seed0634
    │   │   │   ├── seed0634gr.ini
    │   │   │   └── seed0634newton.ini
    │   │   ├── seed0635
    │   │   │   ├── seed0635gr.ini
    │   │   │   └── seed0635newton.ini
    │   │   ├── seed0636
    │   │   │   ├── seed0636gr.ini
    │   │   │   └── seed0636newton.ini
    │   │   ├── seed0637
    │   │   │   ├── seed0637gr.ini
    │   │   │   └── seed0637newton.ini
    │   │   ├── seed0638
    │   │   │   ├── seed0638gr.ini
    │   │   │   └── seed0638newton.ini
    │   │   ├── seed0639
    │   │   │   ├── seed0639gr.ini
    │   │   │   └── seed0639newton.ini
    │   │   ├── seed0640
    │   │   │   ├── seed0640gr.ini
    │   │   │   └── seed0640newton.ini
    │   │   ├── seed0641
    │   │   │   ├── seed0641gr.ini
    │   │   │   └── seed0641newton.ini
    │   │   ├── seed0642
    │   │   │   ├── seed0642gr.ini
    │   │   │   └── seed0642newton.ini
    │   │   ├── seed0643
    │   │   │   ├── seed0643gr.ini
    │   │   │   └── seed0643newton.ini
    │   │   ├── seed0644
    │   │   │   ├── seed0644gr.ini
    │   │   │   └── seed0644newton.ini
    │   │   ├── seed0645
    │   │   │   ├── seed0645gr.ini
    │   │   │   └── seed0645newton.ini
    │   │   ├── seed0646
    │   │   │   ├── seed0646gr.ini
    │   │   │   └── seed0646newton.ini
    │   │   ├── seed0647
    │   │   │   ├── seed0647gr.ini
    │   │   │   └── seed0647newton.ini
    │   │   ├── seed0648
    │   │   │   ├── seed0648gr.ini
    │   │   │   └── seed0648newton.ini
    │   │   ├── seed0649
    │   │   │   ├── seed0649gr.ini
    │   │   │   └── seed0649newton.ini
    │   │   ├── seed0650
    │   │   │   ├── seed0650gr.ini
    │   │   │   └── seed0650newton.ini
    │   │   ├── seed0651
    │   │   │   ├── seed0651gr.ini
    │   │   │   └── seed0651newton.ini
    │   │   ├── seed0652
    │   │   │   ├── seed0652gr.ini
    │   │   │   └── seed0652newton.ini
    │   │   ├── seed0653
    │   │   │   ├── seed0653gr.ini
    │   │   │   └── seed0653newton.ini
    │   │   ├── seed0654
    │   │   │   ├── seed0654gr.ini
    │   │   │   └── seed0654newton.ini
    │   │   ├── seed0655
    │   │   │   ├── seed0655gr.ini
    │   │   │   └── seed0655newton.ini
    │   │   ├── seed0656
    │   │   │   ├── seed0656gr.ini
    │   │   │   └── seed0656newton.ini
    │   │   ├── seed0657
    │   │   │   ├── seed0657gr.ini
    │   │   │   └── seed0657newton.ini
    │   │   ├── seed0658
    │   │   │   ├── seed0658gr.ini
    │   │   │   └── seed0658newton.ini
    │   │   ├── seed0659
    │   │   │   ├── seed0659gr.ini
    │   │   │   └── seed0659newton.ini
    │   │   ├── seed0660
    │   │   │   ├── seed0660gr.ini
    │   │   │   └── seed0660newton.ini
    │   │   ├── seed0661
    │   │   │   ├── seed0661gr.ini
    │   │   │   └── seed0661newton.ini
    │   │   ├── seed0662
    │   │   │   ├── seed0662gr.ini
    │   │   │   └── seed0662newton.ini
    │   │   ├── seed0663
    │   │   │   ├── seed0663gr.ini
    │   │   │   └── seed0663newton.ini
    │   │   ├── seed0664
    │   │   │   ├── seed0664gr.ini
    │   │   │   └── seed0664newton.ini
    │   │   ├── seed0665
    │   │   │   ├── seed0665gr.ini
    │   │   │   └── seed0665newton.ini
    │   │   ├── seed0666
    │   │   │   ├── seed0666gr.ini
    │   │   │   └── seed0666newton.ini
    │   │   ├── seed0667
    │   │   │   ├── seed0667gr.ini
    │   │   │   └── seed0667newton.ini
    │   │   ├── seed0668
    │   │   │   ├── seed0668gr.ini
    │   │   │   └── seed0668newton.ini
    │   │   ├── seed0669
    │   │   │   ├── seed0669gr.ini
    │   │   │   └── seed0669newton.ini
    │   │   ├── seed0670
    │   │   │   ├── seed0670gr.ini
    │   │   │   └── seed0670newton.ini
    │   │   ├── seed0671
    │   │   │   ├── seed0671gr.ini
    │   │   │   └── seed0671newton.ini
    │   │   ├── seed0672
    │   │   │   ├── seed0672gr.ini
    │   │   │   └── seed0672newton.ini
    │   │   ├── seed0673
    │   │   │   ├── seed0673gr.ini
    │   │   │   └── seed0673newton.ini
    │   │   ├── seed0674
    │   │   │   ├── seed0674gr.ini
    │   │   │   └── seed0674newton.ini
    │   │   ├── seed0675
    │   │   │   ├── seed0675gr.ini
    │   │   │   └── seed0675newton.ini
    │   │   ├── seed0676
    │   │   │   ├── seed0676gr.ini
    │   │   │   └── seed0676newton.ini
    │   │   ├── seed0677
    │   │   │   ├── seed0677gr.ini
    │   │   │   └── seed0677newton.ini
    │   │   ├── seed0678
    │   │   │   ├── seed0678gr.ini
    │   │   │   └── seed0678newton.ini
    │   │   ├── seed0679
    │   │   │   ├── seed0679gr.ini
    │   │   │   └── seed0679newton.ini
    │   │   ├── seed0680
    │   │   │   ├── seed0680gr.ini
    │   │   │   └── seed0680newton.ini
    │   │   ├── seed0681
    │   │   │   ├── seed0681gr.ini
    │   │   │   └── seed0681newton.ini
    │   │   ├── seed0682
    │   │   │   ├── seed0682gr.ini
    │   │   │   └── seed0682newton.ini
    │   │   ├── seed0683
    │   │   │   ├── seed0683gr.ini
    │   │   │   └── seed0683newton.ini
    │   │   ├── seed0684
    │   │   │   ├── seed0684gr.ini
    │   │   │   └── seed0684newton.ini
    │   │   ├── seed0685
    │   │   │   ├── seed0685gr.ini
    │   │   │   └── seed0685newton.ini
    │   │   ├── seed0686
    │   │   │   ├── seed0686gr.ini
    │   │   │   └── seed0686newton.ini
    │   │   ├── seed0687
    │   │   │   ├── seed0687gr.ini
    │   │   │   └── seed0687newton.ini
    │   │   ├── seed0688
    │   │   │   ├── seed0688gr.ini
    │   │   │   └── seed0688newton.ini
    │   │   ├── seed0689
    │   │   │   ├── seed0689gr.ini
    │   │   │   └── seed0689newton.ini
    │   │   ├── seed0690
    │   │   │   ├── seed0690gr.ini
    │   │   │   └── seed0690newton.ini
    │   │   ├── seed0691
    │   │   │   ├── seed0691gr.ini
    │   │   │   └── seed0691newton.ini
    │   │   ├── seed0692
    │   │   │   ├── seed0692gr.ini
    │   │   │   └── seed0692newton.ini
    │   │   ├── seed0693
    │   │   │   ├── seed0693gr.ini
    │   │   │   └── seed0693newton.ini
    │   │   ├── seed0694
    │   │   │   ├── seed0694gr.ini
    │   │   │   └── seed0694newton.ini
    │   │   ├── seed0695
    │   │   │   ├── seed0695gr.ini
    │   │   │   └── seed0695newton.ini
    │   │   ├── seed0696
    │   │   │   ├── seed0696gr.ini
    │   │   │   └── seed0696newton.ini
    │   │   ├── seed0697
    │   │   │   ├── seed0697gr.ini
    │   │   │   └── seed0697newton.ini
    │   │   ├── seed0698
    │   │   │   ├── seed0698gr.ini
    │   │   │   └── seed0698newton.ini
    │   │   ├── seed0699
    │   │   │   ├── seed0699gr.ini
    │   │   │   └── seed0699newton.ini
    │   │   ├── seed0700
    │   │   │   ├── seed0700gr.ini
    │   │   │   └── seed0700newton.ini
    │   │   ├── seed0701
    │   │   │   ├── seed0701gr.ini
    │   │   │   └── seed0701newton.ini
    │   │   ├── seed0702
    │   │   │   ├── seed0702gr.ini
    │   │   │   └── seed0702newton.ini
    │   │   ├── seed0703
    │   │   │   ├── seed0703gr.ini
    │   │   │   └── seed0703newton.ini
    │   │   ├── seed0704
    │   │   │   ├── seed0704gr.ini
    │   │   │   └── seed0704newton.ini
    │   │   ├── seed0705
    │   │   │   ├── seed0705gr.ini
    │   │   │   └── seed0705newton.ini
    │   │   ├── seed0706
    │   │   │   ├── seed0706gr.ini
    │   │   │   └── seed0706newton.ini
    │   │   ├── seed0707
    │   │   │   ├── seed0707gr.ini
    │   │   │   └── seed0707newton.ini
    │   │   ├── seed0708
    │   │   │   ├── seed0708gr.ini
    │   │   │   └── seed0708newton.ini
    │   │   ├── seed0709
    │   │   │   ├── seed0709gr.ini
    │   │   │   └── seed0709newton.ini
    │   │   ├── seed0710
    │   │   │   ├── seed0710gr.ini
    │   │   │   └── seed0710newton.ini
    │   │   ├── seed0711
    │   │   │   ├── seed0711gr.ini
    │   │   │   └── seed0711newton.ini
    │   │   ├── seed0712
    │   │   │   ├── seed0712gr.ini
    │   │   │   └── seed0712newton.ini
    │   │   ├── seed0713
    │   │   │   ├── seed0713gr.ini
    │   │   │   └── seed0713newton.ini
    │   │   ├── seed0714
    │   │   │   ├── seed0714gr.ini
    │   │   │   └── seed0714newton.ini
    │   │   ├── seed0715
    │   │   │   ├── seed0715gr.ini
    │   │   │   └── seed0715newton.ini
    │   │   ├── seed0716
    │   │   │   ├── seed0716gr.ini
    │   │   │   └── seed0716newton.ini
    │   │   ├── seed0717
    │   │   │   ├── seed0717gr.ini
    │   │   │   └── seed0717newton.ini
    │   │   ├── seed0718
    │   │   │   ├── seed0718gr.ini
    │   │   │   └── seed0718newton.ini
    │   │   ├── seed0719
    │   │   │   ├── seed0719gr.ini
    │   │   │   └── seed0719newton.ini
    │   │   ├── seed0720
    │   │   │   ├── seed0720gr.ini
    │   │   │   └── seed0720newton.ini
    │   │   ├── seed0721
    │   │   │   ├── seed0721gr.ini
    │   │   │   └── seed0721newton.ini
    │   │   ├── seed0722
    │   │   │   ├── seed0722gr.ini
    │   │   │   └── seed0722newton.ini
    │   │   ├── seed0723
    │   │   │   ├── seed0723gr.ini
    │   │   │   └── seed0723newton.ini
    │   │   ├── seed0724
    │   │   │   ├── seed0724gr.ini
    │   │   │   └── seed0724newton.ini
    │   │   ├── seed0725
    │   │   │   ├── seed0725gr.ini
    │   │   │   └── seed0725newton.ini
    │   │   ├── seed0726
    │   │   │   ├── seed0726gr.ini
    │   │   │   └── seed0726newton.ini
    │   │   ├── seed0727
    │   │   │   ├── seed0727gr.ini
    │   │   │   └── seed0727newton.ini
    │   │   ├── seed0728
    │   │   │   ├── seed0728gr.ini
    │   │   │   └── seed0728newton.ini
    │   │   ├── seed0729
    │   │   │   ├── seed0729gr.ini
    │   │   │   └── seed0729newton.ini
    │   │   ├── seed0730
    │   │   │   ├── seed0730gr.ini
    │   │   │   └── seed0730newton.ini
    │   │   ├── seed0731
    │   │   │   ├── seed0731gr.ini
    │   │   │   └── seed0731newton.ini
    │   │   ├── seed0732
    │   │   │   ├── seed0732gr.ini
    │   │   │   └── seed0732newton.ini
    │   │   ├── seed0733
    │   │   │   ├── seed0733gr.ini
    │   │   │   └── seed0733newton.ini
    │   │   ├── seed0734
    │   │   │   ├── seed0734gr.ini
    │   │   │   └── seed0734newton.ini
    │   │   ├── seed0735
    │   │   │   ├── seed0735gr.ini
    │   │   │   └── seed0735newton.ini
    │   │   ├── seed0736
    │   │   │   ├── seed0736gr.ini
    │   │   │   └── seed0736newton.ini
    │   │   ├── seed0737
    │   │   │   ├── seed0737gr.ini
    │   │   │   └── seed0737newton.ini
    │   │   ├── seed0738
    │   │   │   ├── seed0738gr.ini
    │   │   │   └── seed0738newton.ini
    │   │   ├── seed0739
    │   │   │   ├── seed0739gr.ini
    │   │   │   └── seed0739newton.ini
    │   │   ├── seed0740
    │   │   │   ├── seed0740gr.ini
    │   │   │   └── seed0740newton.ini
    │   │   ├── seed0741
    │   │   │   ├── seed0741gr.ini
    │   │   │   └── seed0741newton.ini
    │   │   ├── seed0742
    │   │   │   ├── seed0742gr.ini
    │   │   │   └── seed0742newton.ini
    │   │   ├── seed0743
    │   │   │   ├── seed0743gr.ini
    │   │   │   └── seed0743newton.ini
    │   │   ├── seed0744
    │   │   │   ├── seed0744gr.ini
    │   │   │   └── seed0744newton.ini
    │   │   ├── seed0745
    │   │   │   ├── seed0745gr.ini
    │   │   │   └── seed0745newton.ini
    │   │   ├── seed0746
    │   │   │   ├── seed0746gr.ini
    │   │   │   └── seed0746newton.ini
    │   │   ├── seed0747
    │   │   │   ├── seed0747gr.ini
    │   │   │   └── seed0747newton.ini
    │   │   ├── seed0748
    │   │   │   ├── seed0748gr.ini
    │   │   │   └── seed0748newton.ini
    │   │   ├── seed0749
    │   │   │   ├── seed0749gr.ini
    │   │   │   └── seed0749newton.ini
    │   │   ├── seed0750
    │   │   │   ├── seed0750gr.ini
    │   │   │   └── seed0750newton.ini
    │   │   ├── seed0751
    │   │   │   ├── seed0751gr.ini
    │   │   │   └── seed0751newton.ini
    │   │   ├── seed0752
    │   │   │   ├── seed0752gr.ini
    │   │   │   └── seed0752newton.ini
    │   │   ├── seed0753
    │   │   │   ├── seed0753gr.ini
    │   │   │   └── seed0753newton.ini
    │   │   ├── seed0754
    │   │   │   ├── seed0754gr.ini
    │   │   │   └── seed0754newton.ini
    │   │   ├── seed0755
    │   │   │   ├── seed0755gr.ini
    │   │   │   └── seed0755newton.ini
    │   │   ├── seed0756
    │   │   │   ├── seed0756gr.ini
    │   │   │   └── seed0756newton.ini
    │   │   ├── seed0757
    │   │   │   ├── seed0757gr.ini
    │   │   │   └── seed0757newton.ini
    │   │   ├── seed0758
    │   │   │   ├── seed0758gr.ini
    │   │   │   └── seed0758newton.ini
    │   │   ├── seed0759
    │   │   │   ├── seed0759gr.ini
    │   │   │   └── seed0759newton.ini
    │   │   ├── seed0760
    │   │   │   ├── seed0760gr.ini
    │   │   │   └── seed0760newton.ini
    │   │   ├── seed0761
    │   │   │   ├── seed0761gr.ini
    │   │   │   └── seed0761newton.ini
    │   │   ├── seed0762
    │   │   │   ├── seed0762gr.ini
    │   │   │   └── seed0762newton.ini
    │   │   ├── seed0763
    │   │   │   ├── seed0763gr.ini
    │   │   │   └── seed0763newton.ini
    │   │   ├── seed0764
    │   │   │   ├── seed0764gr.ini
    │   │   │   └── seed0764newton.ini
    │   │   ├── seed0765
    │   │   │   ├── seed0765gr.ini
    │   │   │   └── seed0765newton.ini
    │   │   ├── seed0766
    │   │   │   ├── seed0766gr.ini
    │   │   │   └── seed0766newton.ini
    │   │   ├── seed0767
    │   │   │   ├── seed0767gr.ini
    │   │   │   └── seed0767newton.ini
    │   │   ├── seed0768
    │   │   │   ├── seed0768gr.ini
    │   │   │   └── seed0768newton.ini
    │   │   ├── seed0769
    │   │   │   ├── seed0769gr.ini
    │   │   │   └── seed0769newton.ini
    │   │   ├── seed0770
    │   │   │   ├── seed0770gr.ini
    │   │   │   └── seed0770newton.ini
    │   │   ├── seed0771
    │   │   │   ├── seed0771gr.ini
    │   │   │   └── seed0771newton.ini
    │   │   ├── seed0772
    │   │   │   ├── seed0772gr.ini
    │   │   │   └── seed0772newton.ini
    │   │   ├── seed0773
    │   │   │   ├── seed0773gr.ini
    │   │   │   └── seed0773newton.ini
    │   │   ├── seed0774
    │   │   │   ├── seed0774gr.ini
    │   │   │   └── seed0774newton.ini
    │   │   ├── seed0775
    │   │   │   ├── seed0775gr.ini
    │   │   │   └── seed0775newton.ini
    │   │   ├── seed0776
    │   │   │   ├── seed0776gr.ini
    │   │   │   └── seed0776newton.ini
    │   │   ├── seed0777
    │   │   │   ├── seed0777gr.ini
    │   │   │   └── seed0777newton.ini
    │   │   ├── seed0778
    │   │   │   ├── seed0778gr.ini
    │   │   │   └── seed0778newton.ini
    │   │   ├── seed0779
    │   │   │   ├── seed0779gr.ini
    │   │   │   └── seed0779newton.ini
    │   │   ├── seed0780
    │   │   │   ├── seed0780gr.ini
    │   │   │   └── seed0780newton.ini
    │   │   ├── seed0781
    │   │   │   ├── seed0781gr.ini
    │   │   │   └── seed0781newton.ini
    │   │   ├── seed0782
    │   │   │   ├── seed0782gr.ini
    │   │   │   └── seed0782newton.ini
    │   │   ├── seed0783
    │   │   │   ├── seed0783gr.ini
    │   │   │   └── seed0783newton.ini
    │   │   ├── seed0784
    │   │   │   ├── seed0784gr.ini
    │   │   │   └── seed0784newton.ini
    │   │   ├── seed0785
    │   │   │   ├── seed0785gr.ini
    │   │   │   └── seed0785newton.ini
    │   │   ├── seed0786
    │   │   │   ├── seed0786gr.ini
    │   │   │   └── seed0786newton.ini
    │   │   ├── seed0787
    │   │   │   ├── seed0787gr.ini
    │   │   │   └── seed0787newton.ini
    │   │   ├── seed0788
    │   │   │   ├── seed0788gr.ini
    │   │   │   └── seed0788newton.ini
    │   │   ├── seed0789
    │   │   │   ├── seed0789gr.ini
    │   │   │   └── seed0789newton.ini
    │   │   ├── seed0790
    │   │   │   ├── seed0790gr.ini
    │   │   │   └── seed0790newton.ini
    │   │   ├── seed0791
    │   │   │   ├── seed0791gr.ini
    │   │   │   └── seed0791newton.ini
    │   │   ├── seed0792
    │   │   │   ├── seed0792gr.ini
    │   │   │   └── seed0792newton.ini
    │   │   ├── seed0793
    │   │   │   ├── seed0793gr.ini
    │   │   │   └── seed0793newton.ini
    │   │   ├── seed0794
    │   │   │   ├── seed0794gr.ini
    │   │   │   └── seed0794newton.ini
    │   │   ├── seed0795
    │   │   │   ├── seed0795gr.ini
    │   │   │   └── seed0795newton.ini
    │   │   ├── seed0796
    │   │   │   ├── seed0796gr.ini
    │   │   │   └── seed0796newton.ini
    │   │   ├── seed0797
    │   │   │   ├── seed0797gr.ini
    │   │   │   └── seed0797newton.ini
    │   │   ├── seed0798
    │   │   │   ├── seed0798gr.ini
    │   │   │   └── seed0798newton.ini
    │   │   ├── seed0799
    │   │   │   ├── seed0799gr.ini
    │   │   │   └── seed0799newton.ini
    │   │   ├── seed0800
    │   │   │   ├── seed0800gr.ini
    │   │   │   └── seed0800newton.ini
    │   │   ├── seed0801
    │   │   │   ├── seed0801gr.ini
    │   │   │   └── seed0801newton.ini
    │   │   ├── seed0802
    │   │   │   ├── seed0802gr.ini
    │   │   │   └── seed0802newton.ini
    │   │   ├── seed0803
    │   │   │   ├── seed0803gr.ini
    │   │   │   └── seed0803newton.ini
    │   │   ├── seed0804
    │   │   │   ├── seed0804gr.ini
    │   │   │   └── seed0804newton.ini
    │   │   ├── seed0805
    │   │   │   ├── seed0805gr.ini
    │   │   │   └── seed0805newton.ini
    │   │   ├── seed0806
    │   │   │   ├── seed0806gr.ini
    │   │   │   └── seed0806newton.ini
    │   │   ├── seed0807
    │   │   │   ├── seed0807gr.ini
    │   │   │   └── seed0807newton.ini
    │   │   ├── seed0808
    │   │   │   ├── seed0808gr.ini
    │   │   │   └── seed0808newton.ini
    │   │   ├── seed0809
    │   │   │   ├── seed0809gr.ini
    │   │   │   └── seed0809newton.ini
    │   │   ├── seed0810
    │   │   │   ├── seed0810gr.ini
    │   │   │   └── seed0810newton.ini
    │   │   ├── seed0811
    │   │   │   ├── seed0811gr.ini
    │   │   │   └── seed0811newton.ini
    │   │   ├── seed0812
    │   │   │   ├── seed0812gr.ini
    │   │   │   └── seed0812newton.ini
    │   │   ├── seed0813
    │   │   │   ├── seed0813gr.ini
    │   │   │   └── seed0813newton.ini
    │   │   ├── seed0814
    │   │   │   ├── seed0814gr.ini
    │   │   │   └── seed0814newton.ini
    │   │   ├── seed0815
    │   │   │   ├── seed0815gr.ini
    │   │   │   └── seed0815newton.ini
    │   │   ├── seed0816
    │   │   │   ├── seed0816gr.ini
    │   │   │   └── seed0816newton.ini
    │   │   ├── seed0817
    │   │   │   ├── seed0817gr.ini
    │   │   │   └── seed0817newton.ini
    │   │   ├── seed0818
    │   │   │   ├── seed0818gr.ini
    │   │   │   └── seed0818newton.ini
    │   │   ├── seed0819
    │   │   │   ├── seed0819gr.ini
    │   │   │   └── seed0819newton.ini
    │   │   ├── seed0820
    │   │   │   ├── seed0820gr.ini
    │   │   │   └── seed0820newton.ini
    │   │   ├── seed0821
    │   │   │   ├── seed0821gr.ini
    │   │   │   └── seed0821newton.ini
    │   │   ├── seed0822
    │   │   │   ├── seed0822gr.ini
    │   │   │   └── seed0822newton.ini
    │   │   ├── seed0823
    │   │   │   ├── seed0823gr.ini
    │   │   │   └── seed0823newton.ini
    │   │   ├── seed0824
    │   │   │   ├── seed0824gr.ini
    │   │   │   └── seed0824newton.ini
    │   │   ├── seed0825
    │   │   │   ├── seed0825gr.ini
    │   │   │   └── seed0825newton.ini
    │   │   ├── seed0826
    │   │   │   ├── seed0826gr.ini
    │   │   │   └── seed0826newton.ini
    │   │   ├── seed0827
    │   │   │   ├── seed0827gr.ini
    │   │   │   └── seed0827newton.ini
    │   │   ├── seed0828
    │   │   │   ├── seed0828gr.ini
    │   │   │   └── seed0828newton.ini
    │   │   ├── seed0829
    │   │   │   ├── seed0829gr.ini
    │   │   │   └── seed0829newton.ini
    │   │   ├── seed0830
    │   │   │   ├── seed0830gr.ini
    │   │   │   └── seed0830newton.ini
    │   │   ├── seed0831
    │   │   │   ├── seed0831gr.ini
    │   │   │   └── seed0831newton.ini
    │   │   ├── seed0832
    │   │   │   ├── seed0832gr.ini
    │   │   │   └── seed0832newton.ini
    │   │   ├── seed0833
    │   │   │   ├── seed0833gr.ini
    │   │   │   └── seed0833newton.ini
    │   │   ├── seed0834
    │   │   │   ├── seed0834gr.ini
    │   │   │   └── seed0834newton.ini
    │   │   ├── seed0835
    │   │   │   ├── seed0835gr.ini
    │   │   │   └── seed0835newton.ini
    │   │   ├── seed0836
    │   │   │   ├── seed0836gr.ini
    │   │   │   └── seed0836newton.ini
    │   │   ├── seed0837
    │   │   │   ├── seed0837gr.ini
    │   │   │   └── seed0837newton.ini
    │   │   ├── seed0838
    │   │   │   ├── seed0838gr.ini
    │   │   │   └── seed0838newton.ini
    │   │   ├── seed0839
    │   │   │   ├── seed0839gr.ini
    │   │   │   └── seed0839newton.ini
    │   │   ├── seed0840
    │   │   │   ├── seed0840gr.ini
    │   │   │   └── seed0840newton.ini
    │   │   ├── seed0841
    │   │   │   ├── seed0841gr.ini
    │   │   │   └── seed0841newton.ini
    │   │   ├── seed0842
    │   │   │   ├── seed0842gr.ini
    │   │   │   └── seed0842newton.ini
    │   │   ├── seed0843
    │   │   │   ├── seed0843gr.ini
    │   │   │   └── seed0843newton.ini
    │   │   ├── seed0844
    │   │   │   ├── seed0844gr.ini
    │   │   │   └── seed0844newton.ini
    │   │   ├── seed0845
    │   │   │   ├── seed0845gr.ini
    │   │   │   └── seed0845newton.ini
    │   │   ├── seed0846
    │   │   │   ├── seed0846gr.ini
    │   │   │   └── seed0846newton.ini
    │   │   ├── seed0847
    │   │   │   ├── seed0847gr.ini
    │   │   │   └── seed0847newton.ini
    │   │   ├── seed0848
    │   │   │   ├── seed0848gr.ini
    │   │   │   └── seed0848newton.ini
    │   │   ├── seed0849
    │   │   │   ├── seed0849gr.ini
    │   │   │   └── seed0849newton.ini
    │   │   ├── seed0850
    │   │   │   ├── seed0850gr.ini
    │   │   │   └── seed0850newton.ini
    │   │   ├── seed0851
    │   │   │   ├── seed0851gr.ini
    │   │   │   └── seed0851newton.ini
    │   │   ├── seed0852
    │   │   │   ├── seed0852gr.ini
    │   │   │   └── seed0852newton.ini
    │   │   ├── seed0853
    │   │   │   ├── seed0853gr.ini
    │   │   │   └── seed0853newton.ini
    │   │   ├── seed0854
    │   │   │   ├── seed0854gr.ini
    │   │   │   └── seed0854newton.ini
    │   │   ├── seed0855
    │   │   │   ├── seed0855gr.ini
    │   │   │   └── seed0855newton.ini
    │   │   ├── seed0856
    │   │   │   ├── seed0856gr.ini
    │   │   │   └── seed0856newton.ini
    │   │   ├── seed0857
    │   │   │   ├── seed0857gr.ini
    │   │   │   └── seed0857newton.ini
    │   │   ├── seed0858
    │   │   │   ├── seed0858gr.ini
    │   │   │   └── seed0858newton.ini
    │   │   ├── seed0859
    │   │   │   ├── seed0859gr.ini
    │   │   │   └── seed0859newton.ini
    │   │   ├── seed0860
    │   │   │   ├── seed0860gr.ini
    │   │   │   └── seed0860newton.ini
    │   │   ├── seed0861
    │   │   │   ├── seed0861gr.ini
    │   │   │   └── seed0861newton.ini
    │   │   ├── seed0862
    │   │   │   ├── seed0862gr.ini
    │   │   │   └── seed0862newton.ini
    │   │   ├── seed0863
    │   │   │   ├── seed0863gr.ini
    │   │   │   └── seed0863newton.ini
    │   │   ├── seed0864
    │   │   │   ├── seed0864gr.ini
    │   │   │   └── seed0864newton.ini
    │   │   ├── seed0865
    │   │   │   ├── seed0865gr.ini
    │   │   │   └── seed0865newton.ini
    │   │   ├── seed0866
    │   │   │   ├── seed0866gr.ini
    │   │   │   └── seed0866newton.ini
    │   │   ├── seed0867
    │   │   │   ├── seed0867gr.ini
    │   │   │   └── seed0867newton.ini
    │   │   ├── seed0868
    │   │   │   ├── seed0868gr.ini
    │   │   │   └── seed0868newton.ini
    │   │   ├── seed0869
    │   │   │   ├── seed0869gr.ini
    │   │   │   └── seed0869newton.ini
    │   │   ├── seed0870
    │   │   │   ├── seed0870gr.ini
    │   │   │   └── seed0870newton.ini
    │   │   ├── seed0871
    │   │   │   ├── seed0871gr.ini
    │   │   │   └── seed0871newton.ini
    │   │   ├── seed0872
    │   │   │   ├── seed0872gr.ini
    │   │   │   └── seed0872newton.ini
    │   │   ├── seed0873
    │   │   │   ├── seed0873gr.ini
    │   │   │   └── seed0873newton.ini
    │   │   ├── seed0874
    │   │   │   ├── seed0874gr.ini
    │   │   │   └── seed0874newton.ini
    │   │   ├── seed0875
    │   │   │   ├── seed0875gr.ini
    │   │   │   └── seed0875newton.ini
    │   │   ├── seed0876
    │   │   │   ├── seed0876gr.ini
    │   │   │   └── seed0876newton.ini
    │   │   ├── seed0877
    │   │   │   ├── seed0877gr.ini
    │   │   │   └── seed0877newton.ini
    │   │   ├── seed0878
    │   │   │   ├── seed0878gr.ini
    │   │   │   └── seed0878newton.ini
    │   │   ├── seed0879
    │   │   │   ├── seed0879gr.ini
    │   │   │   └── seed0879newton.ini
    │   │   ├── seed0880
    │   │   │   ├── seed0880gr.ini
    │   │   │   └── seed0880newton.ini
    │   │   ├── seed0881
    │   │   │   ├── seed0881gr.ini
    │   │   │   └── seed0881newton.ini
    │   │   ├── seed0882
    │   │   │   ├── seed0882gr.ini
    │   │   │   └── seed0882newton.ini
    │   │   ├── seed0883
    │   │   │   ├── seed0883gr.ini
    │   │   │   └── seed0883newton.ini
    │   │   ├── seed0884
    │   │   │   ├── seed0884gr.ini
    │   │   │   └── seed0884newton.ini
    │   │   ├── seed0885
    │   │   │   ├── seed0885gr.ini
    │   │   │   └── seed0885newton.ini
    │   │   ├── seed0886
    │   │   │   ├── seed0886gr.ini
    │   │   │   └── seed0886newton.ini
    │   │   ├── seed0887
    │   │   │   ├── seed0887gr.ini
    │   │   │   └── seed0887newton.ini
    │   │   ├── seed0888
    │   │   │   ├── seed0888gr.ini
    │   │   │   └── seed0888newton.ini
    │   │   ├── seed0889
    │   │   │   ├── seed0889gr.ini
    │   │   │   └── seed0889newton.ini
    │   │   ├── seed0890
    │   │   │   ├── seed0890gr.ini
    │   │   │   └── seed0890newton.ini
    │   │   ├── seed0891
    │   │   │   ├── seed0891gr.ini
    │   │   │   └── seed0891newton.ini
    │   │   ├── seed0892
    │   │   │   ├── seed0892gr.ini
    │   │   │   └── seed0892newton.ini
    │   │   ├── seed0893
    │   │   │   ├── seed0893gr.ini
    │   │   │   └── seed0893newton.ini
    │   │   ├── seed0894
    │   │   │   ├── seed0894gr.ini
    │   │   │   └── seed0894newton.ini
    │   │   ├── seed0895
    │   │   │   ├── seed0895gr.ini
    │   │   │   └── seed0895newton.ini
    │   │   ├── seed0896
    │   │   │   ├── seed0896gr.ini
    │   │   │   └── seed0896newton.ini
    │   │   ├── seed0897
    │   │   │   ├── seed0897gr.ini
    │   │   │   └── seed0897newton.ini
    │   │   ├── seed0898
    │   │   │   ├── seed0898gr.ini
    │   │   │   └── seed0898newton.ini
    │   │   ├── seed0899
    │   │   │   ├── seed0899gr.ini
    │   │   │   └── seed0899newton.ini
    │   │   ├── seed0900
    │   │   │   ├── seed0900gr.ini
    │   │   │   └── seed0900newton.ini
    │   │   ├── seed0901
    │   │   │   ├── seed0901gr.ini
    │   │   │   └── seed0901newton.ini
    │   │   ├── seed0902
    │   │   │   ├── seed0902gr.ini
    │   │   │   └── seed0902newton.ini
    │   │   ├── seed0903
    │   │   │   ├── seed0903gr.ini
    │   │   │   └── seed0903newton.ini
    │   │   ├── seed0904
    │   │   │   ├── seed0904gr.ini
    │   │   │   └── seed0904newton.ini
    │   │   ├── seed0905
    │   │   │   ├── seed0905gr.ini
    │   │   │   └── seed0905newton.ini
    │   │   ├── seed0906
    │   │   │   ├── seed0906gr.ini
    │   │   │   └── seed0906newton.ini
    │   │   ├── seed0907
    │   │   │   ├── seed0907gr.ini
    │   │   │   └── seed0907newton.ini
    │   │   ├── seed0908
    │   │   │   ├── seed0908gr.ini
    │   │   │   └── seed0908newton.ini
    │   │   ├── seed0909
    │   │   │   ├── seed0909gr.ini
    │   │   │   └── seed0909newton.ini
    │   │   ├── seed0910
    │   │   │   ├── seed0910gr.ini
    │   │   │   └── seed0910newton.ini
    │   │   ├── seed0911
    │   │   │   ├── seed0911gr.ini
    │   │   │   └── seed0911newton.ini
    │   │   ├── seed0912
    │   │   │   ├── seed0912gr.ini
    │   │   │   └── seed0912newton.ini
    │   │   ├── seed0913
    │   │   │   ├── seed0913gr.ini
    │   │   │   └── seed0913newton.ini
    │   │   ├── seed0914
    │   │   │   ├── seed0914gr.ini
    │   │   │   └── seed0914newton.ini
    │   │   ├── seed0915
    │   │   │   ├── seed0915gr.ini
    │   │   │   └── seed0915newton.ini
    │   │   ├── seed0916
    │   │   │   ├── seed0916gr.ini
    │   │   │   └── seed0916newton.ini
    │   │   ├── seed0917
    │   │   │   ├── seed0917gr.ini
    │   │   │   └── seed0917newton.ini
    │   │   ├── seed0918
    │   │   │   ├── seed0918gr.ini
    │   │   │   └── seed0918newton.ini
    │   │   ├── seed0919
    │   │   │   ├── seed0919gr.ini
    │   │   │   └── seed0919newton.ini
    │   │   ├── seed0920
    │   │   │   ├── seed0920gr.ini
    │   │   │   └── seed0920newton.ini
    │   │   ├── seed0921
    │   │   │   ├── seed0921gr.ini
    │   │   │   └── seed0921newton.ini
    │   │   ├── seed0922
    │   │   │   ├── seed0922gr.ini
    │   │   │   └── seed0922newton.ini
    │   │   ├── seed0923
    │   │   │   ├── seed0923gr.ini
    │   │   │   └── seed0923newton.ini
    │   │   ├── seed0924
    │   │   │   ├── seed0924gr.ini
    │   │   │   └── seed0924newton.ini
    │   │   ├── seed0925
    │   │   │   ├── seed0925gr.ini
    │   │   │   └── seed0925newton.ini
    │   │   ├── seed0926
    │   │   │   ├── seed0926gr.ini
    │   │   │   └── seed0926newton.ini
    │   │   ├── seed0927
    │   │   │   ├── seed0927gr.ini
    │   │   │   └── seed0927newton.ini
    │   │   ├── seed0928
    │   │   │   ├── seed0928gr.ini
    │   │   │   └── seed0928newton.ini
    │   │   ├── seed0929
    │   │   │   ├── seed0929gr.ini
    │   │   │   └── seed0929newton.ini
    │   │   ├── seed0930
    │   │   │   ├── seed0930gr.ini
    │   │   │   └── seed0930newton.ini
    │   │   ├── seed0931
    │   │   │   ├── seed0931gr.ini
    │   │   │   └── seed0931newton.ini
    │   │   ├── seed0932
    │   │   │   ├── seed0932gr.ini
    │   │   │   └── seed0932newton.ini
    │   │   ├── seed0933
    │   │   │   ├── seed0933gr.ini
    │   │   │   └── seed0933newton.ini
    │   │   ├── seed0934
    │   │   │   ├── seed0934gr.ini
    │   │   │   └── seed0934newton.ini
    │   │   ├── seed0935
    │   │   │   ├── seed0935gr.ini
    │   │   │   └── seed0935newton.ini
    │   │   ├── seed0936
    │   │   │   ├── seed0936gr.ini
    │   │   │   └── seed0936newton.ini
    │   │   ├── seed0937
    │   │   │   ├── seed0937gr.ini
    │   │   │   └── seed0937newton.ini
    │   │   ├── seed0938
    │   │   │   ├── seed0938gr.ini
    │   │   │   └── seed0938newton.ini
    │   │   ├── seed0939
    │   │   │   ├── seed0939gr.ini
    │   │   │   └── seed0939newton.ini
    │   │   ├── seed0940
    │   │   │   ├── seed0940gr.ini
    │   │   │   └── seed0940newton.ini
    │   │   ├── seed0941
    │   │   │   ├── seed0941gr.ini
    │   │   │   └── seed0941newton.ini
    │   │   ├── seed0942
    │   │   │   ├── seed0942gr.ini
    │   │   │   └── seed0942newton.ini
    │   │   ├── seed0943
    │   │   │   ├── seed0943gr.ini
    │   │   │   └── seed0943newton.ini
    │   │   ├── seed0944
    │   │   │   ├── seed0944gr.ini
    │   │   │   └── seed0944newton.ini
    │   │   ├── seed0945
    │   │   │   ├── seed0945gr.ini
    │   │   │   └── seed0945newton.ini
    │   │   ├── seed0946
    │   │   │   ├── seed0946gr.ini
    │   │   │   └── seed0946newton.ini
    │   │   ├── seed0947
    │   │   │   ├── seed0947gr.ini
    │   │   │   └── seed0947newton.ini
    │   │   ├── seed0948
    │   │   │   ├── seed0948gr.ini
    │   │   │   └── seed0948newton.ini
    │   │   ├── seed0949
    │   │   │   ├── seed0949gr.ini
    │   │   │   └── seed0949newton.ini
    │   │   ├── seed0950
    │   │   │   ├── seed0950gr.ini
    │   │   │   └── seed0950newton.ini
    │   │   ├── seed0951
    │   │   │   ├── seed0951gr.ini
    │   │   │   └── seed0951newton.ini
    │   │   ├── seed0952
    │   │   │   ├── seed0952gr.ini
    │   │   │   └── seed0952newton.ini
    │   │   ├── seed0953
    │   │   │   ├── seed0953gr.ini
    │   │   │   └── seed0953newton.ini
    │   │   ├── seed0954
    │   │   │   ├── seed0954gr.ini
    │   │   │   └── seed0954newton.ini
    │   │   ├── seed0955
    │   │   │   ├── seed0955gr.ini
    │   │   │   └── seed0955newton.ini
    │   │   ├── seed0956
    │   │   │   ├── seed0956gr.ini
    │   │   │   └── seed0956newton.ini
    │   │   ├── seed0957
    │   │   │   ├── seed0957gr.ini
    │   │   │   └── seed0957newton.ini
    │   │   ├── seed0958
    │   │   │   ├── seed0958gr.ini
    │   │   │   └── seed0958newton.ini
    │   │   ├── seed0959
    │   │   │   ├── seed0959gr.ini
    │   │   │   └── seed0959newton.ini
    │   │   ├── seed0960
    │   │   │   ├── seed0960gr.ini
    │   │   │   └── seed0960newton.ini
    │   │   ├── seed0961
    │   │   │   ├── seed0961gr.ini
    │   │   │   └── seed0961newton.ini
    │   │   ├── seed0962
    │   │   │   ├── seed0962gr.ini
    │   │   │   └── seed0962newton.ini
    │   │   ├── seed0963
    │   │   │   ├── seed0963gr.ini
    │   │   │   └── seed0963newton.ini
    │   │   ├── seed0964
    │   │   │   ├── seed0964gr.ini
    │   │   │   └── seed0964newton.ini
    │   │   ├── seed0965
    │   │   │   ├── seed0965gr.ini
    │   │   │   └── seed0965newton.ini
    │   │   ├── seed0966
    │   │   │   ├── seed0966gr.ini
    │   │   │   └── seed0966newton.ini
    │   │   ├── seed0967
    │   │   │   ├── seed0967gr.ini
    │   │   │   └── seed0967newton.ini
    │   │   ├── seed0968
    │   │   │   ├── seed0968gr.ini
    │   │   │   └── seed0968newton.ini
    │   │   ├── seed0969
    │   │   │   ├── seed0969gr.ini
    │   │   │   └── seed0969newton.ini
    │   │   ├── seed0970
    │   │   │   ├── seed0970gr.ini
    │   │   │   └── seed0970newton.ini
    │   │   ├── seed0971
    │   │   │   ├── seed0971gr.ini
    │   │   │   └── seed0971newton.ini
    │   │   ├── seed0972
    │   │   │   ├── seed0972gr.ini
    │   │   │   └── seed0972newton.ini
    │   │   ├── seed0973
    │   │   │   ├── seed0973gr.ini
    │   │   │   └── seed0973newton.ini
    │   │   ├── seed0974
    │   │   │   ├── seed0974gr.ini
    │   │   │   └── seed0974newton.ini
    │   │   ├── seed0975
    │   │   │   ├── seed0975gr.ini
    │   │   │   └── seed0975newton.ini
    │   │   ├── seed0976
    │   │   │   ├── seed0976gr.ini
    │   │   │   └── seed0976newton.ini
    │   │   ├── seed0977
    │   │   │   ├── seed0977gr.ini
    │   │   │   └── seed0977newton.ini
    │   │   ├── seed0978
    │   │   │   ├── seed0978gr.ini
    │   │   │   └── seed0978newton.ini
    │   │   ├── seed0979
    │   │   │   ├── seed0979gr.ini
    │   │   │   └── seed0979newton.ini
    │   │   ├── seed0980
    │   │   │   ├── seed0980gr.ini
    │   │   │   └── seed0980newton.ini
    │   │   ├── seed0981
    │   │   │   ├── seed0981gr.ini
    │   │   │   └── seed0981newton.ini
    │   │   ├── seed0982
    │   │   │   ├── seed0982gr.ini
    │   │   │   └── seed0982newton.ini
    │   │   ├── seed0983
    │   │   │   ├── seed0983gr.ini
    │   │   │   └── seed0983newton.ini
    │   │   ├── seed0984
    │   │   │   ├── seed0984gr.ini
    │   │   │   └── seed0984newton.ini
    │   │   ├── seed0985
    │   │   │   ├── seed0985gr.ini
    │   │   │   └── seed0985newton.ini
    │   │   ├── seed0986
    │   │   │   ├── seed0986gr.ini
    │   │   │   └── seed0986newton.ini
    │   │   ├── seed0987
    │   │   │   ├── seed0987gr.ini
    │   │   │   └── seed0987newton.ini
    │   │   ├── seed0988
    │   │   │   ├── seed0988gr.ini
    │   │   │   └── seed0988newton.ini
    │   │   ├── seed0989
    │   │   │   ├── seed0989gr.ini
    │   │   │   └── seed0989newton.ini
    │   │   ├── seed0990
    │   │   │   ├── seed0990gr.ini
    │   │   │   └── seed0990newton.ini
    │   │   ├── seed0991
    │   │   │   ├── seed0991gr.ini
    │   │   │   └── seed0991newton.ini
    │   │   ├── seed0992
    │   │   │   ├── seed0992gr.ini
    │   │   │   └── seed0992newton.ini
    │   │   ├── seed0993
    │   │   │   ├── seed0993gr.ini
    │   │   │   └── seed0993newton.ini
    │   │   ├── seed0994
    │   │   │   ├── seed0994gr.ini
    │   │   │   └── seed0994newton.ini
    │   │   ├── seed0995
    │   │   │   ├── seed0995gr.ini
    │   │   │   └── seed0995newton.ini
    │   │   ├── seed0996
    │   │   │   ├── seed0996gr.ini
    │   │   │   └── seed0996newton.ini
    │   │   ├── seed0997
    │   │   │   ├── seed0997gr.ini
    │   │   │   └── seed0997newton.ini
    │   │   ├── seed0998
    │   │   │   ├── seed0998gr.ini
    │   │   │   └── seed0998newton.ini
    │   │   ├── seed0999
    │   │   │   ├── seed0999gr.ini
    │   │   │   └── seed0999newton.ini
    │   │   ├── seed1000
    │   │   │   ├── seed1000gr.ini
    │   │   │   └── seed1000newton.ini
    │   │   ├── seed1001
    │   │   │   ├── seed1001gr.ini
    │   │   │   └── seed1001newton.ini
    │   │   ├── seed1002
    │   │   │   ├── seed1002gr.ini
    │   │   │   └── seed1002newton.ini
    │   │   ├── seed1003
    │   │   │   ├── seed1003gr.ini
    │   │   │   └── seed1003newton.ini
    │   │   ├── seed1004
    │   │   │   ├── seed1004gr.ini
    │   │   │   └── seed1004newton.ini
    │   │   ├── seed1005
    │   │   │   ├── seed1005gr.ini
    │   │   │   └── seed1005newton.ini
    │   │   ├── seed1006
    │   │   │   ├── seed1006gr.ini
    │   │   │   └── seed1006newton.ini
    │   │   ├── seed1007
    │   │   │   ├── seed1007gr.ini
    │   │   │   └── seed1007newton.ini
    │   │   ├── seed1008
    │   │   │   ├── seed1008gr.ini
    │   │   │   └── seed1008newton.ini
    │   │   ├── seed1009
    │   │   │   ├── seed1009gr.ini
    │   │   │   └── seed1009newton.ini
    │   │   ├── seed1010
    │   │   │   ├── seed1010gr.ini
    │   │   │   └── seed1010newton.ini
    │   │   ├── seed1011
    │   │   │   ├── seed1011gr.ini
    │   │   │   └── seed1011newton.ini
    │   │   ├── seed1012
    │   │   │   ├── seed1012gr.ini
    │   │   │   └── seed1012newton.ini
    │   │   ├── seed1013
    │   │   │   ├── seed1013gr.ini
    │   │   │   └── seed1013newton.ini
    │   │   ├── seed1014
    │   │   │   ├── seed1014gr.ini
    │   │   │   └── seed1014newton.ini
    │   │   ├── seed1015
    │   │   │   ├── seed1015gr.ini
    │   │   │   └── seed1015newton.ini
    │   │   ├── seed1016
    │   │   │   ├── seed1016gr.ini
    │   │   │   └── seed1016newton.ini
    │   │   ├── seed1017
    │   │   │   ├── seed1017gr.ini
    │   │   │   └── seed1017newton.ini
    │   │   ├── seed1018
    │   │   │   ├── seed1018gr.ini
    │   │   │   └── seed1018newton.ini
    │   │   ├── seed1019
    │   │   │   ├── seed1019gr.ini
    │   │   │   └── seed1019newton.ini
    │   │   ├── seed1020
    │   │   │   ├── seed1020gr.ini
    │   │   │   └── seed1020newton.ini
    │   │   ├── seed1021
    │   │   │   ├── seed1021gr.ini
    │   │   │   └── seed1021newton.ini
    │   │   ├── seed1022
    │   │   │   ├── seed1022gr.ini
    │   │   │   └── seed1022newton.ini
    │   │   ├── seed1023
    │   │   │   ├── seed1023gr.ini
    │   │   │   └── seed1023newton.ini
    │   │   ├── seed1024
    │   │   │   ├── seed1024gr.ini
    │   │   │   └── seed1024newton.ini
    │   │   ├── seed1025
    │   │   │   ├── seed1025gr.ini
    │   │   │   └── seed1025newton.ini
    │   │   ├── seed1026
    │   │   │   ├── seed1026gr.ini
    │   │   │   └── seed1026newton.ini
    │   │   ├── seed1027
    │   │   │   ├── seed1027gr.ini
    │   │   │   └── seed1027newton.ini
    │   │   ├── seed1028
    │   │   │   ├── seed1028gr.ini
    │   │   │   └── seed1028newton.ini
    │   │   ├── seed1029
    │   │   │   ├── seed1029gr.ini
    │   │   │   └── seed1029newton.ini
    │   │   ├── seed1030
    │   │   │   ├── seed1030gr.ini
    │   │   │   └── seed1030newton.ini
    │   │   ├── seed1031
    │   │   │   ├── seed1031gr.ini
    │   │   │   └── seed1031newton.ini
    │   │   ├── seed1032
    │   │   │   ├── seed1032gr.ini
    │   │   │   └── seed1032newton.ini
    │   │   ├── seed1033
    │   │   │   ├── seed1033gr.ini
    │   │   │   └── seed1033newton.ini
    │   │   ├── seed1034
    │   │   │   ├── seed1034gr.ini
    │   │   │   └── seed1034newton.ini
    │   │   ├── seed1035
    │   │   │   ├── seed1035gr.ini
    │   │   │   └── seed1035newton.ini
    │   │   ├── seed1036
    │   │   │   ├── seed1036gr.ini
    │   │   │   └── seed1036newton.ini
    │   │   ├── seed1037
    │   │   │   ├── seed1037gr.ini
    │   │   │   └── seed1037newton.ini
    │   │   ├── seed1038
    │   │   │   ├── seed1038gr.ini
    │   │   │   └── seed1038newton.ini
    │   │   ├── seed1039
    │   │   │   ├── seed1039gr.ini
    │   │   │   └── seed1039newton.ini
    │   │   ├── seed1040
    │   │   │   ├── seed1040gr.ini
    │   │   │   └── seed1040newton.ini
    │   │   ├── seed1041
    │   │   │   ├── seed1041gr.ini
    │   │   │   └── seed1041newton.ini
    │   │   ├── seed1042
    │   │   │   ├── seed1042gr.ini
    │   │   │   └── seed1042newton.ini
    │   │   ├── seed1043
    │   │   │   ├── seed1043gr.ini
    │   │   │   └── seed1043newton.ini
    │   │   ├── seed1044
    │   │   │   ├── seed1044gr.ini
    │   │   │   └── seed1044newton.ini
    │   │   ├── seed1045
    │   │   │   ├── seed1045gr.ini
    │   │   │   └── seed1045newton.ini
    │   │   ├── seed1046
    │   │   │   ├── seed1046gr.ini
    │   │   │   └── seed1046newton.ini
    │   │   ├── seed1047
    │   │   │   ├── seed1047gr.ini
    │   │   │   └── seed1047newton.ini
    │   │   ├── seed1048
    │   │   │   ├── seed1048gr.ini
    │   │   │   └── seed1048newton.ini
    │   │   ├── seed1049
    │   │   │   ├── seed1049gr.ini
    │   │   │   └── seed1049newton.ini
    │   │   ├── seed1050
    │   │   │   ├── seed1050gr.ini
    │   │   │   └── seed1050newton.ini
    │   │   ├── seed1051
    │   │   │   ├── seed1051gr.ini
    │   │   │   └── seed1051newton.ini
    │   │   ├── seed1052
    │   │   │   ├── seed1052gr.ini
    │   │   │   └── seed1052newton.ini
    │   │   ├── seed1053
    │   │   │   ├── seed1053gr.ini
    │   │   │   └── seed1053newton.ini
    │   │   ├── seed1054
    │   │   │   ├── seed1054gr.ini
    │   │   │   └── seed1054newton.ini
    │   │   ├── seed1055
    │   │   │   ├── seed1055gr.ini
    │   │   │   └── seed1055newton.ini
    │   │   ├── seed1056
    │   │   │   ├── seed1056gr.ini
    │   │   │   └── seed1056newton.ini
    │   │   ├── seed1057
    │   │   │   ├── seed1057gr.ini
    │   │   │   └── seed1057newton.ini
    │   │   ├── seed1058
    │   │   │   ├── seed1058gr.ini
    │   │   │   └── seed1058newton.ini
    │   │   ├── seed1059
    │   │   │   ├── seed1059gr.ini
    │   │   │   └── seed1059newton.ini
    │   │   ├── seed1060
    │   │   │   ├── seed1060gr.ini
    │   │   │   └── seed1060newton.ini
    │   │   ├── seed1061
    │   │   │   ├── seed1061gr.ini
    │   │   │   └── seed1061newton.ini
    │   │   ├── seed1062
    │   │   │   ├── seed1062gr.ini
    │   │   │   └── seed1062newton.ini
    │   │   ├── seed1063
    │   │   │   ├── seed1063gr.ini
    │   │   │   └── seed1063newton.ini
    │   │   ├── seed1064
    │   │   │   ├── seed1064gr.ini
    │   │   │   └── seed1064newton.ini
    │   │   ├── seed1065
    │   │   │   ├── seed1065gr.ini
    │   │   │   └── seed1065newton.ini
    │   │   ├── seed1066
    │   │   │   ├── seed1066gr.ini
    │   │   │   └── seed1066newton.ini
    │   │   ├── seed1067
    │   │   │   ├── seed1067gr.ini
    │   │   │   └── seed1067newton.ini
    │   │   ├── seed1068
    │   │   │   ├── seed1068gr.ini
    │   │   │   └── seed1068newton.ini
    │   │   ├── seed1069
    │   │   │   ├── seed1069gr.ini
    │   │   │   └── seed1069newton.ini
    │   │   ├── seed1070
    │   │   │   ├── seed1070gr.ini
    │   │   │   └── seed1070newton.ini
    │   │   ├── seed1071
    │   │   │   ├── seed1071gr.ini
    │   │   │   └── seed1071newton.ini
    │   │   ├── seed1072
    │   │   │   ├── seed1072gr.ini
    │   │   │   └── seed1072newton.ini
    │   │   ├── seed1073
    │   │   │   ├── seed1073gr.ini
    │   │   │   └── seed1073newton.ini
    │   │   ├── seed1074
    │   │   │   ├── seed1074gr.ini
    │   │   │   └── seed1074newton.ini
    │   │   ├── seed1075
    │   │   │   ├── seed1075gr.ini
    │   │   │   └── seed1075newton.ini
    │   │   ├── seed1076
    │   │   │   ├── seed1076gr.ini
    │   │   │   └── seed1076newton.ini
    │   │   ├── seed1077
    │   │   │   ├── seed1077gr.ini
    │   │   │   └── seed1077newton.ini
    │   │   ├── seed1078
    │   │   │   ├── seed1078gr.ini
    │   │   │   └── seed1078newton.ini
    │   │   ├── seed1079
    │   │   │   ├── seed1079gr.ini
    │   │   │   └── seed1079newton.ini
    │   │   ├── seed1080
    │   │   │   ├── seed1080gr.ini
    │   │   │   └── seed1080newton.ini
    │   │   ├── seed1081
    │   │   │   ├── seed1081gr.ini
    │   │   │   └── seed1081newton.ini
    │   │   ├── seed1082
    │   │   │   ├── seed1082gr.ini
    │   │   │   └── seed1082newton.ini
    │   │   ├── seed1083
    │   │   │   ├── seed1083gr.ini
    │   │   │   └── seed1083newton.ini
    │   │   ├── seed1084
    │   │   │   ├── seed1084gr.ini
    │   │   │   └── seed1084newton.ini
    │   │   ├── seed1085
    │   │   │   ├── seed1085gr.ini
    │   │   │   └── seed1085newton.ini
    │   │   ├── seed1086
    │   │   │   ├── seed1086gr.ini
    │   │   │   └── seed1086newton.ini
    │   │   ├── seed1087
    │   │   │   ├── seed1087gr.ini
    │   │   │   └── seed1087newton.ini
    │   │   ├── seed1088
    │   │   │   ├── seed1088gr.ini
    │   │   │   └── seed1088newton.ini
    │   │   ├── seed1089
    │   │   │   ├── seed1089gr.ini
    │   │   │   └── seed1089newton.ini
    │   │   ├── seed1090
    │   │   │   ├── seed1090gr.ini
    │   │   │   └── seed1090newton.ini
    │   │   ├── seed1091
    │   │   │   ├── seed1091gr.ini
    │   │   │   └── seed1091newton.ini
    │   │   ├── seed1092
    │   │   │   ├── seed1092gr.ini
    │   │   │   └── seed1092newton.ini
    │   │   ├── seed1093
    │   │   │   ├── seed1093gr.ini
    │   │   │   └── seed1093newton.ini
    │   │   ├── seed1094
    │   │   │   ├── seed1094gr.ini
    │   │   │   └── seed1094newton.ini
    │   │   ├── seed1095
    │   │   │   ├── seed1095gr.ini
    │   │   │   └── seed1095newton.ini
    │   │   ├── seed1096
    │   │   │   ├── seed1096gr.ini
    │   │   │   └── seed1096newton.ini
    │   │   ├── seed1097
    │   │   │   ├── seed1097gr.ini
    │   │   │   └── seed1097newton.ini
    │   │   ├── seed1098
    │   │   │   ├── seed1098gr.ini
    │   │   │   └── seed1098newton.ini
    │   │   ├── seed1099
    │   │   │   ├── seed1099gr.ini
    │   │   │   └── seed1099newton.ini
    │   │   ├── seed1100
    │   │   │   ├── seed1100gr.ini
    │   │   │   └── seed1100newton.ini
    │   │   ├── seed1101
    │   │   │   ├── seed1101gr.ini
    │   │   │   └── seed1101newton.ini
    │   │   ├── seed1102
    │   │   │   ├── seed1102gr.ini
    │   │   │   └── seed1102newton.ini
    │   │   ├── seed1103
    │   │   │   ├── seed1103gr.ini
    │   │   │   └── seed1103newton.ini
    │   │   ├── seed1104
    │   │   │   ├── seed1104gr.ini
    │   │   │   └── seed1104newton.ini
    │   │   ├── seed1105
    │   │   │   ├── seed1105gr.ini
    │   │   │   └── seed1105newton.ini
    │   │   ├── seed1106
    │   │   │   ├── seed1106gr.ini
    │   │   │   └── seed1106newton.ini
    │   │   ├── seed1107
    │   │   │   ├── seed1107gr.ini
    │   │   │   └── seed1107newton.ini
    │   │   ├── seed1108
    │   │   │   ├── seed1108gr.ini
    │   │   │   └── seed1108newton.ini
    │   │   ├── seed1109
    │   │   │   ├── seed1109gr.ini
    │   │   │   └── seed1109newton.ini
    │   │   ├── seed1110
    │   │   │   ├── seed1110gr.ini
    │   │   │   └── seed1110newton.ini
    │   │   ├── seed1111
    │   │   │   ├── seed1111gr.ini
    │   │   │   └── seed1111newton.ini
    │   │   ├── seed1112
    │   │   │   ├── seed1112gr.ini
    │   │   │   └── seed1112newton.ini
    │   │   ├── seed1113
    │   │   │   ├── seed1113gr.ini
    │   │   │   └── seed1113newton.ini
    │   │   ├── seed1114
    │   │   │   ├── seed1114gr.ini
    │   │   │   └── seed1114newton.ini
    │   │   ├── seed1115
    │   │   │   ├── seed1115gr.ini
    │   │   │   └── seed1115newton.ini
    │   │   ├── seed1116
    │   │   │   ├── seed1116gr.ini
    │   │   │   └── seed1116newton.ini
    │   │   ├── seed1117
    │   │   │   ├── seed1117gr.ini
    │   │   │   └── seed1117newton.ini
    │   │   ├── seed1118
    │   │   │   ├── seed1118gr.ini
    │   │   │   └── seed1118newton.ini
    │   │   ├── seed1119
    │   │   │   ├── seed1119gr.ini
    │   │   │   └── seed1119newton.ini
    │   │   ├── seed1120
    │   │   │   ├── seed1120gr.ini
    │   │   │   └── seed1120newton.ini
    │   │   ├── seed1121
    │   │   │   ├── seed1121gr.ini
    │   │   │   └── seed1121newton.ini
    │   │   ├── seed1122
    │   │   │   ├── seed1122gr.ini
    │   │   │   └── seed1122newton.ini
    │   │   ├── seed1123
    │   │   │   ├── seed1123gr.ini
    │   │   │   └── seed1123newton.ini
    │   │   ├── seed1124
    │   │   │   ├── seed1124gr.ini
    │   │   │   └── seed1124newton.ini
    │   │   ├── seed1125
    │   │   │   ├── seed1125gr.ini
    │   │   │   └── seed1125newton.ini
    │   │   ├── seed1126
    │   │   │   ├── seed1126gr.ini
    │   │   │   └── seed1126newton.ini
    │   │   ├── seed1127
    │   │   │   ├── seed1127gr.ini
    │   │   │   └── seed1127newton.ini
    │   │   ├── seed1128
    │   │   │   ├── seed1128gr.ini
    │   │   │   └── seed1128newton.ini
    │   │   ├── seed1129
    │   │   │   ├── seed1129gr.ini
    │   │   │   └── seed1129newton.ini
    │   │   ├── seed1130
    │   │   │   ├── seed1130gr.ini
    │   │   │   └── seed1130newton.ini
    │   │   ├── seed1131
    │   │   │   ├── seed1131gr.ini
    │   │   │   └── seed1131newton.ini
    │   │   ├── seed1132
    │   │   │   ├── seed1132gr.ini
    │   │   │   └── seed1132newton.ini
    │   │   ├── seed1133
    │   │   │   ├── seed1133gr.ini
    │   │   │   └── seed1133newton.ini
    │   │   ├── seed1134
    │   │   │   ├── seed1134gr.ini
    │   │   │   └── seed1134newton.ini
    │   │   ├── seed1135
    │   │   │   ├── seed1135gr.ini
    │   │   │   └── seed1135newton.ini
    │   │   ├── seed1136
    │   │   │   ├── seed1136gr.ini
    │   │   │   └── seed1136newton.ini
    │   │   ├── seed1137
    │   │   │   ├── seed1137gr.ini
    │   │   │   └── seed1137newton.ini
    │   │   ├── seed1138
    │   │   │   ├── seed1138gr.ini
    │   │   │   └── seed1138newton.ini
    │   │   ├── seed1139
    │   │   │   ├── seed1139gr.ini
    │   │   │   └── seed1139newton.ini
    │   │   ├── seed1140
    │   │   │   ├── seed1140gr.ini
    │   │   │   └── seed1140newton.ini
    │   │   ├── seed1141
    │   │   │   ├── seed1141gr.ini
    │   │   │   └── seed1141newton.ini
    │   │   ├── seed1142
    │   │   │   ├── seed1142gr.ini
    │   │   │   └── seed1142newton.ini
    │   │   ├── seed1143
    │   │   │   ├── seed1143gr.ini
    │   │   │   └── seed1143newton.ini
    │   │   ├── seed1144
    │   │   │   ├── seed1144gr.ini
    │   │   │   └── seed1144newton.ini
    │   │   ├── seed1145
    │   │   │   ├── seed1145gr.ini
    │   │   │   └── seed1145newton.ini
    │   │   ├── seed1146
    │   │   │   ├── seed1146gr.ini
    │   │   │   └── seed1146newton.ini
    │   │   ├── seed1147
    │   │   │   ├── seed1147gr.ini
    │   │   │   └── seed1147newton.ini
    │   │   ├── seed1148
    │   │   │   ├── seed1148gr.ini
    │   │   │   └── seed1148newton.ini
    │   │   ├── seed1149
    │   │   │   ├── seed1149gr.ini
    │   │   │   └── seed1149newton.ini
    │   │   ├── seed1150
    │   │   │   ├── seed1150gr.ini
    │   │   │   └── seed1150newton.ini
    │   │   ├── seed1151
    │   │   │   ├── seed1151gr.ini
    │   │   │   └── seed1151newton.ini
    │   │   ├── seed1152
    │   │   │   ├── seed1152gr.ini
    │   │   │   └── seed1152newton.ini
    │   │   ├── seed1153
    │   │   │   ├── seed1153gr.ini
    │   │   │   └── seed1153newton.ini
    │   │   ├── seed1154
    │   │   │   ├── seed1154gr.ini
    │   │   │   └── seed1154newton.ini
    │   │   ├── seed1155
    │   │   │   ├── seed1155gr.ini
    │   │   │   └── seed1155newton.ini
    │   │   ├── seed1156
    │   │   │   ├── seed1156gr.ini
    │   │   │   └── seed1156newton.ini
    │   │   ├── seed1157
    │   │   │   ├── seed1157gr.ini
    │   │   │   └── seed1157newton.ini
    │   │   ├── seed1158
    │   │   │   ├── seed1158gr.ini
    │   │   │   └── seed1158newton.ini
    │   │   ├── seed1159
    │   │   │   ├── seed1159gr.ini
    │   │   │   └── seed1159newton.ini
    │   │   ├── seed1160
    │   │   │   ├── seed1160gr.ini
    │   │   │   └── seed1160newton.ini
    │   │   ├── seed1161
    │   │   │   ├── seed1161gr.ini
    │   │   │   └── seed1161newton.ini
    │   │   ├── seed1162
    │   │   │   ├── seed1162gr.ini
    │   │   │   └── seed1162newton.ini
    │   │   ├── seed1163
    │   │   │   ├── seed1163gr.ini
    │   │   │   └── seed1163newton.ini
    │   │   ├── seed1164
    │   │   │   ├── seed1164gr.ini
    │   │   │   └── seed1164newton.ini
    │   │   ├── seed1165
    │   │   │   ├── seed1165gr.ini
    │   │   │   └── seed1165newton.ini
    │   │   ├── seed1166
    │   │   │   ├── seed1166gr.ini
    │   │   │   └── seed1166newton.ini
    │   │   ├── seed1167
    │   │   │   ├── seed1167gr.ini
    │   │   │   └── seed1167newton.ini
    │   │   ├── seed1168
    │   │   │   ├── seed1168gr.ini
    │   │   │   └── seed1168newton.ini
    │   │   ├── seed1169
    │   │   │   ├── seed1169gr.ini
    │   │   │   └── seed1169newton.ini
    │   │   ├── seed1170
    │   │   │   ├── seed1170gr.ini
    │   │   │   └── seed1170newton.ini
    │   │   ├── seed1171
    │   │   │   ├── seed1171gr.ini
    │   │   │   └── seed1171newton.ini
    │   │   ├── seed1172
    │   │   │   ├── seed1172gr.ini
    │   │   │   └── seed1172newton.ini
    │   │   ├── seed1173
    │   │   │   ├── seed1173gr.ini
    │   │   │   └── seed1173newton.ini
    │   │   ├── seed1174
    │   │   │   ├── seed1174gr.ini
    │   │   │   └── seed1174newton.ini
    │   │   ├── seed1175
    │   │   │   ├── seed1175gr.ini
    │   │   │   └── seed1175newton.ini
    │   │   ├── seed1176
    │   │   │   ├── seed1176gr.ini
    │   │   │   └── seed1176newton.ini
    │   │   ├── seed1177
    │   │   │   ├── seed1177gr.ini
    │   │   │   └── seed1177newton.ini
    │   │   ├── seed1178
    │   │   │   ├── seed1178gr.ini
    │   │   │   └── seed1178newton.ini
    │   │   ├── seed1179
    │   │   │   ├── seed1179gr.ini
    │   │   │   └── seed1179newton.ini
    │   │   ├── seed1180
    │   │   │   ├── seed1180gr.ini
    │   │   │   └── seed1180newton.ini
    │   │   ├── seed1181
    │   │   │   ├── seed1181gr.ini
    │   │   │   └── seed1181newton.ini
    │   │   ├── seed1182
    │   │   │   ├── seed1182gr.ini
    │   │   │   └── seed1182newton.ini
    │   │   ├── seed1183
    │   │   │   ├── seed1183gr.ini
    │   │   │   └── seed1183newton.ini
    │   │   ├── seed1184
    │   │   │   ├── seed1184gr.ini
    │   │   │   └── seed1184newton.ini
    │   │   ├── seed1185
    │   │   │   ├── seed1185gr.ini
    │   │   │   └── seed1185newton.ini
    │   │   ├── seed1186
    │   │   │   ├── seed1186gr.ini
    │   │   │   └── seed1186newton.ini
    │   │   ├── seed1187
    │   │   │   ├── seed1187gr.ini
    │   │   │   └── seed1187newton.ini
    │   │   ├── seed1188
    │   │   │   ├── seed1188gr.ini
    │   │   │   └── seed1188newton.ini
    │   │   ├── seed1189
    │   │   │   ├── seed1189gr.ini
    │   │   │   └── seed1189newton.ini
    │   │   ├── seed1190
    │   │   │   ├── seed1190gr.ini
    │   │   │   └── seed1190newton.ini
    │   │   ├── seed1191
    │   │   │   ├── seed1191gr.ini
    │   │   │   └── seed1191newton.ini
    │   │   ├── seed1192
    │   │   │   ├── seed1192gr.ini
    │   │   │   └── seed1192newton.ini
    │   │   ├── seed1193
    │   │   │   ├── seed1193gr.ini
    │   │   │   └── seed1193newton.ini
    │   │   ├── seed1194
    │   │   │   ├── seed1194gr.ini
    │   │   │   └── seed1194newton.ini
    │   │   ├── seed1195
    │   │   │   ├── seed1195gr.ini
    │   │   │   └── seed1195newton.ini
    │   │   ├── seed1196
    │   │   │   ├── seed1196gr.ini
    │   │   │   └── seed1196newton.ini
    │   │   ├── seed1197
    │   │   │   ├── seed1197gr.ini
    │   │   │   └── seed1197newton.ini
    │   │   ├── seed1198
    │   │   │   ├── seed1198gr.ini
    │   │   │   └── seed1198newton.ini
    │   │   ├── seed1199
    │   │   │   ├── seed1199gr.ini
    │   │   │   └── seed1199newton.ini
    │   │   ├── seed1200
    │   │   │   ├── seed1200gr.ini
    │   │   │   └── seed1200newton.ini
    │   │   ├── seed1201
    │   │   │   ├── seed1201gr.ini
    │   │   │   └── seed1201newton.ini
    │   │   ├── seed1202
    │   │   │   ├── seed1202gr.ini
    │   │   │   └── seed1202newton.ini
    │   │   ├── seed1203
    │   │   │   ├── seed1203gr.ini
    │   │   │   └── seed1203newton.ini
    │   │   ├── seed1204
    │   │   │   ├── seed1204gr.ini
    │   │   │   └── seed1204newton.ini
    │   │   ├── seed1205
    │   │   │   ├── seed1205gr.ini
    │   │   │   └── seed1205newton.ini
    │   │   ├── seed1206
    │   │   │   ├── seed1206gr.ini
    │   │   │   └── seed1206newton.ini
    │   │   ├── seed1207
    │   │   │   ├── seed1207gr.ini
    │   │   │   └── seed1207newton.ini
    │   │   ├── seed1208
    │   │   │   ├── seed1208gr.ini
    │   │   │   └── seed1208newton.ini
    │   │   ├── seed1209
    │   │   │   ├── seed1209gr.ini
    │   │   │   └── seed1209newton.ini
    │   │   ├── seed1210
    │   │   │   ├── seed1210gr.ini
    │   │   │   └── seed1210newton.ini
    │   │   ├── seed1211
    │   │   │   ├── seed1211gr.ini
    │   │   │   └── seed1211newton.ini
    │   │   ├── seed1212
    │   │   │   ├── seed1212gr.ini
    │   │   │   └── seed1212newton.ini
    │   │   ├── seed1213
    │   │   │   ├── seed1213gr.ini
    │   │   │   └── seed1213newton.ini
    │   │   ├── seed1214
    │   │   │   ├── seed1214gr.ini
    │   │   │   └── seed1214newton.ini
    │   │   ├── seed1215
    │   │   │   ├── seed1215gr.ini
    │   │   │   └── seed1215newton.ini
    │   │   ├── seed1216
    │   │   │   ├── seed1216gr.ini
    │   │   │   └── seed1216newton.ini
    │   │   ├── seed1217
    │   │   │   ├── seed1217gr.ini
    │   │   │   └── seed1217newton.ini
    │   │   ├── seed1218
    │   │   │   ├── seed1218gr.ini
    │   │   │   └── seed1218newton.ini
    │   │   ├── seed1219
    │   │   │   ├── seed1219gr.ini
    │   │   │   └── seed1219newton.ini
    │   │   ├── seed1220
    │   │   │   ├── seed1220gr.ini
    │   │   │   └── seed1220newton.ini
    │   │   ├── seed1221
    │   │   │   ├── seed1221gr.ini
    │   │   │   └── seed1221newton.ini
    │   │   ├── seed1222
    │   │   │   ├── seed1222gr.ini
    │   │   │   └── seed1222newton.ini
    │   │   ├── seed1223
    │   │   │   ├── seed1223gr.ini
    │   │   │   └── seed1223newton.ini
    │   │   ├── seed1224
    │   │   │   ├── seed1224gr.ini
    │   │   │   └── seed1224newton.ini
    │   │   ├── seed1225
    │   │   │   ├── seed1225gr.ini
    │   │   │   └── seed1225newton.ini
    │   │   ├── seed1226
    │   │   │   ├── seed1226gr.ini
    │   │   │   └── seed1226newton.ini
    │   │   ├── seed1227
    │   │   │   ├── seed1227gr.ini
    │   │   │   └── seed1227newton.ini
    │   │   ├── seed1228
    │   │   │   ├── seed1228gr.ini
    │   │   │   └── seed1228newton.ini
    │   │   ├── seed1229
    │   │   │   ├── seed1229gr.ini
    │   │   │   └── seed1229newton.ini
    │   │   ├── seed1230
    │   │   │   ├── seed1230gr.ini
    │   │   │   └── seed1230newton.ini
    │   │   ├── seed1231
    │   │   │   ├── seed1231gr.ini
    │   │   │   └── seed1231newton.ini
    │   │   ├── seed1232
    │   │   │   ├── seed1232gr.ini
    │   │   │   └── seed1232newton.ini
    │   │   ├── seed1233
    │   │   │   ├── seed1233gr.ini
    │   │   │   └── seed1233newton.ini
    │   │   ├── seed1234
    │   │   │   ├── seed1234gr.ini
    │   │   │   └── seed1234newton.ini
    │   │   ├── seed1235
    │   │   │   ├── seed1235gr.ini
    │   │   │   └── seed1235newton.ini
    │   │   ├── seed1236
    │   │   │   ├── seed1236gr.ini
    │   │   │   └── seed1236newton.ini
    │   │   ├── seed1237
    │   │   │   ├── seed1237gr.ini
    │   │   │   └── seed1237newton.ini
    │   │   ├── seed1238
    │   │   │   ├── seed1238gr.ini
    │   │   │   └── seed1238newton.ini
    │   │   ├── seed1239
    │   │   │   ├── seed1239gr.ini
    │   │   │   └── seed1239newton.ini
    │   │   ├── seed1240
    │   │   │   ├── seed1240gr.ini
    │   │   │   └── seed1240newton.ini
    │   │   ├── seed1241
    │   │   │   ├── seed1241gr.ini
    │   │   │   └── seed1241newton.ini
    │   │   ├── seed1242
    │   │   │   ├── seed1242gr.ini
    │   │   │   └── seed1242newton.ini
    │   │   ├── seed1243
    │   │   │   ├── seed1243gr.ini
    │   │   │   └── seed1243newton.ini
    │   │   ├── seed1244
    │   │   │   ├── seed1244gr.ini
    │   │   │   └── seed1244newton.ini
    │   │   ├── seed1245
    │   │   │   ├── seed1245gr.ini
    │   │   │   └── seed1245newton.ini
    │   │   ├── seed1246
    │   │   │   ├── seed1246gr.ini
    │   │   │   └── seed1246newton.ini
    │   │   ├── seed1247
    │   │   │   ├── seed1247gr.ini
    │   │   │   └── seed1247newton.ini
    │   │   ├── seed1248
    │   │   │   ├── seed1248gr.ini
    │   │   │   └── seed1248newton.ini
    │   │   ├── seed1249
    │   │   │   ├── seed1249gr.ini
    │   │   │   └── seed1249newton.ini
    │   │   ├── seed1250
    │   │   │   ├── seed1250gr.ini
    │   │   │   └── seed1250newton.ini
    │   │   ├── seed1251
    │   │   │   ├── seed1251gr.ini
    │   │   │   └── seed1251newton.ini
    │   │   ├── seed1252
    │   │   │   ├── seed1252gr.ini
    │   │   │   └── seed1252newton.ini
    │   │   ├── seed1253
    │   │   │   ├── seed1253gr.ini
    │   │   │   └── seed1253newton.ini
    │   │   ├── seed1254
    │   │   │   ├── seed1254gr.ini
    │   │   │   └── seed1254newton.ini
    │   │   ├── seed1255
    │   │   │   ├── seed1255gr.ini
    │   │   │   └── seed1255newton.ini
    │   │   ├── seed1256
    │   │   │   ├── seed1256gr.ini
    │   │   │   └── seed1256newton.ini
    │   │   ├── seed1257
    │   │   │   ├── seed1257gr.ini
    │   │   │   └── seed1257newton.ini
    │   │   ├── seed1258
    │   │   │   ├── seed1258gr.ini
    │   │   │   └── seed1258newton.ini
    │   │   ├── seed1259
    │   │   │   ├── seed1259gr.ini
    │   │   │   └── seed1259newton.ini
    │   │   ├── seed1260
    │   │   │   ├── seed1260gr.ini
    │   │   │   └── seed1260newton.ini
    │   │   ├── seed1261
    │   │   │   ├── seed1261gr.ini
    │   │   │   └── seed1261newton.ini
    │   │   ├── seed1262
    │   │   │   ├── seed1262gr.ini
    │   │   │   └── seed1262newton.ini
    │   │   ├── seed1263
    │   │   │   ├── seed1263gr.ini
    │   │   │   └── seed1263newton.ini
    │   │   ├── seed1264
    │   │   │   ├── seed1264gr.ini
    │   │   │   └── seed1264newton.ini
    │   │   ├── seed1265
    │   │   │   ├── seed1265gr.ini
    │   │   │   └── seed1265newton.ini
    │   │   ├── seed1266
    │   │   │   ├── seed1266gr.ini
    │   │   │   └── seed1266newton.ini
    │   │   ├── seed1267
    │   │   │   ├── seed1267gr.ini
    │   │   │   └── seed1267newton.ini
    │   │   ├── seed1268
    │   │   │   ├── seed1268gr.ini
    │   │   │   └── seed1268newton.ini
    │   │   ├── seed1269
    │   │   │   ├── seed1269gr.ini
    │   │   │   └── seed1269newton.ini
    │   │   ├── seed1270
    │   │   │   ├── seed1270gr.ini
    │   │   │   └── seed1270newton.ini
    │   │   ├── seed1271
    │   │   │   ├── seed1271gr.ini
    │   │   │   └── seed1271newton.ini
    │   │   ├── seed1272
    │   │   │   ├── seed1272gr.ini
    │   │   │   └── seed1272newton.ini
    │   │   ├── seed1273
    │   │   │   ├── seed1273gr.ini
    │   │   │   └── seed1273newton.ini
    │   │   ├── seed1274
    │   │   │   ├── seed1274gr.ini
    │   │   │   └── seed1274newton.ini
    │   │   ├── seed1275
    │   │   │   ├── seed1275gr.ini
    │   │   │   └── seed1275newton.ini
    │   │   ├── seed1276
    │   │   │   ├── seed1276gr.ini
    │   │   │   └── seed1276newton.ini
    │   │   ├── seed1277
    │   │   │   ├── seed1277gr.ini
    │   │   │   └── seed1277newton.ini
    │   │   ├── seed1278
    │   │   │   ├── seed1278gr.ini
    │   │   │   └── seed1278newton.ini
    │   │   ├── seed1279
    │   │   │   ├── seed1279gr.ini
    │   │   │   └── seed1279newton.ini
    │   │   ├── seed1280
    │   │   │   ├── seed1280gr.ini
    │   │   │   └── seed1280newton.ini
    │   │   ├── seed1281
    │   │   │   ├── seed1281gr.ini
    │   │   │   └── seed1281newton.ini
    │   │   ├── seed1282
    │   │   │   ├── seed1282gr.ini
    │   │   │   └── seed1282newton.ini
    │   │   ├── seed1283
    │   │   │   ├── seed1283gr.ini
    │   │   │   └── seed1283newton.ini
    │   │   ├── seed1284
    │   │   │   ├── seed1284gr.ini
    │   │   │   └── seed1284newton.ini
    │   │   ├── seed1285
    │   │   │   ├── seed1285gr.ini
    │   │   │   └── seed1285newton.ini
    │   │   ├── seed1286
    │   │   │   ├── seed1286gr.ini
    │   │   │   └── seed1286newton.ini
    │   │   ├── seed1287
    │   │   │   ├── seed1287gr.ini
    │   │   │   └── seed1287newton.ini
    │   │   ├── seed1288
    │   │   │   ├── seed1288gr.ini
    │   │   │   └── seed1288newton.ini
    │   │   ├── seed1289
    │   │   │   ├── seed1289gr.ini
    │   │   │   └── seed1289newton.ini
    │   │   ├── seed1290
    │   │   │   ├── seed1290gr.ini
    │   │   │   └── seed1290newton.ini
    │   │   ├── seed1291
    │   │   │   ├── seed1291gr.ini
    │   │   │   └── seed1291newton.ini
    │   │   ├── seed1292
    │   │   │   ├── seed1292gr.ini
    │   │   │   └── seed1292newton.ini
    │   │   ├── seed1293
    │   │   │   ├── seed1293gr.ini
    │   │   │   └── seed1293newton.ini
    │   │   ├── seed1294
    │   │   │   ├── seed1294gr.ini
    │   │   │   └── seed1294newton.ini
    │   │   ├── seed1295
    │   │   │   ├── seed1295gr.ini
    │   │   │   └── seed1295newton.ini
    │   │   ├── seed1296
    │   │   │   ├── seed1296gr.ini
    │   │   │   └── seed1296newton.ini
    │   │   ├── seed1297
    │   │   │   ├── seed1297gr.ini
    │   │   │   └── seed1297newton.ini
    │   │   ├── seed1298
    │   │   │   ├── seed1298gr.ini
    │   │   │   └── seed1298newton.ini
    │   │   ├── seed1299
    │   │   │   ├── seed1299gr.ini
    │   │   │   └── seed1299newton.ini
    │   │   ├── seed1300
    │   │   │   ├── seed1300gr.ini
    │   │   │   └── seed1300newton.ini
    │   │   ├── seed1301
    │   │   │   ├── seed1301gr.ini
    │   │   │   └── seed1301newton.ini
    │   │   ├── seed1302
    │   │   │   ├── seed1302gr.ini
    │   │   │   └── seed1302newton.ini
    │   │   ├── seed1303
    │   │   │   ├── seed1303gr.ini
    │   │   │   └── seed1303newton.ini
    │   │   ├── seed1304
    │   │   │   ├── seed1304gr.ini
    │   │   │   └── seed1304newton.ini
    │   │   ├── seed1305
    │   │   │   ├── seed1305gr.ini
    │   │   │   └── seed1305newton.ini
    │   │   ├── seed1306
    │   │   │   ├── seed1306gr.ini
    │   │   │   └── seed1306newton.ini
    │   │   ├── seed1307
    │   │   │   ├── seed1307gr.ini
    │   │   │   └── seed1307newton.ini
    │   │   ├── seed1308
    │   │   │   ├── seed1308gr.ini
    │   │   │   └── seed1308newton.ini
    │   │   ├── seed1309
    │   │   │   ├── seed1309gr.ini
    │   │   │   └── seed1309newton.ini
    │   │   ├── seed1310
    │   │   │   ├── seed1310gr.ini
    │   │   │   └── seed1310newton.ini
    │   │   ├── seed1311
    │   │   │   ├── seed1311gr.ini
    │   │   │   └── seed1311newton.ini
    │   │   ├── seed1312
    │   │   │   ├── seed1312gr.ini
    │   │   │   └── seed1312newton.ini
    │   │   ├── seed1313
    │   │   │   ├── seed1313gr.ini
    │   │   │   └── seed1313newton.ini
    │   │   ├── seed1314
    │   │   │   ├── seed1314gr.ini
    │   │   │   └── seed1314newton.ini
    │   │   ├── seed1315
    │   │   │   ├── seed1315gr.ini
    │   │   │   └── seed1315newton.ini
    │   │   ├── seed1316
    │   │   │   ├── seed1316gr.ini
    │   │   │   └── seed1316newton.ini
    │   │   ├── seed1317
    │   │   │   ├── seed1317gr.ini
    │   │   │   └── seed1317newton.ini
    │   │   ├── seed1318
    │   │   │   ├── seed1318gr.ini
    │   │   │   └── seed1318newton.ini
    │   │   ├── seed1319
    │   │   │   ├── seed1319gr.ini
    │   │   │   └── seed1319newton.ini
    │   │   ├── seed1320
    │   │   │   ├── seed1320gr.ini
    │   │   │   └── seed1320newton.ini
    │   │   ├── seed1321
    │   │   │   ├── seed1321gr.ini
    │   │   │   └── seed1321newton.ini
    │   │   ├── seed1322
    │   │   │   ├── seed1322gr.ini
    │   │   │   └── seed1322newton.ini
    │   │   ├── seed1323
    │   │   │   ├── seed1323gr.ini
    │   │   │   └── seed1323newton.ini
    │   │   ├── seed1324
    │   │   │   ├── seed1324gr.ini
    │   │   │   └── seed1324newton.ini
    │   │   ├── seed1325
    │   │   │   ├── seed1325gr.ini
    │   │   │   └── seed1325newton.ini
    │   │   ├── seed1326
    │   │   │   ├── seed1326gr.ini
    │   │   │   └── seed1326newton.ini
    │   │   ├── seed1327
    │   │   │   ├── seed1327gr.ini
    │   │   │   └── seed1327newton.ini
    │   │   ├── seed1328
    │   │   │   ├── seed1328gr.ini
    │   │   │   └── seed1328newton.ini
    │   │   ├── seed1329
    │   │   │   ├── seed1329gr.ini
    │   │   │   └── seed1329newton.ini
    │   │   ├── seed1330
    │   │   │   ├── seed1330gr.ini
    │   │   │   └── seed1330newton.ini
    │   │   ├── seed1331
    │   │   │   ├── seed1331gr.ini
    │   │   │   └── seed1331newton.ini
    │   │   ├── seed1332
    │   │   │   ├── seed1332gr.ini
    │   │   │   └── seed1332newton.ini
    │   │   ├── seed1333
    │   │   │   ├── seed1333gr.ini
    │   │   │   └── seed1333newton.ini
    │   │   ├── seed1334
    │   │   │   ├── seed1334gr.ini
    │   │   │   └── seed1334newton.ini
    │   │   ├── seed1335
    │   │   │   ├── seed1335gr.ini
    │   │   │   └── seed1335newton.ini
    │   │   ├── seed1336
    │   │   │   ├── seed1336gr.ini
    │   │   │   └── seed1336newton.ini
    │   │   ├── seed1337
    │   │   │   ├── seed1337gr.ini
    │   │   │   └── seed1337newton.ini
    │   │   ├── seed1338
    │   │   │   ├── seed1338gr.ini
    │   │   │   └── seed1338newton.ini
    │   │   ├── seed1339
    │   │   │   ├── seed1339gr.ini
    │   │   │   └── seed1339newton.ini
    │   │   ├── seed1340
    │   │   │   ├── seed1340gr.ini
    │   │   │   └── seed1340newton.ini
    │   │   ├── seed1341
    │   │   │   ├── seed1341gr.ini
    │   │   │   └── seed1341newton.ini
    │   │   ├── seed1342
    │   │   │   ├── seed1342gr.ini
    │   │   │   └── seed1342newton.ini
    │   │   ├── seed1343
    │   │   │   ├── seed1343gr.ini
    │   │   │   └── seed1343newton.ini
    │   │   ├── seed1344
    │   │   │   ├── seed1344gr.ini
    │   │   │   └── seed1344newton.ini
    │   │   ├── seed1345
    │   │   │   ├── seed1345gr.ini
    │   │   │   └── seed1345newton.ini
    │   │   ├── seed1346
    │   │   │   ├── seed1346gr.ini
    │   │   │   └── seed1346newton.ini
    │   │   ├── seed1347
    │   │   │   ├── seed1347gr.ini
    │   │   │   └── seed1347newton.ini
    │   │   ├── seed1348
    │   │   │   ├── seed1348gr.ini
    │   │   │   └── seed1348newton.ini
    │   │   ├── seed1349
    │   │   │   ├── seed1349gr.ini
    │   │   │   └── seed1349newton.ini
    │   │   ├── seed1350
    │   │   │   ├── seed1350gr.ini
    │   │   │   └── seed1350newton.ini
    │   │   ├── seed1351
    │   │   │   ├── seed1351gr.ini
    │   │   │   └── seed1351newton.ini
    │   │   ├── seed1352
    │   │   │   ├── seed1352gr.ini
    │   │   │   └── seed1352newton.ini
    │   │   ├── seed1353
    │   │   │   ├── seed1353gr.ini
    │   │   │   └── seed1353newton.ini
    │   │   ├── seed1354
    │   │   │   ├── seed1354gr.ini
    │   │   │   └── seed1354newton.ini
    │   │   ├── seed1355
    │   │   │   ├── seed1355gr.ini
    │   │   │   └── seed1355newton.ini
    │   │   ├── seed1356
    │   │   │   ├── seed1356gr.ini
    │   │   │   └── seed1356newton.ini
    │   │   ├── seed1357
    │   │   │   ├── seed1357gr.ini
    │   │   │   └── seed1357newton.ini
    │   │   ├── seed1358
    │   │   │   ├── seed1358gr.ini
    │   │   │   └── seed1358newton.ini
    │   │   ├── seed1359
    │   │   │   ├── seed1359gr.ini
    │   │   │   └── seed1359newton.ini
    │   │   ├── seed1360
    │   │   │   ├── seed1360gr.ini
    │   │   │   └── seed1360newton.ini
    │   │   ├── seed1361
    │   │   │   ├── seed1361gr.ini
    │   │   │   └── seed1361newton.ini
    │   │   ├── seed1362
    │   │   │   ├── seed1362gr.ini
    │   │   │   └── seed1362newton.ini
    │   │   ├── seed1363
    │   │   │   ├── seed1363gr.ini
    │   │   │   └── seed1363newton.ini
    │   │   ├── seed1364
    │   │   │   ├── seed1364gr.ini
    │   │   │   └── seed1364newton.ini
    │   │   ├── seed1365
    │   │   │   ├── seed1365gr.ini
    │   │   │   └── seed1365newton.ini
    │   │   ├── seed1366
    │   │   │   ├── seed1366gr.ini
    │   │   │   └── seed1366newton.ini
    │   │   ├── seed1367
    │   │   │   ├── seed1367gr.ini
    │   │   │   └── seed1367newton.ini
    │   │   ├── seed1368
    │   │   │   ├── seed1368gr.ini
    │   │   │   └── seed1368newton.ini
    │   │   ├── seed1369
    │   │   │   ├── seed1369gr.ini
    │   │   │   └── seed1369newton.ini
    │   │   ├── seed1370
    │   │   │   ├── seed1370gr.ini
    │   │   │   └── seed1370newton.ini
    │   │   ├── seed1371
    │   │   │   ├── seed1371gr.ini
    │   │   │   └── seed1371newton.ini
    │   │   ├── seed1372
    │   │   │   ├── seed1372gr.ini
    │   │   │   └── seed1372newton.ini
    │   │   ├── seed1373
    │   │   │   ├── seed1373gr.ini
    │   │   │   └── seed1373newton.ini
    │   │   ├── seed1374
    │   │   │   ├── seed1374gr.ini
    │   │   │   └── seed1374newton.ini
    │   │   ├── seed1375
    │   │   │   ├── seed1375gr.ini
    │   │   │   └── seed1375newton.ini
    │   │   ├── seed1376
    │   │   │   ├── seed1376gr.ini
    │   │   │   └── seed1376newton.ini
    │   │   ├── seed1377
    │   │   │   ├── seed1377gr.ini
    │   │   │   └── seed1377newton.ini
    │   │   ├── seed1378
    │   │   │   ├── seed1378gr.ini
    │   │   │   └── seed1378newton.ini
    │   │   ├── seed1379
    │   │   │   ├── seed1379gr.ini
    │   │   │   └── seed1379newton.ini
    │   │   ├── seed1380
    │   │   │   ├── seed1380gr.ini
    │   │   │   └── seed1380newton.ini
    │   │   ├── seed1381
    │   │   │   ├── seed1381gr.ini
    │   │   │   └── seed1381newton.ini
    │   │   ├── seed1382
    │   │   │   ├── seed1382gr.ini
    │   │   │   └── seed1382newton.ini
    │   │   ├── seed1383
    │   │   │   ├── seed1383gr.ini
    │   │   │   └── seed1383newton.ini
    │   │   ├── seed1384
    │   │   │   ├── seed1384gr.ini
    │   │   │   └── seed1384newton.ini
    │   │   ├── seed1385
    │   │   │   ├── seed1385gr.ini
    │   │   │   └── seed1385newton.ini
    │   │   ├── seed1386
    │   │   │   ├── seed1386gr.ini
    │   │   │   └── seed1386newton.ini
    │   │   ├── seed1387
    │   │   │   ├── seed1387gr.ini
    │   │   │   └── seed1387newton.ini
    │   │   ├── seed1388
    │   │   │   ├── seed1388gr.ini
    │   │   │   └── seed1388newton.ini
    │   │   ├── seed1389
    │   │   │   ├── seed1389gr.ini
    │   │   │   └── seed1389newton.ini
    │   │   ├── seed1390
    │   │   │   ├── seed1390gr.ini
    │   │   │   └── seed1390newton.ini
    │   │   ├── seed1391
    │   │   │   ├── seed1391gr.ini
    │   │   │   └── seed1391newton.ini
    │   │   ├── seed1392
    │   │   │   ├── seed1392gr.ini
    │   │   │   └── seed1392newton.ini
    │   │   ├── seed1393
    │   │   │   ├── seed1393gr.ini
    │   │   │   └── seed1393newton.ini
    │   │   ├── seed1394
    │   │   │   ├── seed1394gr.ini
    │   │   │   └── seed1394newton.ini
    │   │   ├── seed1395
    │   │   │   ├── seed1395gr.ini
    │   │   │   └── seed1395newton.ini
    │   │   ├── seed1396
    │   │   │   ├── seed1396gr.ini
    │   │   │   └── seed1396newton.ini
    │   │   ├── seed1397
    │   │   │   ├── seed1397gr.ini
    │   │   │   └── seed1397newton.ini
    │   │   ├── seed1398
    │   │   │   ├── seed1398gr.ini
    │   │   │   └── seed1398newton.ini
    │   │   ├── seed1399
    │   │   │   ├── seed1399gr.ini
    │   │   │   └── seed1399newton.ini
    │   │   ├── seed1400
    │   │   │   ├── seed1400gr.ini
    │   │   │   └── seed1400newton.ini
    │   │   ├── seed1401
    │   │   │   ├── seed1401gr.ini
    │   │   │   └── seed1401newton.ini
    │   │   ├── seed1402
    │   │   │   ├── seed1402gr.ini
    │   │   │   └── seed1402newton.ini
    │   │   ├── seed1403
    │   │   │   ├── seed1403gr.ini
    │   │   │   └── seed1403newton.ini
    │   │   ├── seed1404
    │   │   │   ├── seed1404gr.ini
    │   │   │   └── seed1404newton.ini
    │   │   ├── seed1405
    │   │   │   ├── seed1405gr.ini
    │   │   │   └── seed1405newton.ini
    │   │   ├── seed1406
    │   │   │   ├── seed1406gr.ini
    │   │   │   └── seed1406newton.ini
    │   │   ├── seed1407
    │   │   │   ├── seed1407gr.ini
    │   │   │   └── seed1407newton.ini
    │   │   ├── seed1408
    │   │   │   ├── seed1408gr.ini
    │   │   │   └── seed1408newton.ini
    │   │   ├── seed1409
    │   │   │   ├── seed1409gr.ini
    │   │   │   └── seed1409newton.ini
    │   │   ├── seed1410
    │   │   │   ├── seed1410gr.ini
    │   │   │   └── seed1410newton.ini
    │   │   ├── seed1411
    │   │   │   ├── seed1411gr.ini
    │   │   │   └── seed1411newton.ini
    │   │   ├── seed1412
    │   │   │   ├── seed1412gr.ini
    │   │   │   └── seed1412newton.ini
    │   │   ├── seed1413
    │   │   │   ├── seed1413gr.ini
    │   │   │   └── seed1413newton.ini
    │   │   ├── seed1414
    │   │   │   ├── seed1414gr.ini
    │   │   │   └── seed1414newton.ini
    │   │   ├── seed1415
    │   │   │   ├── seed1415gr.ini
    │   │   │   └── seed1415newton.ini
    │   │   ├── seed1416
    │   │   │   ├── seed1416gr.ini
    │   │   │   └── seed1416newton.ini
    │   │   ├── seed1417
    │   │   │   ├── seed1417gr.ini
    │   │   │   └── seed1417newton.ini
    │   │   ├── seed1418
    │   │   │   ├── seed1418gr.ini
    │   │   │   └── seed1418newton.ini
    │   │   ├── seed1419
    │   │   │   ├── seed1419gr.ini
    │   │   │   └── seed1419newton.ini
    │   │   ├── seed1420
    │   │   │   ├── seed1420gr.ini
    │   │   │   └── seed1420newton.ini
    │   │   ├── seed1421
    │   │   │   ├── seed1421gr.ini
    │   │   │   └── seed1421newton.ini
    │   │   ├── seed1422
    │   │   │   ├── seed1422gr.ini
    │   │   │   └── seed1422newton.ini
    │   │   ├── seed1423
    │   │   │   ├── seed1423gr.ini
    │   │   │   └── seed1423newton.ini
    │   │   ├── seed1424
    │   │   │   ├── seed1424gr.ini
    │   │   │   └── seed1424newton.ini
    │   │   ├── seed1425
    │   │   │   ├── seed1425gr.ini
    │   │   │   └── seed1425newton.ini
    │   │   ├── seed1426
    │   │   │   ├── seed1426gr.ini
    │   │   │   └── seed1426newton.ini
    │   │   ├── seed1427
    │   │   │   ├── seed1427gr.ini
    │   │   │   └── seed1427newton.ini
    │   │   ├── seed1428
    │   │   │   ├── seed1428gr.ini
    │   │   │   └── seed1428newton.ini
    │   │   ├── seed1429
    │   │   │   ├── seed1429gr.ini
    │   │   │   └── seed1429newton.ini
    │   │   ├── seed1430
    │   │   │   ├── seed1430gr.ini
    │   │   │   └── seed1430newton.ini
    │   │   ├── seed1431
    │   │   │   ├── seed1431gr.ini
    │   │   │   └── seed1431newton.ini
    │   │   ├── seed1432
    │   │   │   ├── seed1432gr.ini
    │   │   │   └── seed1432newton.ini
    │   │   ├── seed1433
    │   │   │   ├── seed1433gr.ini
    │   │   │   └── seed1433newton.ini
    │   │   ├── seed1434
    │   │   │   ├── seed1434gr.ini
    │   │   │   └── seed1434newton.ini
    │   │   ├── seed1435
    │   │   │   ├── seed1435gr.ini
    │   │   │   └── seed1435newton.ini
    │   │   ├── seed1436
    │   │   │   ├── seed1436gr.ini
    │   │   │   └── seed1436newton.ini
    │   │   ├── seed1437
    │   │   │   ├── seed1437gr.ini
    │   │   │   └── seed1437newton.ini
    │   │   ├── seed1438
    │   │   │   ├── seed1438gr.ini
    │   │   │   └── seed1438newton.ini
    │   │   ├── seed1439
    │   │   │   ├── seed1439gr.ini
    │   │   │   └── seed1439newton.ini
    │   │   ├── seed1440
    │   │   │   ├── seed1440gr.ini
    │   │   │   └── seed1440newton.ini
    │   │   ├── seed1441
    │   │   │   ├── seed1441gr.ini
    │   │   │   └── seed1441newton.ini
    │   │   ├── seed1442
    │   │   │   ├── seed1442gr.ini
    │   │   │   └── seed1442newton.ini
    │   │   ├── seed1443
    │   │   │   ├── seed1443gr.ini
    │   │   │   └── seed1443newton.ini
    │   │   ├── seed1444
    │   │   │   ├── seed1444gr.ini
    │   │   │   └── seed1444newton.ini
    │   │   ├── seed1445
    │   │   │   ├── seed1445gr.ini
    │   │   │   └── seed1445newton.ini
    │   │   ├── seed1446
    │   │   │   ├── seed1446gr.ini
    │   │   │   └── seed1446newton.ini
    │   │   ├── seed1447
    │   │   │   ├── seed1447gr.ini
    │   │   │   └── seed1447newton.ini
    │   │   ├── seed1448
    │   │   │   ├── seed1448gr.ini
    │   │   │   └── seed1448newton.ini
    │   │   ├── seed1449
    │   │   │   ├── seed1449gr.ini
    │   │   │   └── seed1449newton.ini
    │   │   ├── seed1450
    │   │   │   ├── seed1450gr.ini
    │   │   │   └── seed1450newton.ini
    │   │   ├── seed1451
    │   │   │   ├── seed1451gr.ini
    │   │   │   └── seed1451newton.ini
    │   │   ├── seed1452
    │   │   │   ├── seed1452gr.ini
    │   │   │   └── seed1452newton.ini
    │   │   ├── seed1453
    │   │   │   ├── seed1453gr.ini
    │   │   │   └── seed1453newton.ini
    │   │   ├── seed1454
    │   │   │   ├── seed1454gr.ini
    │   │   │   └── seed1454newton.ini
    │   │   ├── seed1455
    │   │   │   ├── seed1455gr.ini
    │   │   │   └── seed1455newton.ini
    │   │   ├── seed1456
    │   │   │   ├── seed1456gr.ini
    │   │   │   └── seed1456newton.ini
    │   │   ├── seed1457
    │   │   │   ├── seed1457gr.ini
    │   │   │   └── seed1457newton.ini
    │   │   ├── seed1458
    │   │   │   ├── seed1458gr.ini
    │   │   │   └── seed1458newton.ini
    │   │   ├── seed1459
    │   │   │   ├── seed1459gr.ini
    │   │   │   └── seed1459newton.ini
    │   │   ├── seed1460
    │   │   │   ├── seed1460gr.ini
    │   │   │   └── seed1460newton.ini
    │   │   ├── seed1461
    │   │   │   ├── seed1461gr.ini
    │   │   │   └── seed1461newton.ini
    │   │   ├── seed1462
    │   │   │   ├── seed1462gr.ini
    │   │   │   └── seed1462newton.ini
    │   │   ├── seed1463
    │   │   │   ├── seed1463gr.ini
    │   │   │   └── seed1463newton.ini
    │   │   ├── seed1464
    │   │   │   ├── seed1464gr.ini
    │   │   │   └── seed1464newton.ini
    │   │   ├── seed1465
    │   │   │   ├── seed1465gr.ini
    │   │   │   └── seed1465newton.ini
    │   │   ├── seed1466
    │   │   │   ├── seed1466gr.ini
    │   │   │   └── seed1466newton.ini
    │   │   ├── seed1467
    │   │   │   ├── seed1467gr.ini
    │   │   │   └── seed1467newton.ini
    │   │   ├── seed1468
    │   │   │   ├── seed1468gr.ini
    │   │   │   └── seed1468newton.ini
    │   │   ├── seed1469
    │   │   │   ├── seed1469gr.ini
    │   │   │   └── seed1469newton.ini
    │   │   ├── seed1470
    │   │   │   ├── seed1470gr.ini
    │   │   │   └── seed1470newton.ini
    │   │   ├── seed1471
    │   │   │   ├── seed1471gr.ini
    │   │   │   └── seed1471newton.ini
    │   │   ├── seed1472
    │   │   │   ├── seed1472gr.ini
    │   │   │   └── seed1472newton.ini
    │   │   ├── seed1473
    │   │   │   ├── seed1473gr.ini
    │   │   │   └── seed1473newton.ini
    │   │   ├── seed1474
    │   │   │   ├── seed1474gr.ini
    │   │   │   └── seed1474newton.ini
    │   │   ├── seed1475
    │   │   │   ├── seed1475gr.ini
    │   │   │   └── seed1475newton.ini
    │   │   ├── seed1476
    │   │   │   ├── seed1476gr.ini
    │   │   │   └── seed1476newton.ini
    │   │   ├── seed1477
    │   │   │   ├── seed1477gr.ini
    │   │   │   └── seed1477newton.ini
    │   │   ├── seed1478
    │   │   │   ├── seed1478gr.ini
    │   │   │   └── seed1478newton.ini
    │   │   ├── seed1479
    │   │   │   ├── seed1479gr.ini
    │   │   │   └── seed1479newton.ini
    │   │   ├── seed1480
    │   │   │   ├── seed1480gr.ini
    │   │   │   └── seed1480newton.ini
    │   │   ├── seed1481
    │   │   │   ├── seed1481gr.ini
    │   │   │   └── seed1481newton.ini
    │   │   ├── seed1482
    │   │   │   ├── seed1482gr.ini
    │   │   │   └── seed1482newton.ini
    │   │   ├── seed1483
    │   │   │   ├── seed1483gr.ini
    │   │   │   └── seed1483newton.ini
    │   │   ├── seed1484
    │   │   │   ├── seed1484gr.ini
    │   │   │   └── seed1484newton.ini
    │   │   ├── seed1485
    │   │   │   ├── seed1485gr.ini
    │   │   │   └── seed1485newton.ini
    │   │   ├── seed1486
    │   │   │   ├── seed1486gr.ini
    │   │   │   └── seed1486newton.ini
    │   │   ├── seed1487
    │   │   │   ├── seed1487gr.ini
    │   │   │   └── seed1487newton.ini
    │   │   ├── seed1488
    │   │   │   ├── seed1488gr.ini
    │   │   │   └── seed1488newton.ini
    │   │   ├── seed1489
    │   │   │   ├── seed1489gr.ini
    │   │   │   └── seed1489newton.ini
    │   │   ├── seed1490
    │   │   │   ├── seed1490gr.ini
    │   │   │   └── seed1490newton.ini
    │   │   ├── seed1491
    │   │   │   ├── seed1491gr.ini
    │   │   │   └── seed1491newton.ini
    │   │   ├── seed1492
    │   │   │   ├── seed1492gr.ini
    │   │   │   └── seed1492newton.ini
    │   │   ├── seed1493
    │   │   │   ├── seed1493gr.ini
    │   │   │   └── seed1493newton.ini
    │   │   ├── seed1494
    │   │   │   ├── seed1494gr.ini
    │   │   │   └── seed1494newton.ini
    │   │   ├── seed1495
    │   │   │   ├── seed1495gr.ini
    │   │   │   └── seed1495newton.ini
    │   │   ├── seed1496
    │   │   │   ├── seed1496gr.ini
    │   │   │   └── seed1496newton.ini
    │   │   ├── seed1497
    │   │   │   ├── seed1497gr.ini
    │   │   │   └── seed1497newton.ini
    │   │   ├── seed1498
    │   │   │   ├── seed1498gr.ini
    │   │   │   └── seed1498newton.ini
    │   │   ├── seed1499
    │   │   │   ├── seed1499gr.ini
    │   │   │   └── seed1499newton.ini
    │   │   ├── seed1500
    │   │   │   ├── seed1500gr.ini
    │   │   │   └── seed1500newton.ini
    │   │   ├── seed1501
    │   │   │   ├── seed1501gr.ini
    │   │   │   └── seed1501newton.ini
    │   │   ├── seed1502
    │   │   │   ├── seed1502gr.ini
    │   │   │   └── seed1502newton.ini
    │   │   ├── seed1503
    │   │   │   ├── seed1503gr.ini
    │   │   │   └── seed1503newton.ini
    │   │   ├── seed1504
    │   │   │   ├── seed1504gr.ini
    │   │   │   └── seed1504newton.ini
    │   │   ├── seed1505
    │   │   │   ├── seed1505gr.ini
    │   │   │   └── seed1505newton.ini
    │   │   ├── seed1506
    │   │   │   ├── seed1506gr.ini
    │   │   │   └── seed1506newton.ini
    │   │   ├── seed1507
    │   │   │   ├── seed1507gr.ini
    │   │   │   └── seed1507newton.ini
    │   │   ├── seed1508
    │   │   │   ├── seed1508gr.ini
    │   │   │   └── seed1508newton.ini
    │   │   ├── seed1509
    │   │   │   ├── seed1509gr.ini
    │   │   │   └── seed1509newton.ini
    │   │   ├── seed1510
    │   │   │   ├── seed1510gr.ini
    │   │   │   └── seed1510newton.ini
    │   │   ├── seed1511
    │   │   │   ├── seed1511gr.ini
    │   │   │   └── seed1511newton.ini
    │   │   ├── seed1512
    │   │   │   ├── seed1512gr.ini
    │   │   │   └── seed1512newton.ini
    │   │   ├── seed1513
    │   │   │   ├── seed1513gr.ini
    │   │   │   └── seed1513newton.ini
    │   │   ├── seed1514
    │   │   │   ├── seed1514gr.ini
    │   │   │   └── seed1514newton.ini
    │   │   ├── seed1515
    │   │   │   ├── seed1515gr.ini
    │   │   │   └── seed1515newton.ini
    │   │   ├── seed1516
    │   │   │   ├── seed1516gr.ini
    │   │   │   └── seed1516newton.ini
    │   │   ├── seed1517
    │   │   │   ├── seed1517gr.ini
    │   │   │   └── seed1517newton.ini
    │   │   ├── seed1518
    │   │   │   ├── seed1518gr.ini
    │   │   │   └── seed1518newton.ini
    │   │   ├── seed1519
    │   │   │   ├── seed1519gr.ini
    │   │   │   └── seed1519newton.ini
    │   │   ├── seed1520
    │   │   │   ├── seed1520gr.ini
    │   │   │   └── seed1520newton.ini
    │   │   ├── seed1521
    │   │   │   ├── seed1521gr.ini
    │   │   │   └── seed1521newton.ini
    │   │   ├── seed1522
    │   │   │   ├── seed1522gr.ini
    │   │   │   └── seed1522newton.ini
    │   │   ├── seed1523
    │   │   │   ├── seed1523gr.ini
    │   │   │   └── seed1523newton.ini
    │   │   ├── seed1524
    │   │   │   ├── seed1524gr.ini
    │   │   │   └── seed1524newton.ini
    │   │   ├── seed1525
    │   │   │   ├── seed1525gr.ini
    │   │   │   └── seed1525newton.ini
    │   │   ├── seed1526
    │   │   │   ├── seed1526gr.ini
    │   │   │   └── seed1526newton.ini
    │   │   ├── seed1527
    │   │   │   ├── seed1527gr.ini
    │   │   │   └── seed1527newton.ini
    │   │   ├── seed1528
    │   │   │   ├── seed1528gr.ini
    │   │   │   └── seed1528newton.ini
    │   │   ├── seed1529
    │   │   │   ├── seed1529gr.ini
    │   │   │   └── seed1529newton.ini
    │   │   ├── seed1530
    │   │   │   ├── seed1530gr.ini
    │   │   │   └── seed1530newton.ini
    │   │   ├── seed1531
    │   │   │   ├── seed1531gr.ini
    │   │   │   └── seed1531newton.ini
    │   │   ├── seed1532
    │   │   │   ├── seed1532gr.ini
    │   │   │   └── seed1532newton.ini
    │   │   ├── seed1533
    │   │   │   ├── seed1533gr.ini
    │   │   │   └── seed1533newton.ini
    │   │   ├── seed1534
    │   │   │   ├── seed1534gr.ini
    │   │   │   └── seed1534newton.ini
    │   │   ├── seed1535
    │   │   │   ├── seed1535gr.ini
    │   │   │   └── seed1535newton.ini
    │   │   ├── seed1536
    │   │   │   ├── seed1536gr.ini
    │   │   │   └── seed1536newton.ini
    │   │   ├── seed1537
    │   │   │   ├── seed1537gr.ini
    │   │   │   └── seed1537newton.ini
    │   │   ├── seed1538
    │   │   │   ├── seed1538gr.ini
    │   │   │   └── seed1538newton.ini
    │   │   ├── seed1539
    │   │   │   ├── seed1539gr.ini
    │   │   │   └── seed1539newton.ini
    │   │   ├── seed1540
    │   │   │   ├── seed1540gr.ini
    │   │   │   └── seed1540newton.ini
    │   │   ├── seed1541
    │   │   │   ├── seed1541gr.ini
    │   │   │   └── seed1541newton.ini
    │   │   ├── seed1542
    │   │   │   ├── seed1542gr.ini
    │   │   │   └── seed1542newton.ini
    │   │   ├── seed1543
    │   │   │   ├── seed1543gr.ini
    │   │   │   └── seed1543newton.ini
    │   │   ├── seed1544
    │   │   │   ├── seed1544gr.ini
    │   │   │   └── seed1544newton.ini
    │   │   ├── seed1545
    │   │   │   ├── seed1545gr.ini
    │   │   │   └── seed1545newton.ini
    │   │   ├── seed1546
    │   │   │   ├── seed1546gr.ini
    │   │   │   └── seed1546newton.ini
    │   │   ├── seed1547
    │   │   │   ├── seed1547gr.ini
    │   │   │   └── seed1547newton.ini
    │   │   ├── seed1548
    │   │   │   ├── seed1548gr.ini
    │   │   │   └── seed1548newton.ini
    │   │   ├── seed1549
    │   │   │   ├── seed1549gr.ini
    │   │   │   └── seed1549newton.ini
    │   │   ├── seed1550
    │   │   │   ├── seed1550gr.ini
    │   │   │   └── seed1550newton.ini
    │   │   ├── seed1551
    │   │   │   ├── seed1551gr.ini
    │   │   │   └── seed1551newton.ini
    │   │   ├── seed1552
    │   │   │   ├── seed1552gr.ini
    │   │   │   └── seed1552newton.ini
    │   │   ├── seed1553
    │   │   │   ├── seed1553gr.ini
    │   │   │   └── seed1553newton.ini
    │   │   ├── seed1554
    │   │   │   ├── seed1554gr.ini
    │   │   │   └── seed1554newton.ini
    │   │   ├── seed1555
    │   │   │   ├── seed1555gr.ini
    │   │   │   └── seed1555newton.ini
    │   │   ├── seed1556
    │   │   │   ├── seed1556gr.ini
    │   │   │   └── seed1556newton.ini
    │   │   ├── seed1557
    │   │   │   ├── seed1557gr.ini
    │   │   │   └── seed1557newton.ini
    │   │   ├── seed1558
    │   │   │   ├── seed1558gr.ini
    │   │   │   └── seed1558newton.ini
    │   │   ├── seed1559
    │   │   │   ├── seed1559gr.ini
    │   │   │   └── seed1559newton.ini
    │   │   ├── seed1560
    │   │   │   ├── seed1560gr.ini
    │   │   │   └── seed1560newton.ini
    │   │   ├── seed1561
    │   │   │   ├── seed1561gr.ini
    │   │   │   └── seed1561newton.ini
    │   │   ├── seed1562
    │   │   │   ├── seed1562gr.ini
    │   │   │   └── seed1562newton.ini
    │   │   ├── seed1563
    │   │   │   ├── seed1563gr.ini
    │   │   │   └── seed1563newton.ini
    │   │   ├── seed1564
    │   │   │   ├── seed1564gr.ini
    │   │   │   └── seed1564newton.ini
    │   │   ├── seed1565
    │   │   │   ├── seed1565gr.ini
    │   │   │   └── seed1565newton.ini
    │   │   ├── seed1566
    │   │   │   ├── seed1566gr.ini
    │   │   │   └── seed1566newton.ini
    │   │   ├── seed1567
    │   │   │   ├── seed1567gr.ini
    │   │   │   └── seed1567newton.ini
    │   │   ├── seed1568
    │   │   │   ├── seed1568gr.ini
    │   │   │   └── seed1568newton.ini
    │   │   ├── seed1569
    │   │   │   ├── seed1569gr.ini
    │   │   │   └── seed1569newton.ini
    │   │   ├── seed1570
    │   │   │   ├── seed1570gr.ini
    │   │   │   └── seed1570newton.ini
    │   │   ├── seed1571
    │   │   │   ├── seed1571gr.ini
    │   │   │   └── seed1571newton.ini
    │   │   ├── seed1572
    │   │   │   ├── seed1572gr.ini
    │   │   │   └── seed1572newton.ini
    │   │   ├── seed1573
    │   │   │   ├── seed1573gr.ini
    │   │   │   └── seed1573newton.ini
    │   │   ├── seed1574
    │   │   │   ├── seed1574gr.ini
    │   │   │   └── seed1574newton.ini
    │   │   ├── seed1575
    │   │   │   ├── seed1575gr.ini
    │   │   │   └── seed1575newton.ini
    │   │   ├── seed1576
    │   │   │   ├── seed1576gr.ini
    │   │   │   └── seed1576newton.ini
    │   │   ├── seed1577
    │   │   │   ├── seed1577gr.ini
    │   │   │   └── seed1577newton.ini
    │   │   ├── seed1578
    │   │   │   ├── seed1578gr.ini
    │   │   │   └── seed1578newton.ini
    │   │   ├── seed1579
    │   │   │   ├── seed1579gr.ini
    │   │   │   └── seed1579newton.ini
    │   │   ├── seed1580
    │   │   │   ├── seed1580gr.ini
    │   │   │   └── seed1580newton.ini
    │   │   ├── seed1581
    │   │   │   ├── seed1581gr.ini
    │   │   │   └── seed1581newton.ini
    │   │   ├── seed1582
    │   │   │   ├── seed1582gr.ini
    │   │   │   └── seed1582newton.ini
    │   │   ├── seed1583
    │   │   │   ├── seed1583gr.ini
    │   │   │   └── seed1583newton.ini
    │   │   ├── seed1584
    │   │   │   ├── seed1584gr.ini
    │   │   │   └── seed1584newton.ini
    │   │   ├── seed1585
    │   │   │   ├── seed1585gr.ini
    │   │   │   └── seed1585newton.ini
    │   │   ├── seed1586
    │   │   │   ├── seed1586gr.ini
    │   │   │   └── seed1586newton.ini
    │   │   ├── seed1587
    │   │   │   ├── seed1587gr.ini
    │   │   │   └── seed1587newton.ini
    │   │   ├── seed1588
    │   │   │   ├── seed1588gr.ini
    │   │   │   └── seed1588newton.ini
    │   │   ├── seed1589
    │   │   │   ├── seed1589gr.ini
    │   │   │   └── seed1589newton.ini
    │   │   ├── seed1590
    │   │   │   ├── seed1590gr.ini
    │   │   │   └── seed1590newton.ini
    │   │   ├── seed1591
    │   │   │   ├── seed1591gr.ini
    │   │   │   └── seed1591newton.ini
    │   │   ├── seed1592
    │   │   │   ├── seed1592gr.ini
    │   │   │   └── seed1592newton.ini
    │   │   ├── seed1593
    │   │   │   ├── seed1593gr.ini
    │   │   │   └── seed1593newton.ini
    │   │   ├── seed1594
    │   │   │   ├── seed1594gr.ini
    │   │   │   └── seed1594newton.ini
    │   │   ├── seed1595
    │   │   │   ├── seed1595gr.ini
    │   │   │   └── seed1595newton.ini
    │   │   ├── seed1596
    │   │   │   ├── seed1596gr.ini
    │   │   │   └── seed1596newton.ini
    │   │   ├── seed1597
    │   │   │   ├── seed1597gr.ini
    │   │   │   └── seed1597newton.ini
    │   │   ├── seed1598
    │   │   │   ├── seed1598gr.ini
    │   │   │   └── seed1598newton.ini
    │   │   ├── seed1599
    │   │   │   ├── seed1599gr.ini
    │   │   │   └── seed1599newton.ini
    │   │   ├── seed1600
    │   │   │   ├── seed1600gr.ini
    │   │   │   └── seed1600newton.ini
    │   │   ├── seed1601
    │   │   │   ├── seed1601gr.ini
    │   │   │   └── seed1601newton.ini
    │   │   ├── seed1602
    │   │   │   ├── seed1602gr.ini
    │   │   │   └── seed1602newton.ini
    │   │   ├── seed1603
    │   │   │   ├── seed1603gr.ini
    │   │   │   └── seed1603newton.ini
    │   │   ├── seed1604
    │   │   │   ├── seed1604gr.ini
    │   │   │   └── seed1604newton.ini
    │   │   ├── seed1605
    │   │   │   ├── seed1605gr.ini
    │   │   │   └── seed1605newton.ini
    │   │   ├── seed1606
    │   │   │   ├── seed1606gr.ini
    │   │   │   └── seed1606newton.ini
    │   │   ├── seed1607
    │   │   │   ├── seed1607gr.ini
    │   │   │   └── seed1607newton.ini
    │   │   ├── seed1608
    │   │   │   ├── seed1608gr.ini
    │   │   │   └── seed1608newton.ini
    │   │   ├── seed1609
    │   │   │   ├── seed1609gr.ini
    │   │   │   └── seed1609newton.ini
    │   │   ├── seed1610
    │   │   │   ├── seed1610gr.ini
    │   │   │   └── seed1610newton.ini
    │   │   ├── seed1611
    │   │   │   ├── seed1611gr.ini
    │   │   │   └── seed1611newton.ini
    │   │   ├── seed1612
    │   │   │   ├── seed1612gr.ini
    │   │   │   └── seed1612newton.ini
    │   │   ├── seed1613
    │   │   │   ├── seed1613gr.ini
    │   │   │   └── seed1613newton.ini
    │   │   ├── seed1614
    │   │   │   ├── seed1614gr.ini
    │   │   │   └── seed1614newton.ini
    │   │   ├── seed1615
    │   │   │   ├── seed1615gr.ini
    │   │   │   └── seed1615newton.ini
    │   │   ├── seed1616
    │   │   │   ├── seed1616gr.ini
    │   │   │   └── seed1616newton.ini
    │   │   ├── seed1617
    │   │   │   ├── seed1617gr.ini
    │   │   │   └── seed1617newton.ini
    │   │   ├── seed1618
    │   │   │   ├── seed1618gr.ini
    │   │   │   └── seed1618newton.ini
    │   │   ├── seed1619
    │   │   │   ├── seed1619gr.ini
    │   │   │   └── seed1619newton.ini
    │   │   ├── seed1620
    │   │   │   ├── seed1620gr.ini
    │   │   │   └── seed1620newton.ini
    │   │   ├── seed1621
    │   │   │   ├── seed1621gr.ini
    │   │   │   └── seed1621newton.ini
    │   │   ├── seed1622
    │   │   │   ├── seed1622gr.ini
    │   │   │   └── seed1622newton.ini
    │   │   ├── seed1623
    │   │   │   ├── seed1623gr.ini
    │   │   │   └── seed1623newton.ini
    │   │   ├── seed1624
    │   │   │   ├── seed1624gr.ini
    │   │   │   └── seed1624newton.ini
    │   │   ├── seed1625
    │   │   │   ├── seed1625gr.ini
    │   │   │   └── seed1625newton.ini
    │   │   ├── seed1626
    │   │   │   ├── seed1626gr.ini
    │   │   │   └── seed1626newton.ini
    │   │   ├── seed1627
    │   │   │   ├── seed1627gr.ini
    │   │   │   └── seed1627newton.ini
    │   │   ├── seed1628
    │   │   │   ├── seed1628gr.ini
    │   │   │   └── seed1628newton.ini
    │   │   ├── seed1629
    │   │   │   ├── seed1629gr.ini
    │   │   │   └── seed1629newton.ini
    │   │   ├── seed1630
    │   │   │   ├── seed1630gr.ini
    │   │   │   └── seed1630newton.ini
    │   │   ├── seed1631
    │   │   │   ├── seed1631gr.ini
    │   │   │   └── seed1631newton.ini
    │   │   ├── seed1632
    │   │   │   ├── seed1632gr.ini
    │   │   │   └── seed1632newton.ini
    │   │   ├── seed1633
    │   │   │   ├── seed1633gr.ini
    │   │   │   └── seed1633newton.ini
    │   │   ├── seed1634
    │   │   │   ├── seed1634gr.ini
    │   │   │   └── seed1634newton.ini
    │   │   ├── seed1635
    │   │   │   ├── seed1635gr.ini
    │   │   │   └── seed1635newton.ini
    │   │   ├── seed1636
    │   │   │   ├── seed1636gr.ini
    │   │   │   └── seed1636newton.ini
    │   │   ├── seed1637
    │   │   │   ├── seed1637gr.ini
    │   │   │   └── seed1637newton.ini
    │   │   ├── seed1638
    │   │   │   ├── seed1638gr.ini
    │   │   │   └── seed1638newton.ini
    │   │   ├── seed1639
    │   │   │   ├── seed1639gr.ini
    │   │   │   └── seed1639newton.ini
    │   │   ├── seed1640
    │   │   │   ├── seed1640gr.ini
    │   │   │   └── seed1640newton.ini
    │   │   ├── seed1641
    │   │   │   ├── seed1641gr.ini
    │   │   │   └── seed1641newton.ini
    │   │   ├── seed1642
    │   │   │   ├── seed1642gr.ini
    │   │   │   └── seed1642newton.ini
    │   │   ├── seed1643
    │   │   │   ├── seed1643gr.ini
    │   │   │   └── seed1643newton.ini
    │   │   ├── seed1644
    │   │   │   ├── seed1644gr.ini
    │   │   │   └── seed1644newton.ini
    │   │   ├── seed1645
    │   │   │   ├── seed1645gr.ini
    │   │   │   └── seed1645newton.ini
    │   │   ├── seed1646
    │   │   │   ├── seed1646gr.ini
    │   │   │   └── seed1646newton.ini
    │   │   ├── seed1647
    │   │   │   ├── seed1647gr.ini
    │   │   │   └── seed1647newton.ini
    │   │   ├── seed1648
    │   │   │   ├── seed1648gr.ini
    │   │   │   └── seed1648newton.ini
    │   │   ├── seed1649
    │   │   │   ├── seed1649gr.ini
    │   │   │   └── seed1649newton.ini
    │   │   ├── seed1650
    │   │   │   ├── seed1650gr.ini
    │   │   │   └── seed1650newton.ini
    │   │   ├── seed1651
    │   │   │   ├── seed1651gr.ini
    │   │   │   └── seed1651newton.ini
    │   │   ├── seed1652
    │   │   │   ├── seed1652gr.ini
    │   │   │   └── seed1652newton.ini
    │   │   ├── seed1653
    │   │   │   ├── seed1653gr.ini
    │   │   │   └── seed1653newton.ini
    │   │   ├── seed1654
    │   │   │   ├── seed1654gr.ini
    │   │   │   └── seed1654newton.ini
    │   │   ├── seed1655
    │   │   │   ├── seed1655gr.ini
    │   │   │   └── seed1655newton.ini
    │   │   ├── seed1656
    │   │   │   ├── seed1656gr.ini
    │   │   │   └── seed1656newton.ini
    │   │   ├── seed1657
    │   │   │   ├── seed1657gr.ini
    │   │   │   └── seed1657newton.ini
    │   │   ├── seed1658
    │   │   │   ├── seed1658gr.ini
    │   │   │   └── seed1658newton.ini
    │   │   ├── seed1659
    │   │   │   ├── seed1659gr.ini
    │   │   │   └── seed1659newton.ini
    │   │   ├── seed1660
    │   │   │   ├── seed1660gr.ini
    │   │   │   └── seed1660newton.ini
    │   │   ├── seed1661
    │   │   │   ├── seed1661gr.ini
    │   │   │   └── seed1661newton.ini
    │   │   ├── seed1662
    │   │   │   ├── seed1662gr.ini
    │   │   │   └── seed1662newton.ini
    │   │   ├── seed1663
    │   │   │   ├── seed1663gr.ini
    │   │   │   └── seed1663newton.ini
    │   │   ├── seed1664
    │   │   │   ├── seed1664gr.ini
    │   │   │   └── seed1664newton.ini
    │   │   ├── seed1665
    │   │   │   ├── seed1665gr.ini
    │   │   │   └── seed1665newton.ini
    │   │   ├── seed1666
    │   │   │   ├── seed1666gr.ini
    │   │   │   └── seed1666newton.ini
    │   │   ├── seed1667
    │   │   │   ├── seed1667gr.ini
    │   │   │   └── seed1667newton.ini
    │   │   ├── seed1668
    │   │   │   ├── seed1668gr.ini
    │   │   │   └── seed1668newton.ini
    │   │   ├── seed1669
    │   │   │   ├── seed1669gr.ini
    │   │   │   └── seed1669newton.ini
    │   │   ├── seed1670
    │   │   │   ├── seed1670gr.ini
    │   │   │   └── seed1670newton.ini
    │   │   ├── seed1671
    │   │   │   ├── seed1671gr.ini
    │   │   │   └── seed1671newton.ini
    │   │   ├── seed1672
    │   │   │   ├── seed1672gr.ini
    │   │   │   └── seed1672newton.ini
    │   │   ├── seed1673
    │   │   │   ├── seed1673gr.ini
    │   │   │   └── seed1673newton.ini
    │   │   ├── seed1674
    │   │   │   ├── seed1674gr.ini
    │   │   │   └── seed1674newton.ini
    │   │   ├── seed1675
    │   │   │   ├── seed1675gr.ini
    │   │   │   └── seed1675newton.ini
    │   │   ├── seed1676
    │   │   │   ├── seed1676gr.ini
    │   │   │   └── seed1676newton.ini
    │   │   ├── seed1677
    │   │   │   ├── seed1677gr.ini
    │   │   │   └── seed1677newton.ini
    │   │   ├── seed1678
    │   │   │   ├── seed1678gr.ini
    │   │   │   └── seed1678newton.ini
    │   │   ├── seed1679
    │   │   │   ├── seed1679gr.ini
    │   │   │   └── seed1679newton.ini
    │   │   ├── seed1680
    │   │   │   ├── seed1680gr.ini
    │   │   │   └── seed1680newton.ini
    │   │   ├── seed1681
    │   │   │   ├── seed1681gr.ini
    │   │   │   └── seed1681newton.ini
    │   │   ├── seed1682
    │   │   │   ├── seed1682gr.ini
    │   │   │   └── seed1682newton.ini
    │   │   ├── seed1683
    │   │   │   ├── seed1683gr.ini
    │   │   │   └── seed1683newton.ini
    │   │   ├── seed1684
    │   │   │   ├── seed1684gr.ini
    │   │   │   └── seed1684newton.ini
    │   │   ├── seed1685
    │   │   │   ├── seed1685gr.ini
    │   │   │   └── seed1685newton.ini
    │   │   ├── seed1686
    │   │   │   ├── seed1686gr.ini
    │   │   │   └── seed1686newton.ini
    │   │   ├── seed1687
    │   │   │   ├── seed1687gr.ini
    │   │   │   └── seed1687newton.ini
    │   │   ├── seed1688
    │   │   │   ├── seed1688gr.ini
    │   │   │   └── seed1688newton.ini
    │   │   ├── seed1689
    │   │   │   ├── seed1689gr.ini
    │   │   │   └── seed1689newton.ini
    │   │   ├── seed1690
    │   │   │   ├── seed1690gr.ini
    │   │   │   └── seed1690newton.ini
    │   │   ├── seed1691
    │   │   │   ├── seed1691gr.ini
    │   │   │   └── seed1691newton.ini
    │   │   ├── seed1692
    │   │   │   ├── seed1692gr.ini
    │   │   │   └── seed1692newton.ini
    │   │   ├── seed1693
    │   │   │   ├── seed1693gr.ini
    │   │   │   └── seed1693newton.ini
    │   │   ├── seed1694
    │   │   │   ├── seed1694gr.ini
    │   │   │   └── seed1694newton.ini
    │   │   ├── seed1695
    │   │   │   ├── seed1695gr.ini
    │   │   │   └── seed1695newton.ini
    │   │   ├── seed1696
    │   │   │   ├── seed1696gr.ini
    │   │   │   └── seed1696newton.ini
    │   │   ├── seed1697
    │   │   │   ├── seed1697gr.ini
    │   │   │   └── seed1697newton.ini
    │   │   ├── seed1698
    │   │   │   ├── seed1698gr.ini
    │   │   │   └── seed1698newton.ini
    │   │   ├── seed1699
    │   │   │   ├── seed1699gr.ini
    │   │   │   └── seed1699newton.ini
    │   │   ├── seed1700
    │   │   │   ├── seed1700gr.ini
    │   │   │   └── seed1700newton.ini
    │   │   ├── seed1701
    │   │   │   ├── seed1701gr.ini
    │   │   │   └── seed1701newton.ini
    │   │   ├── seed1702
    │   │   │   ├── seed1702gr.ini
    │   │   │   └── seed1702newton.ini
    │   │   ├── seed1703
    │   │   │   ├── seed1703gr.ini
    │   │   │   └── seed1703newton.ini
    │   │   ├── seed1704
    │   │   │   ├── seed1704gr.ini
    │   │   │   └── seed1704newton.ini
    │   │   ├── seed1705
    │   │   │   ├── seed1705gr.ini
    │   │   │   └── seed1705newton.ini
    │   │   ├── seed1706
    │   │   │   ├── seed1706gr.ini
    │   │   │   └── seed1706newton.ini
    │   │   ├── seed1707
    │   │   │   ├── seed1707gr.ini
    │   │   │   └── seed1707newton.ini
    │   │   ├── seed1708
    │   │   │   ├── seed1708gr.ini
    │   │   │   └── seed1708newton.ini
    │   │   ├── seed1709
    │   │   │   ├── seed1709gr.ini
    │   │   │   └── seed1709newton.ini
    │   │   ├── seed1710
    │   │   │   ├── seed1710gr.ini
    │   │   │   └── seed1710newton.ini
    │   │   ├── seed1711
    │   │   │   ├── seed1711gr.ini
    │   │   │   └── seed1711newton.ini
    │   │   ├── seed1712
    │   │   │   ├── seed1712gr.ini
    │   │   │   └── seed1712newton.ini
    │   │   ├── seed1713
    │   │   │   ├── seed1713gr.ini
    │   │   │   └── seed1713newton.ini
    │   │   ├── seed1714
    │   │   │   ├── seed1714gr.ini
    │   │   │   └── seed1714newton.ini
    │   │   ├── seed1715
    │   │   │   ├── seed1715gr.ini
    │   │   │   └── seed1715newton.ini
    │   │   ├── seed1716
    │   │   │   ├── seed1716gr.ini
    │   │   │   └── seed1716newton.ini
    │   │   ├── seed1717
    │   │   │   ├── seed1717gr.ini
    │   │   │   └── seed1717newton.ini
    │   │   ├── seed1718
    │   │   │   ├── seed1718gr.ini
    │   │   │   └── seed1718newton.ini
    │   │   ├── seed1719
    │   │   │   ├── seed1719gr.ini
    │   │   │   └── seed1719newton.ini
    │   │   ├── seed1720
    │   │   │   ├── seed1720gr.ini
    │   │   │   └── seed1720newton.ini
    │   │   ├── seed1721
    │   │   │   ├── seed1721gr.ini
    │   │   │   └── seed1721newton.ini
    │   │   ├── seed1722
    │   │   │   ├── seed1722gr.ini
    │   │   │   └── seed1722newton.ini
    │   │   ├── seed1723
    │   │   │   ├── seed1723gr.ini
    │   │   │   └── seed1723newton.ini
    │   │   ├── seed1724
    │   │   │   ├── seed1724gr.ini
    │   │   │   └── seed1724newton.ini
    │   │   ├── seed1725
    │   │   │   ├── seed1725gr.ini
    │   │   │   └── seed1725newton.ini
    │   │   ├── seed1726
    │   │   │   ├── seed1726gr.ini
    │   │   │   └── seed1726newton.ini
    │   │   ├── seed1727
    │   │   │   ├── seed1727gr.ini
    │   │   │   └── seed1727newton.ini
    │   │   ├── seed1728
    │   │   │   ├── seed1728gr.ini
    │   │   │   └── seed1728newton.ini
    │   │   ├── seed1729
    │   │   │   ├── seed1729gr.ini
    │   │   │   └── seed1729newton.ini
    │   │   ├── seed1730
    │   │   │   ├── seed1730gr.ini
    │   │   │   └── seed1730newton.ini
    │   │   ├── seed1731
    │   │   │   ├── seed1731gr.ini
    │   │   │   └── seed1731newton.ini
    │   │   ├── seed1732
    │   │   │   ├── seed1732gr.ini
    │   │   │   └── seed1732newton.ini
    │   │   ├── seed1733
    │   │   │   ├── seed1733gr.ini
    │   │   │   └── seed1733newton.ini
    │   │   ├── seed1734
    │   │   │   ├── seed1734gr.ini
    │   │   │   └── seed1734newton.ini
    │   │   ├── seed1735
    │   │   │   ├── seed1735gr.ini
    │   │   │   └── seed1735newton.ini
    │   │   ├── seed1736
    │   │   │   ├── seed1736gr.ini
    │   │   │   └── seed1736newton.ini
    │   │   ├── seed1737
    │   │   │   ├── seed1737gr.ini
    │   │   │   └── seed1737newton.ini
    │   │   ├── seed1738
    │   │   │   ├── seed1738gr.ini
    │   │   │   └── seed1738newton.ini
    │   │   ├── seed1739
    │   │   │   ├── seed1739gr.ini
    │   │   │   └── seed1739newton.ini
    │   │   ├── seed1740
    │   │   │   ├── seed1740gr.ini
    │   │   │   └── seed1740newton.ini
    │   │   ├── seed1741
    │   │   │   ├── seed1741gr.ini
    │   │   │   └── seed1741newton.ini
    │   │   ├── seed1742
    │   │   │   ├── seed1742gr.ini
    │   │   │   └── seed1742newton.ini
    │   │   ├── seed1743
    │   │   │   ├── seed1743gr.ini
    │   │   │   └── seed1743newton.ini
    │   │   ├── seed1744
    │   │   │   ├── seed1744gr.ini
    │   │   │   └── seed1744newton.ini
    │   │   ├── seed1745
    │   │   │   ├── seed1745gr.ini
    │   │   │   └── seed1745newton.ini
    │   │   ├── seed1746
    │   │   │   ├── seed1746gr.ini
    │   │   │   └── seed1746newton.ini
    │   │   ├── seed1747
    │   │   │   ├── seed1747gr.ini
    │   │   │   └── seed1747newton.ini
    │   │   ├── seed1748
    │   │   │   ├── seed1748gr.ini
    │   │   │   └── seed1748newton.ini
    │   │   ├── seed1749
    │   │   │   ├── seed1749gr.ini
    │   │   │   └── seed1749newton.ini
    │   │   ├── seed1750
    │   │   │   ├── seed1750gr.ini
    │   │   │   └── seed1750newton.ini
    │   │   ├── seed1751
    │   │   │   ├── seed1751gr.ini
    │   │   │   └── seed1751newton.ini
    │   │   ├── seed1752
    │   │   │   ├── seed1752gr.ini
    │   │   │   └── seed1752newton.ini
    │   │   ├── seed1753
    │   │   │   ├── seed1753gr.ini
    │   │   │   └── seed1753newton.ini
    │   │   ├── seed1754
    │   │   │   ├── seed1754gr.ini
    │   │   │   └── seed1754newton.ini
    │   │   ├── seed1755
    │   │   │   ├── seed1755gr.ini
    │   │   │   └── seed1755newton.ini
    │   │   ├── seed1756
    │   │   │   ├── seed1756gr.ini
    │   │   │   └── seed1756newton.ini
    │   │   ├── seed1757
    │   │   │   ├── seed1757gr.ini
    │   │   │   └── seed1757newton.ini
    │   │   ├── seed1758
    │   │   │   ├── seed1758gr.ini
    │   │   │   └── seed1758newton.ini
    │   │   ├── seed1759
    │   │   │   ├── seed1759gr.ini
    │   │   │   └── seed1759newton.ini
    │   │   ├── seed1760
    │   │   │   ├── seed1760gr.ini
    │   │   │   └── seed1760newton.ini
    │   │   ├── seed1761
    │   │   │   ├── seed1761gr.ini
    │   │   │   └── seed1761newton.ini
    │   │   ├── seed1762
    │   │   │   ├── seed1762gr.ini
    │   │   │   └── seed1762newton.ini
    │   │   ├── seed1763
    │   │   │   ├── seed1763gr.ini
    │   │   │   └── seed1763newton.ini
    │   │   ├── seed1764
    │   │   │   ├── seed1764gr.ini
    │   │   │   └── seed1764newton.ini
    │   │   ├── seed1765
    │   │   │   ├── seed1765gr.ini
    │   │   │   └── seed1765newton.ini
    │   │   ├── seed1766
    │   │   │   ├── seed1766gr.ini
    │   │   │   └── seed1766newton.ini
    │   │   ├── seed1767
    │   │   │   ├── seed1767gr.ini
    │   │   │   └── seed1767newton.ini
    │   │   ├── seed1768
    │   │   │   ├── seed1768gr.ini
    │   │   │   └── seed1768newton.ini
    │   │   ├── seed1769
    │   │   │   ├── seed1769gr.ini
    │   │   │   └── seed1769newton.ini
    │   │   ├── seed1770
    │   │   │   ├── seed1770gr.ini
    │   │   │   └── seed1770newton.ini
    │   │   ├── seed1771
    │   │   │   ├── seed1771gr.ini
    │   │   │   └── seed1771newton.ini
    │   │   ├── seed1772
    │   │   │   ├── seed1772gr.ini
    │   │   │   └── seed1772newton.ini
    │   │   ├── seed1773
    │   │   │   ├── seed1773gr.ini
    │   │   │   └── seed1773newton.ini
    │   │   ├── seed1774
    │   │   │   ├── seed1774gr.ini
    │   │   │   └── seed1774newton.ini
    │   │   ├── seed1775
    │   │   │   ├── seed1775gr.ini
    │   │   │   └── seed1775newton.ini
    │   │   ├── seed1776
    │   │   │   ├── seed1776gr.ini
    │   │   │   └── seed1776newton.ini
    │   │   ├── seed1777
    │   │   │   ├── seed1777gr.ini
    │   │   │   └── seed1777newton.ini
    │   │   ├── seed1778
    │   │   │   ├── seed1778gr.ini
    │   │   │   └── seed1778newton.ini
    │   │   ├── seed1779
    │   │   │   ├── seed1779gr.ini
    │   │   │   └── seed1779newton.ini
    │   │   ├── seed1780
    │   │   │   ├── seed1780gr.ini
    │   │   │   └── seed1780newton.ini
    │   │   ├── seed1781
    │   │   │   ├── seed1781gr.ini
    │   │   │   └── seed1781newton.ini
    │   │   ├── seed1782
    │   │   │   ├── seed1782gr.ini
    │   │   │   └── seed1782newton.ini
    │   │   ├── seed1783
    │   │   │   ├── seed1783gr.ini
    │   │   │   └── seed1783newton.ini
    │   │   ├── seed1784
    │   │   │   ├── seed1784gr.ini
    │   │   │   └── seed1784newton.ini
    │   │   ├── seed1785
    │   │   │   ├── seed1785gr.ini
    │   │   │   └── seed1785newton.ini
    │   │   ├── seed1786
    │   │   │   ├── seed1786gr.ini
    │   │   │   └── seed1786newton.ini
    │   │   ├── seed1787
    │   │   │   ├── seed1787gr.ini
    │   │   │   └── seed1787newton.ini
    │   │   ├── seed1788
    │   │   │   ├── seed1788gr.ini
    │   │   │   └── seed1788newton.ini
    │   │   ├── seed1789
    │   │   │   ├── seed1789gr.ini
    │   │   │   └── seed1789newton.ini
    │   │   ├── seed1790
    │   │   │   ├── seed1790gr.ini
    │   │   │   └── seed1790newton.ini
    │   │   ├── seed1791
    │   │   │   ├── seed1791gr.ini
    │   │   │   └── seed1791newton.ini
    │   │   ├── seed1792
    │   │   │   ├── seed1792gr.ini
    │   │   │   └── seed1792newton.ini
    │   │   ├── seed1793
    │   │   │   ├── seed1793gr.ini
    │   │   │   └── seed1793newton.ini
    │   │   ├── seed1794
    │   │   │   ├── seed1794gr.ini
    │   │   │   └── seed1794newton.ini
    │   │   ├── seed1795
    │   │   │   ├── seed1795gr.ini
    │   │   │   └── seed1795newton.ini
    │   │   ├── seed1796
    │   │   │   ├── seed1796gr.ini
    │   │   │   └── seed1796newton.ini
    │   │   ├── seed1797
    │   │   │   ├── seed1797gr.ini
    │   │   │   └── seed1797newton.ini
    │   │   ├── seed1798
    │   │   │   ├── seed1798gr.ini
    │   │   │   └── seed1798newton.ini
    │   │   ├── seed1799
    │   │   │   ├── seed1799gr.ini
    │   │   │   └── seed1799newton.ini
    │   │   ├── seed1800
    │   │   │   ├── seed1800gr.ini
    │   │   │   └── seed1800newton.ini
    │   │   ├── seed1801
    │   │   │   ├── seed1801gr.ini
    │   │   │   └── seed1801newton.ini
    │   │   ├── seed1802
    │   │   │   ├── seed1802gr.ini
    │   │   │   └── seed1802newton.ini
    │   │   ├── seed1803
    │   │   │   ├── seed1803gr.ini
    │   │   │   └── seed1803newton.ini
    │   │   ├── seed1804
    │   │   │   ├── seed1804gr.ini
    │   │   │   └── seed1804newton.ini
    │   │   ├── seed1805
    │   │   │   ├── seed1805gr.ini
    │   │   │   └── seed1805newton.ini
    │   │   ├── seed1806
    │   │   │   ├── seed1806gr.ini
    │   │   │   └── seed1806newton.ini
    │   │   ├── seed1807
    │   │   │   ├── seed1807gr.ini
    │   │   │   └── seed1807newton.ini
    │   │   ├── seed1808
    │   │   │   ├── seed1808gr.ini
    │   │   │   └── seed1808newton.ini
    │   │   ├── seed1809
    │   │   │   ├── seed1809gr.ini
    │   │   │   └── seed1809newton.ini
    │   │   ├── seed1810
    │   │   │   ├── seed1810gr.ini
    │   │   │   └── seed1810newton.ini
    │   │   ├── seed1811
    │   │   │   ├── seed1811gr.ini
    │   │   │   └── seed1811newton.ini
    │   │   ├── seed1812
    │   │   │   ├── seed1812gr.ini
    │   │   │   └── seed1812newton.ini
    │   │   ├── seed1813
    │   │   │   ├── seed1813gr.ini
    │   │   │   └── seed1813newton.ini
    │   │   ├── seed1814
    │   │   │   ├── seed1814gr.ini
    │   │   │   └── seed1814newton.ini
    │   │   ├── seed1815
    │   │   │   ├── seed1815gr.ini
    │   │   │   └── seed1815newton.ini
    │   │   ├── seed1816
    │   │   │   ├── seed1816gr.ini
    │   │   │   └── seed1816newton.ini
    │   │   ├── seed1817
    │   │   │   ├── seed1817gr.ini
    │   │   │   └── seed1817newton.ini
    │   │   ├── seed1818
    │   │   │   ├── seed1818gr.ini
    │   │   │   └── seed1818newton.ini
    │   │   ├── seed1819
    │   │   │   ├── seed1819gr.ini
    │   │   │   └── seed1819newton.ini
    │   │   ├── seed1820
    │   │   │   ├── seed1820gr.ini
    │   │   │   └── seed1820newton.ini
    │   │   ├── seed1821
    │   │   │   ├── seed1821gr.ini
    │   │   │   └── seed1821newton.ini
    │   │   ├── seed1822
    │   │   │   ├── seed1822gr.ini
    │   │   │   └── seed1822newton.ini
    │   │   ├── seed1823
    │   │   │   ├── seed1823gr.ini
    │   │   │   └── seed1823newton.ini
    │   │   ├── seed1824
    │   │   │   ├── seed1824gr.ini
    │   │   │   └── seed1824newton.ini
    │   │   ├── seed1825
    │   │   │   ├── seed1825gr.ini
    │   │   │   └── seed1825newton.ini
    │   │   ├── seed1826
    │   │   │   ├── seed1826gr.ini
    │   │   │   └── seed1826newton.ini
    │   │   ├── seed1827
    │   │   │   ├── seed1827gr.ini
    │   │   │   └── seed1827newton.ini
    │   │   ├── seed1828
    │   │   │   ├── seed1828gr.ini
    │   │   │   └── seed1828newton.ini
    │   │   ├── seed1829
    │   │   │   ├── seed1829gr.ini
    │   │   │   └── seed1829newton.ini
    │   │   ├── seed1830
    │   │   │   ├── seed1830gr.ini
    │   │   │   └── seed1830newton.ini
    │   │   ├── seed1831
    │   │   │   ├── seed1831gr.ini
    │   │   │   └── seed1831newton.ini
    │   │   ├── seed1832
    │   │   │   ├── seed1832gr.ini
    │   │   │   └── seed1832newton.ini
    │   │   ├── seed1833
    │   │   │   ├── seed1833gr.ini
    │   │   │   └── seed1833newton.ini
    │   │   ├── seed1834
    │   │   │   ├── seed1834gr.ini
    │   │   │   └── seed1834newton.ini
    │   │   ├── seed1835
    │   │   │   ├── seed1835gr.ini
    │   │   │   └── seed1835newton.ini
    │   │   ├── seed1836
    │   │   │   ├── seed1836gr.ini
    │   │   │   └── seed1836newton.ini
    │   │   ├── seed1837
    │   │   │   ├── seed1837gr.ini
    │   │   │   └── seed1837newton.ini
    │   │   ├── seed1838
    │   │   │   ├── seed1838gr.ini
    │   │   │   └── seed1838newton.ini
    │   │   ├── seed1839
    │   │   │   ├── seed1839gr.ini
    │   │   │   └── seed1839newton.ini
    │   │   ├── seed1840
    │   │   │   ├── seed1840gr.ini
    │   │   │   └── seed1840newton.ini
    │   │   ├── seed1841
    │   │   │   ├── seed1841gr.ini
    │   │   │   └── seed1841newton.ini
    │   │   ├── seed1842
    │   │   │   ├── seed1842gr.ini
    │   │   │   └── seed1842newton.ini
    │   │   ├── seed1843
    │   │   │   ├── seed1843gr.ini
    │   │   │   └── seed1843newton.ini
    │   │   ├── seed1844
    │   │   │   ├── seed1844gr.ini
    │   │   │   └── seed1844newton.ini
    │   │   ├── seed1845
    │   │   │   ├── seed1845gr.ini
    │   │   │   └── seed1845newton.ini
    │   │   ├── seed1846
    │   │   │   ├── seed1846gr.ini
    │   │   │   └── seed1846newton.ini
    │   │   ├── seed1847
    │   │   │   ├── seed1847gr.ini
    │   │   │   └── seed1847newton.ini
    │   │   ├── seed1848
    │   │   │   ├── seed1848gr.ini
    │   │   │   └── seed1848newton.ini
    │   │   ├── seed1849
    │   │   │   ├── seed1849gr.ini
    │   │   │   └── seed1849newton.ini
    │   │   ├── seed1850
    │   │   │   ├── seed1850gr.ini
    │   │   │   └── seed1850newton.ini
    │   │   ├── seed1851
    │   │   │   ├── seed1851gr.ini
    │   │   │   └── seed1851newton.ini
    │   │   ├── seed1852
    │   │   │   ├── seed1852gr.ini
    │   │   │   └── seed1852newton.ini
    │   │   ├── seed1853
    │   │   │   ├── seed1853gr.ini
    │   │   │   └── seed1853newton.ini
    │   │   ├── seed1854
    │   │   │   ├── seed1854gr.ini
    │   │   │   └── seed1854newton.ini
    │   │   ├── seed1855
    │   │   │   ├── seed1855gr.ini
    │   │   │   └── seed1855newton.ini
    │   │   ├── seed1856
    │   │   │   ├── seed1856gr.ini
    │   │   │   └── seed1856newton.ini
    │   │   ├── seed1857
    │   │   │   ├── seed1857gr.ini
    │   │   │   └── seed1857newton.ini
    │   │   ├── seed1858
    │   │   │   ├── seed1858gr.ini
    │   │   │   └── seed1858newton.ini
    │   │   ├── seed1859
    │   │   │   ├── seed1859gr.ini
    │   │   │   └── seed1859newton.ini
    │   │   ├── seed1860
    │   │   │   ├── seed1860gr.ini
    │   │   │   └── seed1860newton.ini
    │   │   ├── seed1861
    │   │   │   ├── seed1861gr.ini
    │   │   │   └── seed1861newton.ini
    │   │   ├── seed1862
    │   │   │   ├── seed1862gr.ini
    │   │   │   └── seed1862newton.ini
    │   │   ├── seed1863
    │   │   │   ├── seed1863gr.ini
    │   │   │   └── seed1863newton.ini
    │   │   ├── seed1864
    │   │   │   ├── seed1864gr.ini
    │   │   │   └── seed1864newton.ini
    │   │   ├── seed1865
    │   │   │   ├── seed1865gr.ini
    │   │   │   └── seed1865newton.ini
    │   │   ├── seed1866
    │   │   │   ├── seed1866gr.ini
    │   │   │   └── seed1866newton.ini
    │   │   ├── seed1867
    │   │   │   ├── seed1867gr.ini
    │   │   │   └── seed1867newton.ini
    │   │   ├── seed1868
    │   │   │   ├── seed1868gr.ini
    │   │   │   └── seed1868newton.ini
    │   │   ├── seed1869
    │   │   │   ├── seed1869gr.ini
    │   │   │   └── seed1869newton.ini
    │   │   ├── seed1870
    │   │   │   ├── seed1870gr.ini
    │   │   │   └── seed1870newton.ini
    │   │   ├── seed1871
    │   │   │   ├── seed1871gr.ini
    │   │   │   └── seed1871newton.ini
    │   │   ├── seed1872
    │   │   │   ├── seed1872gr.ini
    │   │   │   └── seed1872newton.ini
    │   │   ├── seed1873
    │   │   │   ├── seed1873gr.ini
    │   │   │   └── seed1873newton.ini
    │   │   ├── seed1874
    │   │   │   ├── seed1874gr.ini
    │   │   │   └── seed1874newton.ini
    │   │   ├── seed1875
    │   │   │   ├── seed1875gr.ini
    │   │   │   └── seed1875newton.ini
    │   │   ├── seed1876
    │   │   │   ├── seed1876gr.ini
    │   │   │   └── seed1876newton.ini
    │   │   ├── seed1877
    │   │   │   ├── seed1877gr.ini
    │   │   │   └── seed1877newton.ini
    │   │   ├── seed1878
    │   │   │   ├── seed1878gr.ini
    │   │   │   └── seed1878newton.ini
    │   │   ├── seed1879
    │   │   │   ├── seed1879gr.ini
    │   │   │   └── seed1879newton.ini
    │   │   ├── seed1880
    │   │   │   ├── seed1880gr.ini
    │   │   │   └── seed1880newton.ini
    │   │   ├── seed1881
    │   │   │   ├── seed1881gr.ini
    │   │   │   └── seed1881newton.ini
    │   │   ├── seed1882
    │   │   │   ├── seed1882gr.ini
    │   │   │   └── seed1882newton.ini
    │   │   ├── seed1883
    │   │   │   ├── seed1883gr.ini
    │   │   │   └── seed1883newton.ini
    │   │   ├── seed1884
    │   │   │   ├── seed1884gr.ini
    │   │   │   └── seed1884newton.ini
    │   │   ├── seed1885
    │   │   │   ├── seed1885gr.ini
    │   │   │   └── seed1885newton.ini
    │   │   ├── seed1886
    │   │   │   ├── seed1886gr.ini
    │   │   │   └── seed1886newton.ini
    │   │   ├── seed1887
    │   │   │   ├── seed1887gr.ini
    │   │   │   └── seed1887newton.ini
    │   │   ├── seed1888
    │   │   │   ├── seed1888gr.ini
    │   │   │   └── seed1888newton.ini
    │   │   ├── seed1889
    │   │   │   ├── seed1889gr.ini
    │   │   │   └── seed1889newton.ini
    │   │   ├── seed1890
    │   │   │   ├── seed1890gr.ini
    │   │   │   └── seed1890newton.ini
    │   │   ├── seed1891
    │   │   │   ├── seed1891gr.ini
    │   │   │   └── seed1891newton.ini
    │   │   ├── seed1892
    │   │   │   ├── seed1892gr.ini
    │   │   │   └── seed1892newton.ini
    │   │   ├── seed1893
    │   │   │   ├── seed1893gr.ini
    │   │   │   └── seed1893newton.ini
    │   │   ├── seed1894
    │   │   │   ├── seed1894gr.ini
    │   │   │   └── seed1894newton.ini
    │   │   ├── seed1895
    │   │   │   ├── seed1895gr.ini
    │   │   │   └── seed1895newton.ini
    │   │   ├── seed1896
    │   │   │   ├── seed1896gr.ini
    │   │   │   └── seed1896newton.ini
    │   │   ├── seed1897
    │   │   │   ├── seed1897gr.ini
    │   │   │   └── seed1897newton.ini
    │   │   ├── seed1898
    │   │   │   ├── seed1898gr.ini
    │   │   │   └── seed1898newton.ini
    │   │   ├── seed1899
    │   │   │   ├── seed1899gr.ini
    │   │   │   └── seed1899newton.ini
    │   │   ├── seed1900
    │   │   │   ├── seed1900gr.ini
    │   │   │   └── seed1900newton.ini
    │   │   ├── seed1901
    │   │   │   ├── seed1901gr.ini
    │   │   │   └── seed1901newton.ini
    │   │   ├── seed1902
    │   │   │   ├── seed1902gr.ini
    │   │   │   └── seed1902newton.ini
    │   │   ├── seed1903
    │   │   │   ├── seed1903gr.ini
    │   │   │   └── seed1903newton.ini
    │   │   ├── seed1904
    │   │   │   ├── seed1904gr.ini
    │   │   │   └── seed1904newton.ini
    │   │   ├── seed1905
    │   │   │   ├── seed1905gr.ini
    │   │   │   └── seed1905newton.ini
    │   │   ├── seed1906
    │   │   │   ├── seed1906gr.ini
    │   │   │   └── seed1906newton.ini
    │   │   ├── seed1907
    │   │   │   ├── seed1907gr.ini
    │   │   │   └── seed1907newton.ini
    │   │   ├── seed1908
    │   │   │   ├── seed1908gr.ini
    │   │   │   └── seed1908newton.ini
    │   │   ├── seed1909
    │   │   │   ├── seed1909gr.ini
    │   │   │   └── seed1909newton.ini
    │   │   ├── seed1910
    │   │   │   ├── seed1910gr.ini
    │   │   │   └── seed1910newton.ini
    │   │   ├── seed1911
    │   │   │   ├── seed1911gr.ini
    │   │   │   └── seed1911newton.ini
    │   │   ├── seed1912
    │   │   │   ├── seed1912gr.ini
    │   │   │   └── seed1912newton.ini
    │   │   ├── seed1913
    │   │   │   ├── seed1913gr.ini
    │   │   │   └── seed1913newton.ini
    │   │   ├── seed1914
    │   │   │   ├── seed1914gr.ini
    │   │   │   └── seed1914newton.ini
    │   │   ├── seed1915
    │   │   │   ├── seed1915gr.ini
    │   │   │   └── seed1915newton.ini
    │   │   ├── seed1916
    │   │   │   ├── seed1916gr.ini
    │   │   │   └── seed1916newton.ini
    │   │   ├── seed1917
    │   │   │   ├── seed1917gr.ini
    │   │   │   └── seed1917newton.ini
    │   │   ├── seed1918
    │   │   │   ├── seed1918gr.ini
    │   │   │   └── seed1918newton.ini
    │   │   ├── seed1919
    │   │   │   ├── seed1919gr.ini
    │   │   │   └── seed1919newton.ini
    │   │   ├── seed1920
    │   │   │   ├── seed1920gr.ini
    │   │   │   └── seed1920newton.ini
    │   │   ├── seed1921
    │   │   │   ├── seed1921gr.ini
    │   │   │   └── seed1921newton.ini
    │   │   ├── seed1922
    │   │   │   ├── seed1922gr.ini
    │   │   │   └── seed1922newton.ini
    │   │   ├── seed1923
    │   │   │   ├── seed1923gr.ini
    │   │   │   └── seed1923newton.ini
    │   │   ├── seed1924
    │   │   │   ├── seed1924gr.ini
    │   │   │   └── seed1924newton.ini
    │   │   ├── seed1925
    │   │   │   ├── seed1925gr.ini
    │   │   │   └── seed1925newton.ini
    │   │   ├── seed1926
    │   │   │   ├── seed1926gr.ini
    │   │   │   └── seed1926newton.ini
    │   │   ├── seed1927
    │   │   │   ├── seed1927gr.ini
    │   │   │   └── seed1927newton.ini
    │   │   ├── seed1928
    │   │   │   ├── seed1928gr.ini
    │   │   │   └── seed1928newton.ini
    │   │   ├── seed1929
    │   │   │   ├── seed1929gr.ini
    │   │   │   └── seed1929newton.ini
    │   │   ├── seed1930
    │   │   │   ├── seed1930gr.ini
    │   │   │   └── seed1930newton.ini
    │   │   ├── seed1931
    │   │   │   ├── seed1931gr.ini
    │   │   │   └── seed1931newton.ini
    │   │   ├── seed1932
    │   │   │   ├── seed1932gr.ini
    │   │   │   └── seed1932newton.ini
    │   │   ├── seed1933
    │   │   │   ├── seed1933gr.ini
    │   │   │   └── seed1933newton.ini
    │   │   ├── seed1934
    │   │   │   ├── seed1934gr.ini
    │   │   │   └── seed1934newton.ini
    │   │   ├── seed1935
    │   │   │   ├── seed1935gr.ini
    │   │   │   └── seed1935newton.ini
    │   │   ├── seed1936
    │   │   │   ├── seed1936gr.ini
    │   │   │   └── seed1936newton.ini
    │   │   ├── seed1937
    │   │   │   ├── seed1937gr.ini
    │   │   │   └── seed1937newton.ini
    │   │   ├── seed1938
    │   │   │   ├── seed1938gr.ini
    │   │   │   └── seed1938newton.ini
    │   │   ├── seed1939
    │   │   │   ├── seed1939gr.ini
    │   │   │   └── seed1939newton.ini
    │   │   ├── seed1940
    │   │   │   ├── seed1940gr.ini
    │   │   │   └── seed1940newton.ini
    │   │   ├── seed1941
    │   │   │   ├── seed1941gr.ini
    │   │   │   └── seed1941newton.ini
    │   │   ├── seed1942
    │   │   │   ├── seed1942gr.ini
    │   │   │   └── seed1942newton.ini
    │   │   ├── seed1943
    │   │   │   ├── seed1943gr.ini
    │   │   │   └── seed1943newton.ini
    │   │   ├── seed1944
    │   │   │   ├── seed1944gr.ini
    │   │   │   └── seed1944newton.ini
    │   │   ├── seed1945
    │   │   │   ├── seed1945gr.ini
    │   │   │   └── seed1945newton.ini
    │   │   ├── seed1946
    │   │   │   ├── seed1946gr.ini
    │   │   │   └── seed1946newton.ini
    │   │   ├── seed1947
    │   │   │   ├── seed1947gr.ini
    │   │   │   └── seed1947newton.ini
    │   │   ├── seed1948
    │   │   │   ├── seed1948gr.ini
    │   │   │   └── seed1948newton.ini
    │   │   ├── seed1949
    │   │   │   ├── seed1949gr.ini
    │   │   │   └── seed1949newton.ini
    │   │   ├── seed1950
    │   │   │   ├── seed1950gr.ini
    │   │   │   └── seed1950newton.ini
    │   │   ├── seed1951
    │   │   │   ├── seed1951gr.ini
    │   │   │   └── seed1951newton.ini
    │   │   ├── seed1952
    │   │   │   ├── seed1952gr.ini
    │   │   │   └── seed1952newton.ini
    │   │   ├── seed1953
    │   │   │   ├── seed1953gr.ini
    │   │   │   └── seed1953newton.ini
    │   │   ├── seed1954
    │   │   │   ├── seed1954gr.ini
    │   │   │   └── seed1954newton.ini
    │   │   ├── seed1955
    │   │   │   ├── seed1955gr.ini
    │   │   │   └── seed1955newton.ini
    │   │   ├── seed1956
    │   │   │   ├── seed1956gr.ini
    │   │   │   └── seed1956newton.ini
    │   │   ├── seed1957
    │   │   │   ├── seed1957gr.ini
    │   │   │   └── seed1957newton.ini
    │   │   ├── seed1958
    │   │   │   ├── seed1958gr.ini
    │   │   │   └── seed1958newton.ini
    │   │   ├── seed1959
    │   │   │   ├── seed1959gr.ini
    │   │   │   └── seed1959newton.ini
    │   │   ├── seed1960
    │   │   │   ├── seed1960gr.ini
    │   │   │   └── seed1960newton.ini
    │   │   ├── seed1961
    │   │   │   ├── seed1961gr.ini
    │   │   │   └── seed1961newton.ini
    │   │   ├── seed1962
    │   │   │   ├── seed1962gr.ini
    │   │   │   └── seed1962newton.ini
    │   │   ├── seed1963
    │   │   │   ├── seed1963gr.ini
    │   │   │   └── seed1963newton.ini
    │   │   ├── seed1964
    │   │   │   ├── seed1964gr.ini
    │   │   │   └── seed1964newton.ini
    │   │   ├── seed1965
    │   │   │   ├── seed1965gr.ini
    │   │   │   └── seed1965newton.ini
    │   │   ├── seed1966
    │   │   │   ├── seed1966gr.ini
    │   │   │   └── seed1966newton.ini
    │   │   ├── seed1967
    │   │   │   ├── seed1967gr.ini
    │   │   │   └── seed1967newton.ini
    │   │   ├── seed1968
    │   │   │   ├── seed1968gr.ini
    │   │   │   └── seed1968newton.ini
    │   │   ├── seed1969
    │   │   │   ├── seed1969gr.ini
    │   │   │   └── seed1969newton.ini
    │   │   ├── seed1970
    │   │   │   ├── seed1970gr.ini
    │   │   │   └── seed1970newton.ini
    │   │   ├── seed1971
    │   │   │   ├── seed1971gr.ini
    │   │   │   └── seed1971newton.ini
    │   │   ├── seed1972
    │   │   │   ├── seed1972gr.ini
    │   │   │   └── seed1972newton.ini
    │   │   ├── seed1973
    │   │   │   ├── seed1973gr.ini
    │   │   │   └── seed1973newton.ini
    │   │   ├── seed1974
    │   │   │   ├── seed1974gr.ini
    │   │   │   └── seed1974newton.ini
    │   │   ├── seed1975
    │   │   │   ├── seed1975gr.ini
    │   │   │   └── seed1975newton.ini
    │   │   ├── seed1976
    │   │   │   ├── seed1976gr.ini
    │   │   │   └── seed1976newton.ini
    │   │   ├── seed1977
    │   │   │   ├── seed1977gr.ini
    │   │   │   └── seed1977newton.ini
    │   │   ├── seed1978
    │   │   │   ├── seed1978gr.ini
    │   │   │   └── seed1978newton.ini
    │   │   ├── seed1979
    │   │   │   ├── seed1979gr.ini
    │   │   │   └── seed1979newton.ini
    │   │   ├── seed1980
    │   │   │   ├── seed1980gr.ini
    │   │   │   └── seed1980newton.ini
    │   │   ├── seed1981
    │   │   │   ├── seed1981gr.ini
    │   │   │   └── seed1981newton.ini
    │   │   ├── seed1982
    │   │   │   ├── seed1982gr.ini
    │   │   │   └── seed1982newton.ini
    │   │   ├── seed1983
    │   │   │   ├── seed1983gr.ini
    │   │   │   └── seed1983newton.ini
    │   │   ├── seed1984
    │   │   │   ├── seed1984gr.ini
    │   │   │   └── seed1984newton.ini
    │   │   ├── seed1985
    │   │   │   ├── seed1985gr.ini
    │   │   │   └── seed1985newton.ini
    │   │   ├── seed1986
    │   │   │   ├── seed1986gr.ini
    │   │   │   └── seed1986newton.ini
    │   │   ├── seed1987
    │   │   │   ├── seed1987gr.ini
    │   │   │   └── seed1987newton.ini
    │   │   ├── seed1988
    │   │   │   ├── seed1988gr.ini
    │   │   │   └── seed1988newton.ini
    │   │   ├── seed1989
    │   │   │   ├── seed1989gr.ini
    │   │   │   └── seed1989newton.ini
    │   │   ├── seed1990
    │   │   │   ├── seed1990gr.ini
    │   │   │   └── seed1990newton.ini
    │   │   ├── seed1991
    │   │   │   ├── seed1991gr.ini
    │   │   │   └── seed1991newton.ini
    │   │   ├── seed1992
    │   │   │   ├── seed1992gr.ini
    │   │   │   └── seed1992newton.ini
    │   │   ├── seed1993
    │   │   │   ├── seed1993gr.ini
    │   │   │   └── seed1993newton.ini
    │   │   ├── seed1994
    │   │   │   ├── seed1994gr.ini
    │   │   │   └── seed1994newton.ini
    │   │   ├── seed1995
    │   │   │   ├── seed1995gr.ini
    │   │   │   └── seed1995newton.ini
    │   │   ├── seed1996
    │   │   │   ├── seed1996gr.ini
    │   │   │   └── seed1996newton.ini
    │   │   ├── seed1997
    │   │   │   ├── seed1997gr.ini
    │   │   │   └── seed1997newton.ini
    │   │   ├── seed1998
    │   │   │   ├── seed1998gr.ini
    │   │   │   └── seed1998newton.ini
    │   │   ├── seed1999
    │   │   │   ├── seed1999gr.ini
    │   │   │   └── seed1999newton.ini
    │   │   └── tmp.ini
    │   ├── README.md
    │   ├── seeds_0000_0749.txt
    │   ├── seeds_0050_0149.txt
    │   ├── seeds_0150_0249.txt
    │   ├── seeds_0250_0349.txt
    │   ├── seeds_0350_0449.txt
    │   ├── seeds_0450_0549.txt
    │   ├── seeds_0550_0649.txt
    │   ├── seeds_0650_0749.txt
    │   ├── seeds_0750_0999.txt
    │   ├── seeds_0_9.txt
    │   ├── seeds_1000_1249.txt
    │   ├── seeds_10_19.txt
    │   ├── seeds_1250_1499.txt
    │   ├── seeds_1500_1749.txt
    │   ├── seeds_1750_1999.txt
    │   ├── seeds_1_to_200.txt
    │   ├── seeds_1_to_50.txt
    │   ├── seeds_20_29.txt
    │   ├── seeds_30_39.txt
    │   ├── seeds_40_49.txt
    │   ├── seeds.txt
    │   ├── simulate.sh
    │   ├── simulations_run.txt
    │   ├── temp_del_seeds.txt
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
    │   │   │   ├── animations
    │   │   │   │   └── cube_1234_gr_20.mp4
    │   │   │   ├── bispectrum.py
    │   │   │   ├── cambPK.py
    │   │   │   ├── cubehandler.py
    │   │   │   ├── cube.py
    │   │   │   ├── DataHandler.py
    │   │   │   ├── __init__.py
    │   │   │   ├── old_scripts
    │   │   │   │   ├── h5collection.py
    │   │   │   │   ├── h5cube.py
    │   │   │   │   └── h5dataset.py
    │   │   │   ├── plotPS.py
    │   │   │   ├── powerspectra.py
    │   │   │   ├── __pycache__
    │   │   │   │   ├── cambPK.cpython-38.pyc
    │   │   │   │   ├── cube.cpython-38.pyc
    │   │   │   │   ├── h5collection.cpython-38.pyc
    │   │   │   │   ├── h5cube.cpython-38.pyc
    │   │   │   │   ├── h5dataset.cpython-38.pyc
    │   │   │   │   ├── __init__.cpython-38.pyc
    │   │   │   │   └── powerspectra.cpython-38.pyc
    │   │   │   ├── README.md
    │   │   │   └── visualiseCube.py
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
    
    2016 directories, 4104 files
Updated on 2023-08-14
## Subheading about something
sometext testing

