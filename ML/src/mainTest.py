
# from src.network.model import Model
# from src.network.COW import COW

from network import model
from network import COW

import os
from torchsummary import summary


if __name__=="__main__":
    output_path = os.path.abspath("").replace("Summer-Sandbox23/ML/src", "NbodySimulation/gevolution-1.2/output")
    run_type = "/intermediate"

    path = output_path + run_type
    mod = model.Model(COW.COW, path)
    summary(mod.model, (2, 128, 128))
    mod.loadData(stride=2)
    mod.train()
    mod.test()
    
    # print(path)