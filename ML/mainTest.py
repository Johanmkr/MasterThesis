from src.MLutils import *
from src.model import Model
from src.COW import COW


if __name__=="__main__":
    DataPath = os.path.abspath("").replace("Summer-Sandbox23/ML", "NbodySimulation/gevolution-1.2/output/")
    print(DataPath)
    TEST = Model(COW, DataPath)
    TEST.loadData()
    TEST.train()
    TEST.test()
    TEST.model.printSummary()