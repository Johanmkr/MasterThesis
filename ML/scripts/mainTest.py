
from ..src.network.model import Model
from ..src.network.COW import COW


if __name__=="__main__":
    output_path = os.path.abspath("").replace("Summer-Sandbox23/ML/src/data", "NbodySimulation/gevolution-1.2/output")
    run_type = "/intermediate"

    path = output_path + run_type
    mod = Model(COW, path)
    print(path)