import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# add path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)

import src.features.produce_bispectra as pb


seedfile = input("Enter seedfile: ")

pb.produce_bispectra("production_seeds/" + seedfile)
