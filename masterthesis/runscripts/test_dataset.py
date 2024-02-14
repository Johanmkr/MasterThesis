import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# add path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

from src.data import calc_mean_std as cms

from IPython import embed


if __name__ == "__main__":
    print("Calculating mean")
    redshift = float(input("Redshift (0,1,5,10,15,20): "))
    if input("Save statistics? (y/n): ") == "y":
        cms.save_statistics(redshift)
    else:
        print("Not saving statistics.")
