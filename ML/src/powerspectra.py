import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from IPython import embed


output_path = os.path.abspath("").replace("Summer-Sandbox23/ML/src", "NbodySimulation/gevolution-1.2/output")
run_type = "/intermediate"
gr = "/gr"
newton = "/newton"

pk_name = "/lcdm_pk000_phi.dat"

path_gr = output_path + run_type + gr + pk_name
path_newton = output_path + run_type + newton + pk_name 

# path_to_newton_spectrum = 


# Use read_table() to read the .dat file with space-separated values
newton_pk = pd.read_table(path_newton, sep='\s+', comment='#', header=None)
gr_pk = pd.read_table(path_gr, sep='\s+', comment='#', header=None)

# If you want to give column names to the DataFrame, you can do this:
column_names = ['k', 'Pk', 'sigma_k', 'sigma_pk', 'count']
newton_pk.columns = column_names
gr_pk.columns = column_names

nk = newton_pk["k"]
gk = gr_pk["k"]

nPk = newton_pk["Pk"] * newton_pk["count"]
gPk = gr_pk["Pk"] * gr_pk["count"]

fig, ax = plt.subplots()
ax.plot(nk, nPk, label="newton")
ax.plot(gk, gPk, label="gr")
ax.set_xlabel("k")
ax.set_ylabel("Pk")
fig.legend()

fig, ax = plt.subplots()
ax.plot(nk, np.abs(nPk-gPk), label="diff")
ax.set_xlabel("k")
ax.set_ylabel("Pk")
fig.legend()


plt.show()

if __name__=="__main__":
    pass