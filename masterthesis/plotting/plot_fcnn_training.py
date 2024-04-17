import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from IPython import embed
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figure as fg

model_nr = 481
model_train_data_path = f"../Pk_analysis/Pk_analysis/losses/loss_{model_nr}.txt"
info = np.loadtxt(model_train_data_path, delimiter=" ", skiprows=0)
epochs = info[:, 0]
train_loss = info[:, 1]
test_loss = info[:, 2]

train_color = "teal"
test_color = "fuchsia"


def plot_loss():
    # Set plots settings
    plot_settings = dict(
        xlabel="Epoch",
        ylabel=r"$\mathcal{L}_{BCE}$",
        aspect="auto",
    )
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.plot(epochs, train_loss, color=train_color, label="Training", ls="solid", lw=2)
    ax.plot(epochs, test_loss, color=test_color, label="Testing", ls="solid", lw=2)
    ax.set(**plot_settings)
    ax.legend()

    fig.suptitle("FCNN loss")

    savename = "fcnn_loss"
    fg.SaveShow(fig=fig, save_name=savename, save=True, show=True, tight_layout=True)


if __name__ == "__main__":
    plot_loss()
