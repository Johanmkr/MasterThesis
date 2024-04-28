import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import figure as fg

from IPython import embed
import os, sys

train_metric_path = "/mn/stornext/d10/data/johanmkr/new_cnn/runs/training_data/"

metric_path = lambda run, tag, train=True: (
    train_metric_path + f"run-T{run}-tag-{tag}_train.csv"
    if train
    else train_metric_path + f"run-T{run}-tag-{tag}_test.csv"
)

tags = ["loss", "accuracy", "F1_score", "TPR", "FPR", "precision", "recall"]

# Empty dataframes:
T1_train_data = pd.DataFrame()
T1_test_data = pd.DataFrame()
T2_train_data = pd.DataFrame()
T2_test_data = pd.DataFrame()

# Set colors
train_color = "teal"
test_color = "fuchsia"

# Fill step column
T1_train_data["step"] = pd.read_csv(metric_path(1, "loss", train=True))["Step"]
T1_test_data["step"] = pd.read_csv(metric_path(1, "loss", train=False))["Step"]
T2_train_data["step"] = pd.read_csv(metric_path(2, "loss", train=True))["Step"]
T2_test_data["step"] = pd.read_csv(metric_path(2, "loss", train=False))["Step"]

# Fill empty dataframes with data
for tag in tags:
    T1_train_data[tag] = pd.read_csv(metric_path(1, tag, train=True))["Value"]
    T1_test_data[tag] = pd.read_csv(metric_path(1, tag, train=False))["Value"]
    T2_train_data[tag] = pd.read_csv(metric_path(2, tag, train=True))["Value"]
    T2_test_data[tag] = pd.read_csv(metric_path(2, tag, train=False))["Value"]

# Set values above outlier_tolerance to nan:
outlier_tolerance = 5
outlier_func = lambda x: outlier_tolerance if x > outlier_tolerance else x
for tag in tags:
    T1_train_data[tag] = T1_train_data[tag].apply(outlier_func)
    T1_test_data[tag] = T1_test_data[tag].apply(outlier_func)
    T2_train_data[tag] = T2_train_data[tag].apply(outlier_func)
    T2_test_data[tag] = T2_test_data[tag].apply(outlier_func)


def plot_loss():
    # Set plots settings
    plot_settings = dict(
        xlabel="Epoch",
        ylabel=r"$\mathcal{L}_{BCE}$",
        aspect="auto",
    )
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 12), sharex=True, gridspec_kw={"hspace": 0.0}
    )

    # Plot T1 data
    ax1.set(**plot_settings)
    ax1.plot(
        T1_train_data["step"],
        T1_train_data["loss"],
        color=train_color,
        lw=2,
        ls="solid",
        label="Training",
    )
    ax1.plot(
        T1_test_data["step"],
        T1_test_data["loss"],
        color=test_color,
        lw=2,
        ls="solid",
        label="Testing",
    )

    # plot T2 data
    ax2.set(**plot_settings)
    ax2.plot(
        T2_train_data["step"],
        T2_train_data["loss"],
        color=train_color,
        lw=2,
        ls="solid",
        label="Training",
    )
    ax2.plot(
        T2_test_data["step"],
        T2_test_data["loss"],
        color=test_color,
        lw=2,
        ls="solid",
        label="Testing",
    )

    # Add figure legend
    figleg = ax2.legend(loc="upper right")

    # Add T1 and T2 text right of y axis
    ax1.text(
        1.01, 0.5, "CNN-1", fontsize=23, transform=ax1.transAxes, ha="left", va="center"
    )
    ax2.text(
        1.01, 0.5, "CNN-2", fontsize=23, transform=ax2.transAxes, ha="left", va="center"
    )

    # Remove top tick from bottom axes
    ax2.set_yticks(ax2.get_yticks()[1:-2])

    # Add vertical line at 44 epochs
    ax2.axvline(x=44, color="k", ls="dashed")

    # Add figure title
    fig.suptitle("CNN loss")

    savename = "loss_comparison"
    fg.SaveShow(fig=fig, save_name=savename, save=True, show=True, tight_layout=True)


def plot_rest_of_metrics():
    # Set plots settings
    plot_settings = dict(
        xlabel="Epoch",
        aspect="auto",
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 12),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.0, "wspace": 0.0},
    )

    # Plot T1 data
    for i, tag in enumerate(["F1_score", "accuracy", "precision", "recall"]):
        ax = axes.flatten()[i]
        ax.set(**plot_settings)
        ax.plot(
            T2_train_data["step"][1:],
            T2_train_data[tag][1:],
            color=train_color,
            lw=2,
            ls="solid",
            label="Training",
        )
        ax.plot(
            T2_test_data["step"],
            T2_test_data[tag],
            color=test_color,
            lw=2,
            ls="solid",
            label="Testing",
        )

        # Text with frame
        ax.text(
            0.85,
            0.10,
            tag.capitalize().replace("_", " "),
            fontsize=23,
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Vertical line at 44 epochs
        ax.axvline(x=44, color="k", ls="dashed")

    # Add figure legend above axis with two columns
    figleg = fig.legend(
        handles=ax.get_lines(),
        labels=["Training", "Testing"],
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.98, 0.965),
    )

    fig.suptitle("CNN training metrics")

    # Save figure
    savename = "metrics_T2"
    fg.SaveShow(fig=fig, save_name=savename, save=True, show=True, tight_layout=True)


if __name__ == "__main__":
    plot_loss()
    plot_rest_of_metrics()
