import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import figure as fg

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from IPython import embed
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import NEW_CNN.train_utils as tutils

# Load data and create dataframes
S1_data = np.loadtxt("../Pk_analysis/3D_S1_results.txt", skiprows=1)
S2_data = np.loadtxt("../Pk_analysis/3D_S2_results.txt", skiprows=1)
S1_frame = pd.DataFrame(S1_data, columns=["score", "prediction", "true_values", "loss"])
S2_frame = pd.DataFrame(S2_data, columns=["score", "prediction", "true_values", "loss"])

# Apply tolerances to the predictions
TOLERANCE = 0.5
S1_frame["bool_pred"] = (S1_frame["prediction"] > TOLERANCE).astype(bool)
S1_frame["int_pred"] = (S1_frame["prediction"] > TOLERANCE).astype(int)
S1_frame["bool_true"] = (S1_frame["true_values"] > TOLERANCE).astype(bool)
S2_frame["bool_pred"] = (S2_frame["prediction"] > TOLERANCE).astype(bool)
S2_frame["int_pred"] = (S2_frame["prediction"] > TOLERANCE).astype(int)
S2_frame["bool_true"] = (S2_frame["true_values"] > TOLERANCE).astype(bool)

# For S1
cfm1 = confusion_matrix(S1_frame["int_pred"], S1_frame["true_values"])
tn1, fp1, fn1, tp1 = cfm1.ravel()
accuracy1, precision1, recall1, F11, TPR1, FPR1 = tutils.calculate_metrics(
    TP=tp1, TN=tn1, FP=fp1, FN=fn1
)
fpr1, tpr1, thresholds1 = roc_curve(S1_frame["true_values"], S1_frame["prediction"])

# For S1
cfm2 = confusion_matrix(S2_frame["int_pred"], S2_frame["true_values"])
tn2, fp2, fn2, tp2 = cfm2.ravel()
accuracy2, precision2, recall2, F12, TPR2, FPR2 = tutils.calculate_metrics(
    TP=tp2, TN=tn2, FP=fp2, FN=fn2
)
fpr2, tpr2, thresholds2 = roc_curve(S2_frame["true_values"], S2_frame["prediction"])


S1_name = r"$\mathcal{D}_\mathrm{test}$"
S2_name = r"$\mathcal{D}_\mathrm{val}$"


def plot_confusion_matrices():
    S1_disp = ConfusionMatrixDisplay(
        confusion_matrix=cfm1, display_labels=["Newton", "GR"]
    )
    S2_disp = ConfusionMatrixDisplay(
        confusion_matrix=cfm2, display_labels=["Newton", "GR"]
    )

    fig, ax = plt.subplots(
        1, 2, figsize=(15, 10), sharex=True, gridspec_kw={"hspace": 0.0}
    )

    # Disable colorbars

    S1_disp.plot(ax=ax[0], cmap="Blues", values_format="d")
    S2_disp.plot(ax=ax[1], cmap="Blues", values_format="d")

    S1_disp.im_.colorbar.remove()
    S2_disp.im_.colorbar.remove()

    ax[1].set_ylabel("")
    ax[1].set_yticks([])

    ax[0].set_title(S1_name)
    ax[1].set_title(S2_name)

    fig.suptitle("Confusion matrices for model: FCNN")

    # Save and show
    savename = "fcnn_confusion_matrices"
    fg.SaveShow(fig=fig, save_name=savename, save=False, show=True, tight_layout=True)


def print_statistics():
    print(f"\nStatistics for FCNN on {S1_name}:")
    print(f"Accuracy: {accuracy1:.3f}")
    print(f"Precision: {precision1:.3f}")
    print(f"Recall: {recall1:.3f}")
    print(f"F1: {F11:.3f}")
    print(f"TPR: {TPR1:.3f}")
    print(f"FPR: {FPR1:.3f}\n")

    print(f"\nStatistics for FCNN on {S2_name}:")
    print(f"Accuracy: {accuracy2:.3f}")
    print(f"Precision: {precision2:.3f}")
    print(f"Recall: {recall2:.3f}")
    print(f"F1: {F12:.3f}")
    print(f"TPR: {TPR2:.3f}")
    print(f"FPR: {FPR2:.3f}\n")


def plot_roc_curve():
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.plot(
        fpr1,
        tpr1,
        color="darkorange",
        lw=2,
        label=f"{S1_name} (area = {auc(fpr1, tpr1):.2f})",
    )
    ax.plot(
        fpr2,
        tpr2,
        color="navy",
        lw=2,
        label=f"{S2_name} (area = {auc(fpr2, tpr2):.2f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic for FCNN")
    ax.legend(loc="lower right")

    # Save and show
    savename = "fcnn_roc_curve"
    fg.SaveShow(fig=fig, save_name=savename, save=False, show=True, tight_layout=True)


if __name__ == "__main__":
    plot_confusion_matrices()
    print_statistics()
    plot_roc_curve()
