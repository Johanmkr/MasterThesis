import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import figure as fg

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from IPython import embed
import os, sys

# Parent forlder to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import NEW_CNN.train_utils as tutils

########################## CNN DATA ##########################
MODEL_NAME = "T2"
inference_data_path = f"/uio/hume/student-u00/johanmkr/Documents/thesis/masterthesis/NEW_CNN/inference_data/{MODEL_NAME}/"

# Load S1 data
S1_loss = np.load(inference_data_path + f"loss_S1.npy")
(TP, TN, FP, FN) = np.load(inference_data_path + f"cf_metrics_S1.npy")
S1_predicted_scores = np.load(inference_data_path + f"predicted_scores_S1.npy")
S1_predictions = np.load(inference_data_path + f"predictions_S1.npy")
S1_labels = np.load(inference_data_path + f"labels_S1.npy")

# Load S2 data
S2_loss = np.load(inference_data_path + f"loss_S2.npy")
(TP, TN, FP, FN) = np.load(inference_data_path + f"cf_metrics_S2.npy")
S2_predicted_scores = np.load(inference_data_path + f"predicted_scores_S2.npy")
S2_predictions = np.load(inference_data_path + f"predictions_S2.npy")
S2_labels = np.load(inference_data_path + f"labels_S2.npy")

########################## FCNN DATA ##########################
FCNN_S1_data = np.loadtxt("../Pk_analysis/S1_results.txt", skiprows=1)
FCNN_S2_data = np.loadtxt("../Pk_analysis/S2_results.txt", skiprows=1)
FCNN_S1_frame = pd.DataFrame(
    FCNN_S1_data, columns=["score", "prediction", "true_values", "loss"]
)
FCNN_S2_frame = pd.DataFrame(
    FCNN_S2_data, columns=["score", "prediction", "true_values", "loss"]
)

# Apply tolerances to the predictions
TOLERANCE = 0.5
FCNN_S1_frame["bool_pred"] = (FCNN_S1_frame["prediction"] > TOLERANCE).astype(bool)
FCNN_S1_frame["int_pred"] = (FCNN_S1_frame["prediction"] > TOLERANCE).astype(int)
FCNN_S1_frame["bool_true"] = (FCNN_S1_frame["true_values"] > TOLERANCE).astype(bool)
FCNN_S2_frame["bool_pred"] = (FCNN_S2_frame["prediction"] > TOLERANCE).astype(bool)
FCNN_S2_frame["int_pred"] = (FCNN_S2_frame["prediction"] > TOLERANCE).astype(int)
FCNN_S2_frame["bool_true"] = (FCNN_S2_frame["true_values"] > TOLERANCE).astype(bool)

cfm1 = confusion_matrix(FCNN_S1_frame["int_pred"], FCNN_S1_frame["true_values"])
tn1, fp1, fn1, tp1 = cfm1.ravel()
accuracy1, precision1, recall1, F11, TPR1, FPR1 = tutils.calculate_metrics(
    TP=tp1, TN=tn1, FP=fp1, FN=fn1
)
fpr1, tpr1, thresholds1 = roc_curve(
    FCNN_S1_frame["true_values"], FCNN_S1_frame["prediction"]
)

# For S1
cfm2 = confusion_matrix(FCNN_S2_frame["int_pred"], FCNN_S2_frame["true_values"])
tn2, fp2, fn2, tp2 = cfm2.ravel()
accuracy2, precision2, recall2, F12, TPR2, FPR2 = tutils.calculate_metrics(
    TP=tp2, TN=tn2, FP=fp2, FN=fn2
)
fpr2, tpr2, thresholds2 = roc_curve(
    FCNN_S2_frame["true_values"], FCNN_S2_frame["prediction"]
)
######################################################################################
# Create latex names
S1_name = r"$\mathcal{D}_\mathrm{test}$"
S2_name = r"$\mathcal{D}_\mathrm{val}$"


def plot_confusion_matrices():

    S1_cfm = confusion_matrix(S1_labels, S1_predictions)
    S2_cfm = confusion_matrix(S2_labels, S2_predictions)

    S1_disp = ConfusionMatrixDisplay(
        confusion_matrix=S1_cfm, display_labels=["Newton", "GR"]
    )
    S2_disp = ConfusionMatrixDisplay(
        confusion_matrix=S2_cfm, display_labels=["Newton", "GR"]
    )

    FCNN_S1_disp = ConfusionMatrixDisplay(
        confusion_matrix=cfm1, display_labels=["Newton", "GR"]
    )
    FCNN_S2_disp = ConfusionMatrixDisplay(
        confusion_matrix=cfm2, display_labels=["Newton", "GR"]
    )

    fig, ax = plt.subplots(
        2, 2, figsize=(15, 15), sharex=True, gridspec_kw={"hspace": 0.0}
    )

    # Disable colorbars

    S1_disp.plot(ax=ax[0][0], cmap="Greens")
    S2_disp.plot(ax=ax[0][1], cmap="Greens")

    S1_disp.im_.colorbar.remove()
    S2_disp.im_.colorbar.remove()

    FCNN_S1_disp.plot(ax=ax[1][0], cmap="Oranges", values_format="d")
    FCNN_S2_disp.plot(ax=ax[1][1], cmap="Oranges", values_format="d")

    FCNN_S1_disp.im_.colorbar.remove()
    FCNN_S2_disp.im_.colorbar.remove()

    # Disable y label and y ticks for second (right) plot
    ax[0][1].set_ylabel("")
    ax[0][1].set_yticks([])
    ax[1][1].set_ylabel("")
    ax[1][1].set_yticks([])

    ax[0][0].set_xlabel("")
    # ax[0][0].set_xticks([])
    ax[0][1].set_xlabel("")
    # ax[0][1].set_xticks([])

    # ax[1][0].set_xlabel(["Newton", "GR"])

    # Set labels
    ax[0][0].set_title(S1_name)
    ax[0][1].set_title(S2_name)

    # Name to the right of the y-axis
    ax[0][1].text(
        1.05,
        0.5,
        "CNN-2",
        ha="center",
        va="center",
        rotation=90,
        transform=ax[0][1].transAxes,
    )
    ax[1][1].text(
        1.05,
        0.5,
        "FCNN",
        ha="center",
        va="center",
        rotation=90,
        transform=ax[1][1].transAxes,
    )

    fig.suptitle(f"Confusion matrices")
    savename = f"confusion_matrices_both_models.pdf"
    fg.SaveShow(fig=fig, save_name=savename, save=True, show=True, tight_layout=True)


def print_statistics():
    for labels, predictions, name in zip(
        [S1_labels, S2_labels], [S1_predictions, S2_predictions], [S1_name, S2_name]
    ):
        cfm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cfm.ravel()
        accuracy, precision, recall, F1, TPR, FPR = tutils.calculate_metrics(
            TP=tp, TN=tn, FP=fp, FN=fn
        )
        print(f"\nStatistics for {MODEL_NAME} on {name}:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {F1:.3f}")
        print(f"TPR: {TPR:.3f}")
        print(f"FPR: {FPR:.3f}\n")


def plot_roc_curve():

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    # Find data for ROC-curve / AUC for S1 and S2
    s1fpr, s1tpr, s1thresholds = roc_curve(S1_labels, S1_predicted_scores)
    s1_roc_auc = auc(s1fpr, s1tpr)
    s2fpr, s2tpr, s2thresholds = roc_curve(S2_labels, S2_predicted_scores)
    s2_roc_auc = auc(s2fpr, s2tpr)

    # Plot ROC curves
    ax.plot(
        s1fpr,
        s1tpr,
        color="green",
        lw=2,
        ls="solid",
        label=f"CNN-2 on {S1_name} (area = {s1_roc_auc:.2f})",
    )
    ax.plot(
        s2fpr,
        s2tpr,
        color="green",
        lw=2,
        ls="dashed",
        label=f"CNN-2 on {S2_name} (area = {s2_roc_auc:.2f})",
    )

    ax.plot(
        fpr1,
        tpr1,
        color="orange",
        lw=2,
        ls="solid",
        label=f"FCNN on {S1_name} (area = {auc(fpr1, tpr1):.2f})",
    )
    ax.plot(
        fpr2,
        tpr2,
        color="orange",
        lw=2,
        ls="dashed",
        label=f"FCNN on {S2_name} (area = {auc(fpr2, tpr2):.2f})",
    )
    ax.plot([0, 1], [0, 1], color="grey", lw=2, linestyle=":")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.suptitle("Receiver Operating Characteristic for models: CNN-2 and FCNN")
    ax.legend(loc="lower right")

    savename = f"roc_curve_both_models.pdf"
    fg.SaveShow(fig=fig, save_name=savename, save=True, show=True, tight_layout=True)


if __name__ == "__main__":
    plot_confusion_matrices()
    print_statistics()
    plot_roc_curve()
