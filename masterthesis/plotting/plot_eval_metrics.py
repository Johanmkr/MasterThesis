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

MODEL_NAME = "T2"
inference_data_path = f"/uio/hume/student-u00/johanmkr/Documents/thesis/masterthesis/NEW_CNN/inference_data/{MODEL_NAME}/"

# Load S1 data
S1_loss = np.load(inference_data_path + f"loss_S1.npy")
(TP, TN, FP, FN) = np.load(inference_data_path + f"cf_metrics_S1.npy")
S1_predicted_scores = np.load(
    inference_data_path + f"predicted_scores_S1.npy"
)
S1_predictions = np.load(inference_data_path + f"predictions_S1.npy")
S1_labels = np.load(inference_data_path + f"labels_S1.npy")

# Load S2 data
S2_loss = np.load(inference_data_path + f"loss_S2.npy")
(TP, TN, FP, FN) = np.load(inference_data_path + f"cf_metrics_S2.npy")
S2_predicted_scores = np.load(
    inference_data_path + f"predicted_scores_S2.npy"
)
S2_predictions = np.load(inference_data_path + f"predictions_S2.npy")
S2_labels = np.load(inference_data_path + f"labels_S2.npy")

# Create latex names
S1_name = r"$\mathtt{S}_1$" 
S2_name = r"$\mathtt{S}_2$"


def plot_confusion_matrices():
    
    S1_cfm = confusion_matrix(S1_labels, S1_predictions)
    S2_cfm = confusion_matrix(S2_labels, S2_predictions)
    
    S1_disp = ConfusionMatrixDisplay(confusion_matrix=S1_cfm, display_labels=["Newton", "GR"])
    S2_disp = ConfusionMatrixDisplay(confusion_matrix=S2_cfm, display_labels=["Newton", "GR"])
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 10), sharex=True, gridspec_kw={'hspace': 0.0})
    
    # Disable colorbars
    
    S1_disp.plot(ax=ax[0], cmap="Blues")
    S2_disp.plot(ax=ax[1], cmap="Blues")
    
    S1_disp.im_.colorbar.remove()
    S2_disp.im_.colorbar.remove()
    
    # Disable y label and y ticks for second (right) plot
    ax[1].set_ylabel("")
    ax[1].set_yticks([])
    
    # Set labels
    ax[0].set_title(S1_name)
    ax[1].set_title(S2_name)
    
    fig.suptitle(f"Confusion matrices for model: {MODEL_NAME}")
    savename = f"{MODEL_NAME}_confusion_matrix.pdf"
    fg.SaveShow(fig=fig, save_name=savename, save=True, show=True, tight_layout=True)

def print_statistics():
    for labels, predictions, name in zip([S1_labels, S2_labels], [S1_predictions, S2_predictions], [S1_name, S2_name]):
        cfm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cfm.ravel()
        accuracy, precision, recall, F1, TPR, FPR = tutils.calculate_metrics(TP=tp, TN=tn, FP=fp, FN=fn)
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
    ax.plot(s1fpr, s1tpr, color="darkorange", lw=2, label=f"{S1_name} ROC curve (area = {s1_roc_auc:.2f})")
    ax.plot(s2fpr, s2tpr, color="navy", lw=2, label=f"{S2_name} ROC curve (area = {s2_roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.suptitle("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    
    savename = f"{MODEL_NAME}_roc_curve.pdf"
    fg.SaveShow(fig=fig, save_name=savename, save=True, show=True, tight_layout=True)
    

if __name__=="__main__":
    plot_confusion_matrices()
    print_statistics()
    plot_roc_curve()