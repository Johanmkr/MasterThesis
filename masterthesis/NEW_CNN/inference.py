# Import
import os
import torch
import torch.nn as nn
import numpy as np
from IPython import embed
from data import ScaledData
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import train_utils as tutils
from architecture import model_o3_err
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# imshow lower
plt.rcParams["image.origin"] = "lower"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# Model
MODEL_NAME = "T1"
MODEL_PATH = f"models/{MODEL_NAME}.pt"
BATCH_SIZE = 342 if MODEL_NAME == "T2" else 256
NUM_WORKERS = 4

A_s = 2.215e-9
(mean, var) = np.load(f"mean_var_A_s{A_s:.3e}.npy")
norm = mcolors.Normalize(vmin=-5, vmax=5)
S_val = "S1"  # S1 for data tested on, S2 for newly run data

# Load model
if MODEL_NAME == "T1":
    HIDDEN = 8  # Hidden layer size
    DR = 0.15527713188782274  # Dropout rate
elif MODEL_NAME == "T2":
    # Tunable parameters
    HIDDEN = 6  # Hidden layer size
    DR = 0.40189871215174466  # Dropout rate

state = tutils.get_state(MODEL_PATH, load_model=True)
epochs_trained = state["epoch"]
model = model_o3_err(HIDDEN, DR)
try:
    model.load_state_dict(state["model_state_dict"])
    print(f"Model successfully initialized from {MODEL_PATH}")
except FileNotFoundError:
    print("Model state not found")
loss_fn = nn.BCEWithLogitsLoss()
model.to(device)


# try:
#     loss = np.load(f"inference_data/{MODEL_NAME}/loss_{S_val}.npy")
#     (TP, TN, FP, FN) = np.load(f"inference_data/{MODEL_NAME}/cf_metrics_{S_val}.npy")
#     predicted_scores = np.load(
#         f"inference_data/{MODEL_NAME}/predicted_scores_{S_val}.npy"
#     )
#     predictions = np.load(f"inference_data/{MODEL_NAME}/predictions_{S_val}.npy")
#     labels = np.load(f"inference_data/{MODEL_NAME}/labels_{S_val}.npy")
# except FileNotFoundError:
#     # Load some inference data
#     seed_range = np.arange(200, 250) if S_val == "S1" else np.arange(250, 300)
#     inf_data = ScaledData(seeds=seed_range, train=False, mean=mean, variance=var)
#     inf_loader = DataLoader(
#         inf_data,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=True,
#     )
#     # Infer on whole inf-data
#     print("Infering on data: ", S_val, seed_range)
#     loss, TP, TN, FP, FN, predicted_scores, labels = tutils.infer(
#         device=device, model=model, inf_loader=inf_loader, loss_fn=loss_fn
#     )
#     # Calculate predictions
#     predictions = predicted_scores >= 0.5

#     # Move to cpu and make numpy arrays
#     predicted_scores = predicted_scores.cpu().detach().numpy()
#     labels = labels.cpu().detach().numpy()
#     predictions = predictions.cpu().detach().numpy()

#     # Check if directory exists
#     if not os.path.exists(f"inference_data/{MODEL_NAME}"):
#         os.makedirs(f"inference_data/{MODEL_NAME}")
#     # Save models
#     np.save(f"inference_data/{MODEL_NAME}/loss_{S_val}.npy", loss)
#     np.save(
#         f"inference_data/{MODEL_NAME}/cf_metrics_{S_val}.npy",
#         np.array([TP, TN, FP, FN]),
#     )
#     np.save(
#         f"inference_data/{MODEL_NAME}/predicted_scores_{S_val}.npy", predicted_scores
#     )
#     np.save(f"inference_data/{MODEL_NAME}/predictions_{S_val}.npy", predictions)
#     np.save(f"inference_data/{MODEL_NAME}/labels_{S_val}.npy", labels)

# # Find metrics using sklearn:
# cfm = confusion_matrix(labels, predictions)
# cfm_norm = confusion_matrix(labels, predictions, normalize="all")
# tn, fp, fn, tp = cfm.ravel()

# # Assert equalitites
# assert TP == tp
# assert TN == tn
# assert FP == fp
# assert FN == fn

# # Plot confusion matrixS_name = r"$\mathtt{S}_1$" if S_val == "S1" else r"$\mathtt{S}_2$"
# disp = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=["Newton", "GR"])
# norm_disp = ConfusionMatrixDisplay(
#     confusion_matrix=cfm_norm, display_labels=["Newton", "GR"]
# )
# norm_disp.plot()
# disp.plot()

# accuracy, precision, recall, F1, TPR, FPR = tutils.calculate_metrics(TP, TN, FP, FN)
# print(f"Accuracy: {accuracy:.3f}")
# print(f"Precision: {precision:.3f}")
# print(f"Recall: {recall:.3f}")
# print(f"F1: {F1:.3f}")
# print(f"TPR: {TPR:.3f}")
# print(f"FPR: {FPR:.3f}")


# # ROC curve
# fpr, tpr, thresholds = roc_curve(labels, predicted_scores)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic")
# plt.legend(loc="lower right")
# plt.show()


def get_saliency_map(model, image, label):
    flag = False
    if image.dim() == 3:
        flag = True
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
    # Do everything on cpu
    model.to("cpu")
    image = image.to("cpu")
    label = label.to("cpu")
    model.eval()
    image.requires_grad_()
    loss, pred = tutils.one_eval(model, loss_fn, image, label)
    pred.backward()
    saliency_map = image.grad.abs().squeeze().detach()
    if flag:
        image = image.squeeze().detach()
        label = label.squeeze()

    return {
        "map": saliency_map,
        "loss": loss.item(),
        "prediction": pred.item(),
        "label": label.item(),
    }


def get_saliency_percentiles(saliency_map, percentile):
    masked_map = saliency_map.copy()
    masked_map[masked_map < np.percentile(masked_map, percentile)] = 0
    return masked_map


# load some saliency data
sal_data = ScaledData(seeds=np.arange(209, 211), train=False, mean=mean, variance=var)


def plot_saliency_map(saliency_map, image, label):
    saliency = saliency_map.numpy()
    saliency95 = get_saliency_percentiles(saliency, 95)
    saliency99 = get_saliency_percentiles(saliency, 99)
    image = image.squeeze().numpy()

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(15, 15),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    ax[0, 0].imshow(saliency, cmap="hot")

    ax[1, 0].imshow(saliency95, cmap="hot")

    ax[1, 1].imshow(saliency99, cmap="hot")

    ax[0, 1].imshow(image, cmap="Spectral", norm=norm)
    # imcbar = fig.colorbar(
    #     ax[0, 1].imshow(image, cmap="Spectral", norm=norm),
    #     ax=ax[0, 1].twinx(),
    #     orientation="vertical",
    # )

    return fig


for idx in [2, 156, 235]:
    image = sal_data[idx]["image"]
    label = sal_data[idx]["label"]
    saliency = get_saliency_map(model, image, label)
    saliency_map = saliency["map"]
    fig = plot_saliency_map(saliency_map, image, label)
    fig.suptitle(
        f"Loss: {saliency['loss']:.3f}, Prediction: {saliency['prediction']:.3f}"
    )
plt.show()
