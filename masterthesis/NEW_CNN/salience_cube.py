import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import h5py
from torchvision.transforms import Normalize
from tqdm import tqdm, trange

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from scaling_task import scaling
import train_utils as tutils
from architecture import model_o3_err

scaler = scaling.CubeScaler()


# imshow lower
plt.rcParams["image.origin"] = "lower"


""" 
Outline of the script

1. Initialize GPU, load and initialize model.
2. Load data from seed and scale it (load mean and variance first). 
3. Loop through data for each cube, and fill a cube with zeros with the saliency of each pixel. (Main part)
4. Calculate statistics for that specific seed (both Newton and GR), confusion matrix etc.
"""


# 1. Initialize GPU, load and initialize model.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# Model
MODEL_NAME = "T2"
MODEL_PATH = f"models/{MODEL_NAME}.pt"

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
model.eval()  # make sure we are in evaluation mode


# 2. Load data from seed and scale it (load mean and variance first).
datapath = "/mn/stornext/d10/data/johanmkr/simulations/prepared_data/scaled_data_A_s2.215e-09.h5"
(mean, var) = np.load(f"mean_var_A_s2.215e-09.npy")
normalizer = Normalize(mean=[mean], std=[var**0.5])
grlabel = torch.tensor([1.0], dtype=torch.float32)
newtonlabel = torch.tensor([0.0], dtype=torch.float32)


def load_data_cubes(seed):
    with h5py.File(datapath, "r") as f:
        grcube = torch.tensor(f[f"{seed}/gr"][()], dtype=torch.float32)
        newtoncube = torch.tensor(f[f"{seed}/newton"][()], dtype=torch.float32)
    # Normalize cubes
    grcube = normalizer(grcube)
    newtoncube = normalizer(newtoncube)
    return grcube, newtoncube


# 3. Loop through data for each cube, and fill a cube with zeros with the saliency of each pixel. (Main part)
def create_saliency_cubes(seed):
    # Load data
    grcube, newtoncube = load_data_cubes(seed)

    # Initialize saliency cubes
    saliencies = {
        "gr": torch.zeros(grcube.shape),
        "newton": torch.zeros(newtoncube.shape),
    }

    # Containers for statistics
    gr_scores = []
    newton_scores = []
    gr_loss = []
    newton_loss = []

    # Loop through cubes
    for cube, saliency_label, label, loss, scores, description in [
        (
            grcube,
            "gr",
            grlabel,
            gr_loss,
            gr_scores,
            "GR cube",
        ),
        (
            newtoncube,
            "newton",
            newtonlabel,
            newton_loss,
            newton_scores,
            "Newton cube",
        ),
    ]:
        # Send tensors to device
        cube = cube.to(device)
        saliencies[saliency_label] = saliencies[saliency_label].to(device)
        label = label.to(device)
        label = label.unsqueeze(0)  # Add batch dimension
        # Loop through images
        for i in trange(256, desc=f"Seed: {seed:04d} - {description}"):
            # Calculate for image 1
            im1 = (
                cube[i, :, :].unsqueeze(0).unsqueeze(0)
            )  # Add batch and channel dimension
            im1.requires_grad_()
            loss1, score1 = tutils.one_eval(model, loss_fn, im1, label)
            score1.backward()
            saliencies[saliency_label][i, :, :] += im1.grad.squeeze().squeeze().detach()

            # Calculate for image 2
            im2 = (
                cube[:, i, :].unsqueeze(0).unsqueeze(0)
            )  # Add batch and channel dimension
            im2.requires_grad_()
            loss2, score2 = tutils.one_eval(model, loss_fn, im2, label)
            score2.backward()
            saliencies[saliency_label][:, i, :] += im2.grad.squeeze().squeeze().detach()

            # Calculate for image 3
            im3 = (
                cube[:, :, i].unsqueeze(0).unsqueeze(0)
            )  # Add batch and channel dimension
            im3.requires_grad_()
            loss3, score3 = tutils.one_eval(model, loss_fn, im3, label)
            score3.backward()
            saliencies[saliency_label][:, :, i] += im3.grad.squeeze().squeeze().detach()

            # Append scores and losses
            scores.append(score1.item())
            scores.append(score2.item())
            scores.append(score3.item())
            loss.append(loss1.item())
            loss.append(loss2.item())
            loss.append(loss3.item())
            # ---------------------------
    # Save information
    # Check if directory for seed exists
    if not os.path.exists(f"inference_data/saliency_data/{seed:04d}"):
        os.makedirs(f"inference_data/saliency_data/{seed:04d}")
    from IPython import embed

    # embed()
    # Save saliency cubes
    torch.save(
        saliencies["gr"].cpu(),
        f"inference_data/saliency_data/{seed:04d}/gr_saliency.pt",
    )
    torch.save(
        saliencies["newton"].cpu(),
        f"inference_data/saliency_data/{seed:04d}/newton_saliency.pt",
    )

    # Save scores and losses
    # Make dataframe with collective information
    df = pd.DataFrame(
        {
            "gr_scores": gr_scores,
            "newton_scores": newton_scores,
            "gr_loss": gr_loss,
            "newton_loss": newton_loss,
        }
    )
    df.to_pickle(f"inference_data/saliency_data/{seed:04d}/scores_losses.pkl")
    print(f"Seed {seed:04d} done!")

def create_saliency_maps(seed, ax=0, idx=0):
    # Load data
    grcube, newtoncube = load_data_cubes(seed)
    # Load saliency cubes
    gr_saliency = torch.load(f"inference_data/saliency_data/{seed:04d}/gr_saliency.pt")
    newton_saliency = torch.load(
        f"inference_data/saliency_data/{seed:04d}/newton_saliency.pt"
    )
    # Get saliency maps
    gr_saliency = gr_saliency[idx, :, :]
    newton_saliency = newton_saliency[idx, :, :]
    # Get image
    gr_image = grcube[idx, :, :]
    newton_image = newtoncube[idx, :, :]
    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)
    ax[0, 0].imshow(gr_image.cpu(), cmap="hot")
    ax[0, 0].set_title("GR cube")
    ax[0, 1].imshow(gr_saliency.cpu(), cmap="hot")
    ax[0, 1].set_title("GR saliency")
    ax[1, 0].imshow(newton_image.cpu(), cmap="hot")
    ax[1, 0].set_title("Newton cube")
    ax[1, 1].imshow(newton_saliency.cpu(), cmap="hot")
    ax[1, 1].set_title("Newton saliency")
    plt.show(

if __name__ == "__main__":
    for seed in np.arange(250, 300):
        create_saliency_cubes(seed)
    # create_saliency_cubes(250)
