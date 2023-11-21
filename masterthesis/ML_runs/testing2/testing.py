""" 
Script used for testing and improving the performance of the data loading pipeline.

"""


import numpy as np
import sys, os
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import time
import inspect
import h5py

# Add the parent directory of the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

from IPython import embed

# Local imports
from src.data.custom_dataset import CustomDataset, CustomDatasetFAST, make_dataset
from src.models.MOTH import MOTH  # Dummy model of convolutional network
import config as cfg
from src.utils import paths


def test_loader():
    train_loader, test_loader, val_loader = make_dataset(**cfg.DATA_PARAMS)

    total_batches = len(train_loader)
    total_time = 0.0
    set_time = time.time()
    for i, batch in enumerate(train_loader):
        # start_time = time.time()

        images, labels = batch["image"], batch["label"]

        # some stuff happens here
        time.sleep(0.5)

        end_time = time.time()
        batch_time = end_time - set_time
        total_time += batch_time
        set_time = end_time

        print(f"Batch {i+1}/{total_batches} took {batch_time:.4f} seconds.")
    print(f"Cubes loaded: {len(train_loader.dataset)}")
    average_time = total_time / total_batches
    print(f"Average time per batch: {average_time:.4f} seconds.")
    print(f"Total time: {total_time:.4f} seconds.")


def test_load_of_cubes(nr_cubes: int = 50):
    accepted_params = inspect.signature(CustomDataset.__init__).parameters.keys()
    mod_dict = {
        key: value for key, value in cfg.DATA_PARAMS.items() if key in accepted_params
    }
    # print(accepted_params)
    # print(cfg.DATA_PARAMS.keys())
    # print(mod_dict)
    seeds = np.arange(0, 2000, 1)
    init_time_start = time.time()
    dataset = CustomDataset(seeds=seeds, **mod_dict)
    init_time_end = time.time()
    print(f"Time to initialize dataset: {init_time_end-init_time_start:.4f} seconds.")

    zero_time = time.time()
    total_time = 0.0
    for i in np.random.randint(0, len(dataset), nr_cubes):
        # Load one cube
        # print(i)
        cube = dataset[i]
        # dataset.print_image(i)

        now_time = time.time()
        cube_time = now_time - zero_time
        total_time += cube_time
        zero_time = now_time

        print(f"Time to load cube: {cube_time:.4f} seconds.\n")
    print(f"Average time per cube: {total_time/nr_cubes:.4f} seconds.")
    print(f"Total time for {nr_cubes} cubes: {total_time:.4f} seconds.")


def test_time_to_load_npy_file_instead_of_cube(seed=0, z=1.0, N=1000):
    avg_h5_time = 0.0
    avg_npy_time = 0.0
    for _ in trange(N):
        GR_path = str(paths.get_cube_path(seed, "GR", z))
        Newton_path = str(paths.get_cube_path(seed, "Newton", z))
        # Load hd5-file
        load_h5_start = time.time()
        # Gr
        GR_h5File = h5py.File(GR_path, "r")
        GR_h5Data = GR_h5File["data"]
        GR_data = GR_h5Data[()]
        GR_h5File.close()
        # Newton
        Newton_h5File = h5py.File(Newton_path, "r")
        Newton_h5Data = Newton_h5File["data"]
        Newton_data = Newton_h5Data[()]
        Newton_h5File.close()

        load_h5_end = time.time()
        h5time = load_h5_end - load_h5_start
        avg_h5_time += h5time

        GR_npypath = GR_path.replace(".h5", ".npy")
        if not os.path.exists(GR_npypath):
            print(f"Saving to npy-file... for seed: {seed}, gravity: GR, redshift: {z}")
            np.save(GR_npypath, GR_data)
        Newton_npypath = Newton_path.replace(".h5", ".npy")
        if not os.path.exists(Newton_npypath):
            print(
                f"Saving to npy-file... for seed: {seed}, gravity: Newton, redshift: {z}"
            )
            np.save(Newton_npypath, Newton_data)

        load_npy_start = time.time()
        GR_newdata = np.load(GR_npypath)
        Newton_newdata = np.load(Newton_npypath)
        load_npy_end = time.time()
        npy_time = load_npy_end - load_npy_start
        avg_npy_time += npy_time

    avg_h5_time /= 2 * N
    avg_npy_time /= 2 * N
    print(f"Time to load h5-file: {avg_h5_time:.4f} seconds.")
    print(f"Time to load npy-file: {avg_npy_time:.4f} seconds.")


if __name__ == "__main__":
    test_loader()
    # test_load_of_cubes()
    # seeds = np.arange(0, 2000, 25)
    # for s in seeds:
    #     test_time_to_load_npy_file_instead_of_cube(seed=s, z=1.0, N=1)
