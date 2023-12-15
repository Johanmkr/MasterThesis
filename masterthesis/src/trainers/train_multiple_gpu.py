import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

GPU = torch.cuda.is_available()
world_size = torch.cuda.device_count()


class MultipleGPUTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: nn.optim,
        loss_fn: nn.Loss,
        train_dataset,
        test_dataset,
        batch_size,
        num_workers,
        prefetch_factor,
        test_name: str = "test",
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.model_save_path = f"models/{test_name}.pt"
        self.writer_log_path = f"runs/{test_name}"

        # Tensorboard
        self.writer = SummaryWriter(self.writer_log_path)
        self.writer.add_graph(self.model, torch.zeros(model.input_size).unsqueeze(0))

    def create_data_loaders(
        self,
        train_dataset,
        test_dataset,
        train_sampler,
        test_sampler,
        batch_size,
        num_workers,
        prefetch_factor,
    ):
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            sampler=train_sampler,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            sampler=test_sampler,
            pin_memory=True,
        )
        return train_loader, test_loader

    def _setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def _cleanup(self):
        dist.destroy_process_group()

    def _success(self, outputs, labels, tol):
        return (abs(outputs - labels) < tol).sum().item()

    def trainer_function(
        self,
        rank,
        world_size,
    ):
        pass
