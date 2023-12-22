import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import time
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

GPU = torch.cuda.is_available()
world_size = torch.cuda.device_count()


class MultipleGPUTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        test_dataset,
        batch_size,
        num_workers,
        prefetch_factor,
        test_name: str = "test",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.model_save_path = f"models/{test_name}.pt"
        self.writer_log_path = f"runs/{test_name}"
        self.epochs_trained = 0
        self.m = nn.Sigmoid()

        # Tensorboard
        self.writer = SummaryWriter(self.writer_log_path)
        self.writer.add_graph(self.model, torch.zeros(model.input_size).unsqueeze(0))

    def create_data_loaders(
        self,
        rank,
        world_size,
        train_dataset,
        test_dataset,
        batch_size,
        num_workers=0,
        prefetch_factor=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    ):
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            sampler=train_sampler,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            sampler=test_sampler,
            pin_memory=pin_memory,
        )
        return train_loader, test_loader, train_sampler, test_sampler

    def _setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def _cleanup(self):
        dist.destroy_process_group()

    def _success(self, outputs, labels, tol):
        return (abs(self.m(outputs) - labels) < tol).sum().item()

    def model_init(self, rank, world_size, model):
        # setup the process group
        self._setup(rank, world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(42)

        # create model and move it to GPU with id rank
        model = model.to(rank)
        ddp_model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )

        return ddp_model

    def train_one_epoch(
        self,
        rank,
        model,
        train_loader,
        optimizer,
        loss_fn,
        epoch_nr,
        success_tol=1e-2,
    ):
        print(f"---------- Epoch {epoch_nr} ----------\n")
        model.train()
        train_loss = 0
        train_predictions = 0
        train_samples = 0
        max_batches = len(train_loader)
        for i, data in enumerate(train_loader):
            if (i + 1) % 25 == 0 and rank == 0:
                print(f"Batch: {i+1}/{max_batches}")
            # Get the inputs
            images, labels = data["image"], data["label"]
            images = images.to(rank)
            labels = labels.to(rank)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            train_loss += loss.item()
            train_predictions += self._success(outputs, labels, tol=success_tol)
            train_samples += len(labels)
        train_loss /= max_batches  # avg loss per batch

        return train_loss, train_predictions, train_samples

    def evaluate(
        self,
        model,
        rank,
        loss_fn,
        test_loader,
        success_tol=0.5,
    ):
        model.eval()
        test_loss = 0
        evaluation_predictions = 0
        evaluation_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch["image"], batch["label"]
                images = images.to(rank)
                labels = labels.to(rank)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                # Print statistics
                test_loss += loss.item()
                evaluation_predictions += self._success(
                    outputs, labels, tol=success_tol
                )
                evaluation_samples += len(labels)

        return test_loss, evaluation_predictions, evaluation_samples

    def train(
        self,
        rank,
        world_size,
        epochs,
        breakout_loss,
        tol,
        optimizer_params,
    ):
        # Initialize model
        ddp_model = self.model_init(rank, world_size, self.model)

        (
            train_loader,
            test_loader,
            train_sampler,
            test_sampler,
        ) = self.create_data_loaders(
            rank,
            world_size,
            self.train_dataset,
            self.test_dataset,
            self.batch_size,
            self.num_workers,
            self.prefetch_factor,
        )
        optimizer = torch.optim.Adam(ddp_model.parameters(), **optimizer_params)
        loss_fn = nn.BCEWithLogitsLoss()

        # Train
        best_loss = 1e10
        for _ in range(epochs):
            epoch_total_time_start = time.time() if rank == 0 else None
            current_epoch = self.epochs_trained + 1
            train_sampler.set_epoch(current_epoch)
            epoch_train_start_time = time.time() if rank == 0 else None
            (
                local_train_loss,
                local_train_predictions,
                local_train_samples,
            ) = self.train_one_epoch(
                rank,
                ddp_model,
                train_loader,
                optimizer,
                loss_fn,
                current_epoch,
                success_tol=tol,
            )
            epoch_train_end_time = time.time() if rank == 0 else None

            # Testing
            epoch_test_start_time = time.time() if rank == 0 else None
            local_test_loss, local_test_predictions, local_test_samples = self.evaluate(
                rank,
                ddp_model,
                loss_fn,
                test_loader,
                success_tol=tol,
            )
            epoch_test_end_time = time.time() if rank == 0 else None

            # Gather statistics and send to rank 0
            train_metrics = torch.tensor(
                [local_train_loss, local_train_predictions, local_train_samples],
                dtype=torch.float,
            ).to(rank)
            reduced_train_metrics = [
                torch.zeros_like(train_metrics) for _ in range(world_size)
            ]
            dist.all_gather(reduced_train_metrics, train_metrics)

            test_metrics = torch.tensor(
                [local_test_loss, local_test_predictions, local_test_samples],
                dtype=torch.float,
            ).to(rank)
            reduced_test_metrics = [
                torch.zeros_like(test_metrics) for _ in range(world_size)
            ]
            dist.all_gather(reduced_test_metrics, test_metrics)

            # Placeholder for early stopping across all ranks
            should_stop = torch.tensor(False, dtype=torch.bool).to(rank)

            # Calculate and log statistics on master rank
            if rank == 0:
                # Train data
                total_mean_train_loss, total_train_predictions, total_train_samples = (
                    torch.stack(reduced_train_metrics).sum(dim=0)
                ).tolist()
                mean_train_loss = total_mean_train_loss / world_size
                mean_train_accuracy = total_train_predictions / total_train_samples

                # Test data
                total_mean_test_loss, total_test_predictions, total_test_samples = (
                    torch.stack(reduced_test_metrics).sum(dim=0)
                ).tolist()
                mean_test_loss = total_mean_test_loss / world_size
                mean_test_accuracy = total_test_predictions / total_test_samples

                # Print statistics
                print(
                    f"\nTraining:\nTrain loss: {mean_train_loss:.4f}\nTrain predictions: {total_train_predictions}/{total_train_samples}\nTrain accuracy: {mean_train_accuracy*100:.4f} %\nTime elapsed for training: {epoch_train_end_time - epoch_train_start_time:.2f} s\n"
                )

                print(
                    f"Testing:\nTest loss: {mean_test_loss:.4f}\nTest predictions: {total_test_predictions}/{total_test_samples}\nTest accuracy: {mean_test_accuracy*100:.4f} %\nTime elapsed for testing: {epoch_test_end_time - epoch_test_start_time:.2f} s\n"
                )

                # Write to tensorboard
                self.writer.add_scalars(
                    "Loss",
                    {"train": mean_train_loss, "test": mean_test_loss},
                    current_epoch,
                )
                self.writer.add_scalars(
                    "Accuracy",
                    {"train": mean_train_accuracy, "test": mean_test_accuracy},
                    current_epoch,
                )
                self.writer.flush()

                self.epochs_trained += 1  # Only rank 0 can update this
                epoch_total_time_end = time.time()

                print(
                    f"Time elapsed for epoch: {epoch_total_time_end - epoch_total_time_start:.2f} s\n"
                )

                # Early stopping condition
                if mean_test_loss < breakout_loss:
                    print(
                        f"Breaking out of training loop because test loss {mean_test_loss:.4f} < breakout loss {breakout_loss:.4f}"
                    )
                    should_stop = torch.tensor(True, dtype=torch.bool).to(rank)

            # Broadcast early stopping condition to all ranks
            dist.broadcast(should_stop, src=0)

            # Save model if mean loss is better than best loss of if early stopping condition is met
            if rank == 0:
                if should_stop or mean_test_loss < best_loss:
                    best_loss = mean_test_loss
                    torch.save(
                        {
                            "epoch": current_epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": mean_train_loss,
                            "test_loss": mean_test_loss,
                        },
                        self.model_save_path,
                    )
                    print(
                        f"New best loss: {best_loss:.4f}. Saved model to {self.model_save_path}"
                    )

            # Break out of training loop if early stopping condition is met
            if should_stop:
                print(f"Rank {rank} breaking out of training loop")
                break

            # Synchronize all processes
            dist.barrier()

        print(f"Trained for {self.epochs_trained} epochs") if rank == 0 else None
        print(f"Rank {rank} finished training")

        # Closing writer and cleaning up
        if rank == 0:
            self.writer.flush()
            self.writer.close()

        self._cleanup()

    def run(
        self,
        epochs,
        breakout_loss,
        tol,
        optimizer_params,
    ):
        mp.spawn(
            self.train,
            args=(world_size, epochs, breakout_loss, tol, optimizer_params),
            nprocs=world_size,
            join=True,
        )
