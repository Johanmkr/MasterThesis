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
mp.set_sharing_strategy("file_system")

GPU = torch.cuda.is_available()
world_size = torch.cuda.device_count()

output_func = nn.Sigmoid()


# DDP functionality
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def model_init(rank, world_size, model):
    # setup the process group
    setup(rank, world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)

    # create model and move it to GPU with id rank
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    return ddp_model


def cleanup():
    dist.destroy_process_group()


def success(outputs, labels, tol):
    return (abs(output_func(outputs) - labels) < tol).sum().item()


def train_one_epoch(
    rank,
    model,
    train_loader,
    optimizer,
    loss_fn,
    epoch_nr,
    success_tol=1e-2,
):
    print(f"---------- Epoch {epoch_nr} ----------\n") if rank == 0 else None
    model.train()
    train_loss = 0
    train_predictions = 0
    train_samples = 0
    max_batches = len(train_loader)

    # for i, data in enumerate(train_loader):
    #     if ((i + 1) % 25 == 0 or (i + 1) == max_batches) and rank == 0:
    #         print(f"TRAIN - Batch: {i+1}/{max_batches}")
    #     # Get the inputs
    #     images, labels = data["image"], data["label"]
    #     images = images.to(rank, non_blocking=False)
    #     labels = labels.to(rank, non_blocking=False)

    #     # Zero the parameter gradients
    #     optimizer.zero_grad()

    #     # Forward + backward + optimize
    #     outputs = model(images)
    #     loss = loss_fn(outputs, labels)
    #     loss.backward()
    #     optimizer.step()

    #     # Print statistics
    #     train_loss += loss.item()
    #     train_predictions += success(outputs, labels, tol=success_tol)
    #     train_samples += len(labels)

    i = 0
    iterator = iter(train_loader)
    end_of_data = False
    while not end_of_data:
        try:
            batch = next(iterator)

            # if not batch:
            #     end_of_data = True
            #     break

            if ((i + 1) % 25 == 0 or (i + 1) == max_batches) and rank == 0:
                print(f"TRAIN - Batch: {i+1}/{max_batches}")
            # Get the inputs
            images, labels = batch["image"], batch["label"]
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            train_loss += loss.item()
            train_predictions += success(outputs, labels, tol=success_tol)
            train_samples += len(labels)
            i += 1
        except StopIteration:
            end_of_data = True

    train_loss /= max_batches  # avg loss per batch

    return train_loss, train_predictions, train_samples


def evaluate(
    rank,
    model,
    loss_fn,
    test_loader,
    success_tol=1e-2,
):
    model.eval()
    test_loss = 0
    evaluation_predictions = 0
    evaluation_samples = 0
    max_batches = len(test_loader)

    # with torch.no_grad():
    #     for i, batch in enumerate(test_loader):
    #         if ((i + 1) % 25 == 0 or (i + 1) == max_batches) and rank == 0:
    #             print(f"EVAL - Batch: {i+1}/{max_batches}")
    #         images, labels = batch["image"], batch["label"]
    #         images = images.to(rank, non_blocking=False)
    #         labels = labels.to(rank, non_blocking=False)

    #         outputs = model(images)
    #         loss = loss_fn(outputs, labels)

    #         # Print statistics
    #         test_loss += loss.item()
    #         evaluation_predictions += success(outputs, labels, tol=success_tol)
    #         evaluation_samples += len(labels)
    #     test_loss /= max_batches  # avg loss per batch

    i = 0
    iterator = iter(test_loader)
    end_of_data = False

    with torch.no_grad():
        while not end_of_data:
            try:
                batch = next(iterator)
                if ((i + 1) % 25 == 0 or (i + 1) == max_batches) and rank == 0:
                    print(f"EVAL - Batch: {i+1}/{max_batches}")
                images, labels = batch["image"], batch["label"]
                images = images.to(rank, non_blocking=True)
                labels = labels.to(rank, non_blocking=True)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                # Print statistics
                test_loss += loss.item()
                evaluation_predictions += success(outputs, labels, tol=success_tol)
                evaluation_samples += len(labels)
            except StopIteration:
                end_of_data = True
    test_loss /= max_batches  # avg loss per batch

    return test_loss, evaluation_predictions, evaluation_samples


def create_data_loaders(
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


def worker(
    rank,
    world_size,
    model,
    train_dataset,
    test_dataset,
    batch_size,
    num_workers,
    prefetch_factor,
    pin_memory,
    shuffle,
    drop_last,
    optimizer_params,
    epochs,
    tol,
    writer_log_path,
    breakout_loss,
    model_save_path,
):
    # Initialize model
    ddp_model = model_init(rank, world_size, model)

    # Initialise dataloaders
    train_loader, test_loader, train_sampler, test_sampler = create_data_loaders(
        rank,
        world_size,
        train_dataset,
        test_dataset,
        batch_size,
        num_workers,
        prefetch_factor,
        pin_memory,
        shuffle,
        drop_last,
    )

    # Initialise optimizer
    optimizer = torch.optim.Adam(ddp_model.parameters(), **optimizer_params)
    loss_fn = nn.BCEWithLogitsLoss()

    # Tensorboard
    if rank == 0:
        writer = SummaryWriter(writer_log_path)
        # writer.add_graph(ddp_model, torch.zeros((1, 256, 256)).unsqueeze(0))

    # Train and test
    best_loss = 1e10
    for epoch in range(1, epochs + 1):
        epoch_total_time_start = time.time() if rank == 0 else None

        # ---- TRAINING ----
        train_sampler.set_epoch(epoch)
        epoch_train_start_time = time.time() if rank == 0 else None
        (
            local_train_loss,
            local_train_predictions,
            local_train_samples,
        ) = train_one_epoch(
            rank,
            ddp_model,
            train_loader,
            optimizer,
            loss_fn,
            epoch,
            success_tol=tol,
        )
        epoch_train_end_time = time.time() if rank == 0 else None

        # Gather statistics and send to rank 0
        train_metrics = torch.tensor(
            [local_train_loss, local_train_predictions, local_train_samples],
            dtype=torch.float,
        ).to(rank)
        reduced_train_metrics = [
            torch.zeros_like(train_metrics) for _ in range(world_size)
        ]
        dist.all_gather(reduced_train_metrics, train_metrics)

        # Calculate and log statistics on master rank
        if rank == 0:
            # Train data
            total_mean_train_loss, total_train_predictions, total_train_samples = (
                torch.stack(reduced_train_metrics).sum(dim=0)
            ).tolist()
            mean_train_loss = total_mean_train_loss / world_size
            mean_train_accuracy = total_train_predictions / total_train_samples

            # Print statistics
            print(
                f"\nTraining:\nTrain loss: {mean_train_loss:.4f}\nTrain predictions: {total_train_predictions}/{total_train_samples}\nTrain accuracy: {mean_train_accuracy*100:.4f} %\nTime elapsed for training: {epoch_train_end_time - epoch_train_start_time:.2f} s\n"
            )
            # Write to tensorboard
            writer.add_scalar("Loss/train", mean_train_loss, epoch)
            writer.add_scalar("Accuracy/train", mean_train_accuracy, epoch)
            writer.add_scalar("Correct/train", total_train_predictions, epoch)
            writer.flush()

        # Placeholder for early stopping across all ranks
        should_stop = torch.tensor(False, dtype=torch.bool).to(rank)

        if epoch % 10 == 0:
            # ---- TESTING ----
            epoch_test_start_time = time.time() if rank == 0 else None
            local_test_loss, local_test_predictions, local_test_samples = evaluate(
                rank,
                ddp_model,
                loss_fn,
                test_loader,
                success_tol=tol,
            )
            epoch_test_end_time = time.time() if rank == 0 else None

            # Gather statistics and send to rank 0
            test_metrics = torch.tensor(
                [local_test_loss, local_test_predictions, local_test_samples],
                dtype=torch.float,
            ).to(rank)
            reduced_test_metrics = [
                torch.zeros_like(test_metrics) for _ in range(world_size)
            ]
            dist.all_gather(reduced_test_metrics, test_metrics)

            # Calculate and log statistics on master rank
            if rank == 0:
                # Test data
                total_mean_test_loss, total_test_predictions, total_test_samples = (
                    torch.stack(reduced_test_metrics).sum(dim=0)
                ).tolist()
                mean_test_loss = total_mean_test_loss / world_size
                mean_test_accuracy = total_test_predictions / total_test_samples

                # Print statistics
                print(
                    f"Testing:\nTest loss: {mean_test_loss:.4f}\nTest predictions: {total_test_predictions}/{total_test_samples}\nTest accuracy: {mean_test_accuracy*100:.4f} %\nTime elapsed for testing: {epoch_test_end_time - epoch_test_start_time:.2f} s\n"
                )

                # Write to tensorboard
                writer.add_scalar("Loss/test", mean_test_loss, epoch)
                writer.add_scalar("Accuracy/test", mean_test_accuracy, epoch)
                writer.add_scalar("Correct/test", total_test_predictions, epoch)
                writer.flush()

                # Early stopping condition
                if mean_test_loss < breakout_loss:
                    print(
                        f"Breaking out of training loop because test loss {mean_test_loss:.4f} < breakout loss {breakout_loss:.4f}"
                    )
                    should_stop = torch.tensor(True, dtype=torch.bool).to(rank)

                # Save model if mean loss is better than best loss of if early stopping condition is met
                if should_stop or mean_test_loss < best_loss:
                    best_loss = mean_test_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": mean_train_loss,
                            "test_loss": mean_test_loss,
                        },
                        model_save_path,
                    )
                    print(
                        f"New best loss: {best_loss:.4f}. Saved model to {model_save_path}"
                    )
        if rank == 0:
            epoch_total_time_end = time.time()

            print(
                f"Time elapsed for epoch: {epoch_total_time_end - epoch_total_time_start:.2f} s\n"
            )
        # Broadcast early stopping condition to all ranks
        dist.broadcast(should_stop, src=0)

        # Break out of training loop if early stopping condition is met
        if should_stop:
            print(f"Rank {rank} breaking out of training loop")
            break

        # Synchronize all processes
        dist.barrier()

    print(f"Trained for {epoch} epochs") if rank == 0 else None
    print(f"Rank {rank} finished training")

    # Closing writer and cleaning up
    if rank == 0:
        writer.flush()
        writer.close()

    cleanup()


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

        # Tensorboard
        # self.writer = SummaryWriter(self.writer_log_path)
        # self.writer.add_graph(self.model, torch.zeros(model.input_size).unsqueeze(0))

    def run(
        self,
        epochs,
        breakout_loss,
        tol,
        optimizer_params,
    ):
        worker_args = {
            "world_size": world_size,
            "model": self.model,
            "train_dataset": self.train_dataset,
            "test_dataset": self.test_dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": True,
            "shuffle": True,
            "drop_last": True,
            "optimizer_params": optimizer_params,
            "epochs": epochs,
            "tol": tol,
            "writer_log_path": self.writer_log_path,
            "breakout_loss": breakout_loss,
            "model_save_path": self.model_save_path,
        }

        self.processes = []
        # mp.set_start_method("spawn", force=True)

        # for rank in range(world_size):
        #     p = mp.Process(target=worker, args=(rank,), kwargs=worker_args)
        #     p.start()
        #     self.processes.append(p)
        # # p = mp.Process(target=worker, kwargs=worker_args)
        # # p.start()
        # # p.join()

        # for p in self.processes:
        #     p.join()

        mp.spawn(
            worker,
            args=(tuple(worker_args.values())),
            # args=(world_size, self.train_dataset, self.test_dataset, self.model),
            nprocs=world_size,
            join=True,
        )
        self.epochs_trained += epochs


if __name__ == "__main__":
    pass
