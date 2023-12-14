""" 
    Script for training a model on a single GPU
"""

# Import necessary packages
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if GPU else "cpu")
assert GPU, "No GPU found."


class SingleGPUTrainer:
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
        device: torch.device = device,
        test_name: str = "test",
    ) -> None:
        """
        Class for training a model on a single GPU.

        Args:
            model (nn.Module): Model to train.
            optimizer (nn.optim): Optimizer to use.
            loss_fn (nn.Loss): Loss function to use.
            device (torch.device, optional): Device to use. Defaults to device.
            writer (SummaryWriter, optional): Tensorboard writer. Defaults to None.
            test_name (str, optional): Test name for saving model and tensorboard. Defaults to "test".
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model_save_path = f"models/{test_name}.pt"
        self.writer_log_path = f"runs/{test_name}"

        # send to device
        self.model = self.model.to(self.device)

        # Tensorboard
        self.writer = SummaryWriter(self.writer_log_path)
        self.writer.add_graph(
            self.model, torch.zeros(model.input_size).unsqueeze(0).to(self.device)
        )

        # Create data loaders
        self.create_data_loaders(
            train_dataset,
            test_dataset,
            batch_size,
            num_workers,
            prefetch_factor,
        )

    def create_data_loaders(
        self,
        train_dataset,
        test_dataset,
        batch_size,
        num_workers,
        prefetch_factor,
    ):
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle=True,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle=True,
            pin_memory=True,
        )

    def _success(self, outputs, labels, tol):
        return (abs(outputs - labels) < tol).sum().item()

    def train(
        self,
        epochs=1,
        breakout_loss=1e-3,
    ):
        # Initialize variables
        best_loss = 1e10

        # Train model
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_predictions, train_samples = self.train_one_epoch(
                epoch, success_tol=self.tol
            )

            # Testing
            test_loss, test_predictions, test_samples = self.evaluate(
                success_tol=self.tol
            )

            # Calculate accuracy
            train_accuracy = train_predictions / train_samples
            test_accuracy = test_predictions / test_samples

            # Save model
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    },
                    self.model_save_path,
                )

            # Save to tensorboard
            self.writer.add_scalars(
                "Loss", {"train": train_loss, "test": test_loss}, epoch
            )
            self.writer.add_scalars(
                "Accuracy",
                {"train": train_accuracy, "test": test_accuracy},
                epoch,
            )

            # Early stopping
            if best_loss < breakout_loss:
                print(
                    f"Breaking out of training loop because test loss {best_loss:.4f} < breakout loss {breakout_loss:.4f}"
                )
                break

    def train_one_epoch(
        self,
        epoch_nr,
        success_tol=1e-2,
    ):
        print(f"---------- Epoch {epoch_nr} ----------")
        self.model.train()
        train_loss = 0
        train_predictions = 0
        train_samples = 0
        max_batches = len(self.train_loader)
        for i, data in enumerate(self.train_loader):
            if (i + 1) % 25 == 0:
                print(f"Batch: {i+1}/{max_batches}")
            # Get the inputs
            images, labels = data["image"], data["label"]
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Print statistics
            train_loss += loss.item()
            train_predictions += self._success(outputs, labels, tol=success_tol)
            train_samples += len(labels)
        print(
            f"Training: Epoch: {epoch_nr}\nTrain loss: {loss:.4f}\nTrain predictions: {train_predictions}/{train_samples}\nTrain accuracy: {train_predictions/train_samples*100:.4f} %\n"
        )
        return train_loss, train_predictions, train_samples

    def evaluate(
        self,
        success_tol=1e-2,
    ):
        self.model.eval()
        test_loss = 0
        test_predictions = 0
        test_samples = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                # Get the inputs
                images, labels = data["image"], data["label"]
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                # Print statistics
                test_loss += loss.item()
                test_predictions += self._success(outputs, labels, tol=success_tol)
                test_samples += len(labels)
        print(
            f"Testing:\nTest loss: {loss:.4f}\nTest predictions: {test_predictions}/{test_samples}\nTest accuracy: {test_predictions/test_samples*100:.4f} %\n"
        )
        return test_loss, test_predictions, test_samples
