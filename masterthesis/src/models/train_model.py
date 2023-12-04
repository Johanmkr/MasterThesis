import numpy as np
import torch
import torch.nn as nn
from IPython import embed


class ModelTrainer:
    def __init__(
        self, model, optimizer, loss_fn, device, verbose=True, writer=None, tol=1e-2
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.verbose = verbose
        self.tol = tol

        # Data storage
        self.epoch_array = []
        self.train_loss_array = []
        self.train_acc_array = []
        self.test_loss_array = []
        self.test_acc_array = []

        if writer is not None:
            self.writer = writer
            self.writer.add_graph(self.model, torch.zeros(model.input_size))

    def train_model(
        self,
        train_loader,
        test_loader,
        epochs,
    ):
        """
        Train the model.
        """
        # Initialize variables
        best_loss = 1e10

        # Move model to device
        self.model = self.model.to(self.device)

    def train_one_epoch(
        self, train_loader: list | np.ndarray, epoch: int, verbose: bool = True
    ) -> tuple[float, float]:
        print(f"--- Epoch {epoch} ---") if verbose else None
        self.model.train()  # Set model to training mode
        train_loss = 0.0
        train_accuracy = 0.0
        max_batches = len(train_loader)
        for i, data in enumerate(train_loader):
            # Get the inputs
            print(
                f"Training: Epoch: {epoch}, batch {i+1}/{max_batches}"
            ) if verbose else None
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
            train_accuracy += (abs(outputs - labels) < self.tol).sum().item()
        print(
            f"Predicted (train set): [{train_accuracy}/{len(train_loader.dataset)}]"
        ) if verbose else None
        return train_loss, train_accuracy

    def test_model(self, loader: list | np.ndarray) -> tuple[float, float]:
        self.model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get the inputs
                images, labels = data["image"], data["label"]
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                # Print statistics
                test_loss += loss.item()
                test_accuracy += (abs(outputs - labels) < self.tol).sum().item()
        print(f"Predicted (test set): [{test_accuracy}/{len(loader.dataset)}]")
        return test_loss, test_accuracy

    def log_stuff(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        test_loss: float,
        test_accuracy: float,
    ):
        assert self.writer is not None, "Writer is None"
        self.epoch_array.append(epoch)
        self.train_loss_array.append(train_loss)
        self.train_acc_array.append(train_accuracy)
        self.test_loss_array.append(test_loss)
        self.test_acc_array.append(test_accuracy)

        # Add loss and accuracy to tensorboard
        self.writer.add_scalar("Loss", {"train": train_loss, "test": test_loss}, epoch)
        self.writer.add_scalar(
            "Accuracy", {"train": train_accuracy, "test": test_accuracy}, epoch
        )


def overfit_model(
    model,
    optimizer,
    loss_fn,
    train_loader,
    device,
    verbose=True,
    tol: float = 1e-1,
):
    """
    Overfit the model.
    """

    # Move model to device
    model = model.to(device)

    # Loop over epochs
    epoch = 0
    train_loss = 1
    data = next(iter(train_loader))
    max_epoch = 1000
    train_acc = 0.0
    epoch_array = []
    loss_array = []
    acc_array = []
    # for _ in range(10):
    while train_acc < 0.95:
        # Training
        print(f"--- Epoch {epoch+1} ---")
        model.train()
        # Get the inputs
        images, labels = data["image"], data["label"]
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        train_loss = loss.item()
        predictions = (abs(outputs - labels) < tol).sum().item()

        # Print statistics
        train_loss /= len(images)
        train_acc = predictions / len(images)
        epoch += 1

        # Append to arrays
        epoch_array.append(epoch)
        loss_array.append(train_loss)
        acc_array.append(train_acc)

        print(
            f"Epoch: {epoch}: "
            f"Train loss: {train_loss:.5f}, Train acc: {train_acc:.4f}, predicted: {predictions}/{len(images)}\n"
        )
        if epoch >= max_epoch:
            break
    return epoch_array, loss_array, acc_array
