###TODO CHECK THIS
import torch
import torch.nn as nn
from IPython import embed


def train_model(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs,
    device,
    verbose=True,
    tol: float = 1e-2,
):
    """
    Train the model.
    """
    # Initialize variables
    best_loss = 1e10
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Move model to device
    model = model.to(device)

    # Loop over epochs
    for epoch in range(1, epochs + 1):
        # Training
        print(f"--- Epoch {epoch} ---")
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        max_batches = len(train_loader)
        for i, data in enumerate(train_loader):
            # Get the inputs
            print(f"Training: Epoch: {epoch}, batch {i+1}/{max_batches}")
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
            train_loss += loss.item()
            train_acc += (abs(outputs - labels) < tol).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                # Get the inputs
                images, labels = data["image"], data["label"]
                images = images.to(device)
                labels = labels.to(device)

                # Forward
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                # Print statistics
                val_loss += loss.item()
                val_acc += (abs(outputs - labels) < tol).sum().item()
                # embed()
        print(f"Predicted {val_acc} out of {len(val_loader.dataset)}")
        # Print statistics
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_acc /= len(train_loader.dataset)
        val_acc /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if verbose:
            print(
                f"Epoch: {epoch}: " f"Train loss: {train_loss:.4f}",
                f"Val loss: {val_loss:.4f}",
                f"Train acc: {train_acc:.4f}",
                f"Val acc: {val_acc:.4f}\n",
            )

        # Save model if validation loss has decreased
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "model.pt")


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
    max_epoch = 500
    train_acc = 0.0
    # for _ in range(10):
    while train_acc < 0.9:
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
        print(
            f"Epoch: {epoch}: "
            f"Train loss: {train_loss:.5f}, Train acc: {train_acc:.4f}, predicted: {predictions}/{len(images)}\n"
        )
        if epoch >= max_epoch:
            break
