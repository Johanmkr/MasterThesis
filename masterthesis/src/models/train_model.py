###TODO CHECK THIS


def train_model(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs,
    device,
    save_path,
    verbose=True,
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
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for i, data in enumerate(train_loader):
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
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()

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
                val_acc += (outputs.argmax(1) == labels).sum().item()

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
                f"Epoch: {epoch}, " f"Train loss: {train_loss:.4f}",
                f"Val loss: {val_loss:.4f}",
                f"Train acc: {train_acc:.4f}",
                f"Val acc: {val_acc:.4f}",
            )
