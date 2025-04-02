"""
MNIST Classification with CNN - Module 3
----------------------------------------
Implementing a Convolutional Neural Network for classifying handwritten digits.
Demonstrates the power of convolutional operations for image recognition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST digit classification
    Architecture:
    - Conv Layer 1: 1 input channel -> 32 output channels, 3x3 kernel
    - Conv Layer 2: 32 input channels -> 64 output channels, 3x3 kernel
    - Fully Connected Layer 1: 1600 -> 128 units
    - Fully Connected Layer 2: 128 -> 10 units (for 10 digits)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )

        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(
            64 * 7 * 7, 128
        )  # After two 2x2 pooling operations: 28x28 -> 14x14 -> 7x7
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # First conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv1(x)))

        # Second conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)

        # First fully connected layer with dropout
        x = self.dropout(F.relu(self.fc1(x)))

        # Output layer
        x = self.fc2(x)

        return x


def load_data(batch_size=64):
    """
    Load the MNIST dataset using PyTorch's DataLoader

    Args:
        batch_size: Number of images per batch

    Returns:
        train_loader, test_loader: DataLoader objects for training and testing
    """
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    # Download and load training data
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Download and load test data
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train the model for one epoch

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimization algorithm
        criterion: Loss function
        device: Device to run the training on (CPU/GPU)
        epoch: Current epoch number

    Returns:
        average_loss: Average loss for this epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)

        # Calculate loss
        loss = criterion(outputs, target)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Print progress
        if batch_idx % 100 == 0:
            print(
                f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.2f}%"
            )

    # Calculate average loss and accuracy
    average_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    print(
        f"Epoch {epoch} completed | Avg Loss: {average_loss:.4f} | Acc: {accuracy:.2f}%"
    )

    return average_loss, accuracy


def test(model, test_loader, criterion, device):
    """
    Test the model on the test dataset

    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run the testing on (CPU/GPU)

    Returns:
        test_loss, accuracy: Average loss and accuracy on test data
    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0

    # No gradient computation needed for testing
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data)

            # Calculate loss
            loss = criterion(outputs, target)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    # Calculate average loss and accuracy
    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total

    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return test_loss, accuracy


def visualize_results(train_losses, train_accs, test_losses, test_accs):
    """
    Visualize training and testing results

    Args:
        train_losses, test_losses: Lists of losses
        train_accs, test_accs: Lists of accuracies
    """
    epochs = range(1, len(train_losses) + 1)

    # Create figure with two subplots
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, test_losses, "r-", label="Test Loss")
    plt.title("Training and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, "b-", label="Training Accuracy")
    plt.plot(epochs, test_accs, "r-", label="Test Accuracy")
    plt.title("Training and Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, test_loader, device, num_images=10):
    """
    Visualize some example predictions

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run the model on
        num_images: Number of images to visualize
    """
    model.eval()

    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Move to device and make predictions
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Convert images back to CPU for visualization
    images = images.cpu().numpy()

    # Create a grid to display images
    fig = plt.figure(figsize=(15, 3))

    # Display up to num_images
    num_to_display = min(num_images, images.shape[0])

    for i in range(num_to_display):
        ax = fig.add_subplot(1, num_to_display, i + 1)

        # Display image (need to reshape and denormalize)
        img = np.squeeze(images[i])
        img = img * 0.3081 + 0.1307  # Denormalize
        ax.imshow(img, cmap="gray")

        # Add title with prediction and true label
        pred_label = predicted[i].item()
        true_label = labels[i].item()

        title_color = "green" if pred_label == true_label else "red"
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=title_color)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def visualize_filters(model):
    """
    Visualize filters from the first convolutional layer

    Args:
        model: Trained CNN model
    """
    # Get weights from the first conv layer
    # Shape is [out_channels, in_channels, height, width]
    weights = model.conv1.weight.data.cpu().numpy()

    # Number of filters to display
    num_filters = min(32, weights.shape[0])

    # Create a grid to display filters
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("First Convolutional Layer Filters", fontsize=16)

    for i in range(num_filters):
        ax = fig.add_subplot(4, 8, i + 1)

        # Each filter has shape [in_channels, height, width]
        # For grayscale input, in_channels=1, so we can squeeze it
        filt = np.squeeze(weights[i])

        # Normalize filter for better visualization
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-10)

        ax.imshow(filt, cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Filter {i+1}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_data(batch_size=64)

    # Create model
    print("Creating CNN model...")
    model = SimpleCNN().to(device)
    print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Starting training...")
    num_epochs = 5
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    # Track time
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, device, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate on test set
        test_loss, test_acc = test(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # Print total training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

    # Visualize results
    print("Visualizing training results...")
    visualize_results(train_losses, train_accs, test_losses, test_accs)

    # Visualize example predictions
    print("Visualizing example predictions...")
    visualize_predictions(model, test_loader, device)

    # Visualize filters
    print("Visualizing convolutional filters...")
    visualize_filters(model)

    # Haiku moment
    print("\nHaiku:")
    print("Pixels transformed by")
    print("Convolutional vision")
    print("Digits recognized")


if __name__ == "__main__":
    main()
