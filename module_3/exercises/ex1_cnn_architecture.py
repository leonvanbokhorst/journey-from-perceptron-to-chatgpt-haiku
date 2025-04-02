"""
Exercise 3.1: Experimenting with CNN Architecture

In this exercise, you will explore how different CNN architectures affect performance on
the MNIST dataset. You'll experiment with different numbers of layers, filters, kernel sizes,
and other hyperparameters to see how they impact accuracy and training time.

Learning Objectives:
1. Gain hands-on experience with CNN architectures
2. Understand the trade-offs between model complexity and performance
3. Analyze how different architectural choices affect accuracy and training time
4. Learn to implement and modify CNN architectures in PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
from torch.utils.data import DataLoader, random_split

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define a baseline CNN
class SimpleCNN(nn.Module):
    def __init__(self, conv_layers=2, filters=[8, 16], kernel_sizes=[3, 3]):
        super(SimpleCNN, self).__init__()

        # Input validation
        assert (
            len(filters) == conv_layers
        ), "Number of filters must match number of conv layers"
        assert (
            len(kernel_sizes) == conv_layers
        ), "Number of kernel sizes must match number of conv layers"

        self.conv_layers = nn.ModuleList()
        in_channels = 1  # MNIST images are grayscale (1 channel)

        # Add convolutional layers
        for i in range(conv_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, filters[i], kernel_size=kernel_sizes[i], padding=1
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(2),  # Reduce spatial dimensions by half
                )
            )
            in_channels = filters[i]

        # Calculate the size of the flattened feature maps after all conv layers
        # For MNIST (28x28), each MaxPool2d divides dimensions by 2
        feature_size = 28
        for _ in range(conv_layers):
            feature_size = feature_size // 2

        self.fc = nn.Linear(filters[-1] * feature_size * feature_size, 10)

    def forward(self, x):
        # Pass through all convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Final classification layer
        x = self.fc(x)
        return x


def load_mnist_data(batch_size=64, val_split=0.1):
    """Load MNIST data with a validation split"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    # Load training data
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Split into training and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Load test data
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on the given data loader"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Train the model and return training history"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "time_per_epoch": [],
    }

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Record time
        epoch_time = time.time() - start_time

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["time_per_epoch"].append(epoch_time)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    return history


def plot_history(history, title="Training History"):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["val_acc"], label="Validation Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    # Plot time per epoch
    plt.figure(figsize=(10, 4))
    plt.plot(history["time_per_epoch"])
    plt.title("Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def experiment_with_architectures():
    """Run experiments with different CNN architectures"""
    # Load data
    train_loader, val_loader, test_loader = load_mnist_data()

    # Define architectures to test
    architectures = [
        {
            "name": "Single Conv Layer (8 filters)",
            "conv_layers": 1,
            "filters": [8],
            "kernel_sizes": [3],
        },
        {
            "name": "Two Conv Layers (8-16 filters)",
            "conv_layers": 2,
            "filters": [8, 16],
            "kernel_sizes": [3, 3],
        },
        {
            "name": "Three Conv Layers (8-16-32 filters)",
            "conv_layers": 3,
            "filters": [8, 16, 32],
            "kernel_sizes": [3, 3, 3],
        },
        {
            "name": "Two Conv Layers with More Filters (16-32)",
            "conv_layers": 2,
            "filters": [16, 32],
            "kernel_sizes": [3, 3],
        },
        {
            "name": "Two Conv Layers with Different Kernels (3-5)",
            "conv_layers": 2,
            "filters": [8, 16],
            "kernel_sizes": [3, 5],
        },
    ]

    results = {}

    for arch in architectures:
        print(f"\nTraining architecture: {arch['name']}")

        # Create model with specified architecture
        model = SimpleCNN(
            conv_layers=arch["conv_layers"],
            filters=arch["filters"],
            kernel_sizes=arch["kernel_sizes"],
        )

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameter count: {param_count:,}")

        # Train model
        history = train_model(model, train_loader, val_loader, epochs=5)

        # Test final model
        test_loss, test_acc = evaluate(
            model, test_loader, nn.CrossEntropyLoss(), device
        )
        print(f"Test accuracy: {test_acc:.4f}")

        # Store results
        results[arch["name"]] = {
            "param_count": param_count,
            "history": history,
            "test_acc": test_acc,
            "avg_epoch_time": np.mean(history["time_per_epoch"]),
        }

        # Plot this architecture's history
        plot_history(history, title=f"Training History: {arch['name']}")

    # Compare architectures
    compare_architectures(results)


def compare_architectures(results):
    """Compare different architectures based on results"""
    names = list(results.keys())
    test_accs = [results[name]["test_acc"] for name in names]
    param_counts = [results[name]["param_count"] for name in names]
    avg_times = [results[name]["avg_epoch_time"] for name in names]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot test accuracy vs parameter count
    ax1.scatter(param_counts, test_accs)
    for i, name in enumerate(names):
        ax1.annotate(
            name,
            (param_counts[i], test_accs[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    ax1.set_title("Test Accuracy vs. Parameter Count")
    ax1.set_xlabel("Number of Parameters")
    ax1.set_ylabel("Test Accuracy")
    ax1.grid(True)

    # Plot test accuracy vs average epoch time
    ax2.scatter(avg_times, test_accs)
    for i, name in enumerate(names):
        ax2.annotate(
            name,
            (avg_times[i], test_accs[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    ax2.set_title("Test Accuracy vs. Training Time")
    ax2.set_xlabel("Average Time per Epoch (s)")
    ax2.set_ylabel("Test Accuracy")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\nArchitecture Comparison:")
    print("-" * 80)
    print(
        f"{'Architecture':<35} | {'Params':<10} | {'Test Acc':<10} | {'Avg Time':<10}"
    )
    print("-" * 80)

    for name in names:
        print(
            f"{name:<35} | {results[name]['param_count']:<10,} | "
            f"{results[name]['test_acc']:<10.4f} | "
            f"{results[name]['avg_epoch_time']:<10.2f}s"
        )


def main():
    print("Exercise 3.1: Experimenting with CNN Architecture\n")

    # Run architecture experiments
    experiment_with_architectures()

    print("\nExercise Tasks:")
    print("1. Analyze the results and identify which architecture performed best.")
    print(
        "2. What is the relationship between model complexity (parameter count) and performance?"
    )
    print("3. How does training time scale with model complexity?")
    print("4. Create your own custom architecture with different hyperparameters:")
    print("   - Try adding or removing convolutional layers")
    print("   - Experiment with different filter counts")
    print("   - Test different kernel sizes or stride values")
    print("   - Consider adding dropout or batch normalization")
    print("5. Document your findings and explain the trade-offs you observed.")


if __name__ == "__main__":
    main()
