"""
Exercise 3.2: CNN Feature Map Visualization

In this exercise, you will visualize the feature maps/activations of a convolutional
neural network to better understand what features it learns to detect. You will
train a simple CNN on MNIST and visualize its filters and activation patterns
when processing different digit images.

Learning Objectives:
1. Understand how to extract and visualize convolutional filters
2. Learn to visualize activation maps from different CNN layers
3. Interpret what kinds of features each filter is detecting
4. Gain insight into how CNNs build up representations hierarchically
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class VisualizableCNN(nn.Module):
    """
    A simple CNN for MNIST with hooks to capture feature maps
    """

    def __init__(self):
        super(VisualizableCNN, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # Second conv block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # Classifier
        self.fc = nn.Linear(32 * 7 * 7, 10)

        # Store activations
        self.activations = {}

    def forward(self, x):
        # First block
        x = self.conv1(x)
        self.activations["conv1"] = x.clone()
        x = self.relu1(x)
        self.activations["relu1"] = x.clone()
        x = self.pool1(x)
        self.activations["pool1"] = x.clone()

        # Second block
        x = self.conv2(x)
        self.activations["conv2"] = x.clone()
        x = self.relu2(x)
        self.activations["relu2"] = x.clone()
        x = self.pool2(x)
        self.activations["pool2"] = x.clone()

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def load_mnist_data(batch_size=64):
    """Load MNIST dataset"""
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load test data
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    """Train the CNN model on MNIST"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
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

        # Print epoch statistics
        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss = test_loss / total
        test_acc = correct / total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    return model


def visualize_filters(model, layer_name="conv1", save_file=None):
    """Visualize filters/weights of a convolutional layer"""
    # Get the weights of the specified layer
    if layer_name == "conv1":
        weights = model.conv1.weight.data.cpu().numpy()
    elif layer_name == "conv2":
        weights = model.conv2.weight.data.cpu().numpy()
    else:
        raise ValueError(f"Layer {layer_name} not recognized")

    # Number of filters
    num_filters = weights.shape[0]

    # Create figure
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Filters from {layer_name} layer", fontsize=16)

    # Create a grid of subplots
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    gs = gridspec.GridSpec(grid_size, grid_size, figure=fig)

    # Plot each filter
    for i in range(num_filters):
        ax = fig.add_subplot(gs[i])

        # For first layer, filters operate on a single channel
        if layer_name == "conv1":
            ax.imshow(weights[i, 0], cmap="viridis")
        else:
            # For deeper layers, take the average across input channels
            ax.imshow(np.mean(weights[i], axis=0), cmap="viridis")

        ax.set_title(f"Filter {i+1}")
        ax.axis("off")

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)

    if save_file:
        plt.savefig(save_file)
    plt.show()


def get_activation_maps(model, image):
    """Get activation maps for a single image"""
    model.eval()
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        _ = model(image)

    # Return activations
    return {key: val.cpu() for key, val in model.activations.items()}


def visualize_activation_maps(
    activations, layer_name, num_filters=None, save_file=None
):
    """Visualize activation maps from a specific layer"""
    # Get the activations for the specified layer
    if layer_name not in activations:
        raise ValueError(f"Layer {layer_name} not found in activations")

    activation = activations[layer_name][0].numpy()  # First item in batch

    # Number of filters/channels in this activation
    total_filters = activation.shape[0]
    if num_filters is None or num_filters > total_filters:
        num_filters = total_filters

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Activation Maps from {layer_name} layer", fontsize=16)

    # Create a grid of subplots
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    gs = gridspec.GridSpec(grid_size, grid_size, figure=fig)

    # Plot each activation map
    for i in range(num_filters):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(activation[i], cmap="viridis")
        ax.set_title(f"Channel {i+1}")
        ax.axis("off")

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)

    if save_file:
        plt.savefig(save_file)
    plt.show()


def visualize_digit_with_activations(model, test_loader, digit=None):
    """
    Select a sample image (optionally of a specific digit) and visualize
    all activation maps throughout the network
    """
    # Find an image of the requested digit
    if digit is not None:
        for images, labels in test_loader:
            mask = labels == digit
            if torch.any(mask):
                image = images[mask][0]
                break
    else:
        # Just take the first image
        images, labels = next(iter(test_loader))
        image = images[0]
        digit = labels[0].item()

    # Denormalize for visualization
    mean, std = 0.1307, 0.3081
    img_for_display = image.clone()
    img_for_display = img_for_display * std + mean

    # Get activations
    activations = get_activation_maps(model, image)

    # Show the original image
    plt.figure(figsize=(6, 6))
    plt.imshow(img_for_display.squeeze(), cmap="gray")
    plt.title(f"Original Image (Digit {digit})")
    plt.axis("off")
    plt.show()

    # Visualize activations for each layer
    for layer_name in ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2"]:
        visualize_activation_maps(activations, layer_name)


def compare_digits_activations(model, test_loader, digits=[0, 1, 4, 7]):
    """
    Compare activation patterns for different digits to understand
    what features the CNN is using to distinguish them
    """
    # Dictionary to store one image per requested digit
    digit_images = {}

    # Find an image for each requested digit
    for images, labels in test_loader:
        for digit in digits:
            if digit not in digit_images:
                mask = labels == digit
                if torch.any(mask):
                    digit_images[digit] = images[mask][0]

        # Check if we've found all digits
        if len(digit_images) == len(digits):
            break

    # Compare a specific layer's activations across digits
    layer_to_compare = "conv2"  # Choose a layer to analyze

    # Get activations for each digit
    digit_activations = {}
    for digit, image in digit_images.items():
        digit_activations[digit] = get_activation_maps(model, image)

    # Plot comparison - showing the same filter's response to different digits
    num_filters_to_show = 8  # Show first 8 filters for comparison

    fig, axes = plt.subplots(len(digits), num_filters_to_show, figsize=(15, 10))
    fig.suptitle(
        f"Comparing {layer_to_compare} Filter Responses Across Digits", fontsize=16
    )

    for i, digit in enumerate(digits):
        activation = digit_activations[digit][layer_to_compare][0].numpy()
        for j in range(num_filters_to_show):
            ax = axes[i, j]
            im = ax.imshow(activation[j], cmap="viridis")

            if i == 0:
                ax.set_title(f"Filter {j+1}")
            if j == 0:
                ax.set_ylabel(f"Digit {digit}")

            ax.axis("off")

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()

    # Also compare the original images
    fig, axes = plt.subplots(1, len(digits), figsize=(12, 3))
    for i, digit in enumerate(digits):
        mean, std = 0.1307, 0.3081
        img = digit_images[digit].clone() * std + mean
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_title(f"Digit {digit}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    print("Exercise 3.2: CNN Feature Map Visualization\n")

    # Load data
    train_loader, test_loader = load_mnist_data()

    # Create and train model
    model = VisualizableCNN().to(device)

    print("Training model...")
    model = train_model(model, train_loader, test_loader, epochs=5)

    # Part 1: Visualize convolutional filters
    print("\nVisualizing convolutional filters...")
    visualize_filters(model, layer_name="conv1")
    visualize_filters(model, layer_name="conv2")

    # Part 2: Visualize activation maps for a single image
    print("\nVisualizing activation maps for a single digit...")
    visualize_digit_with_activations(model, test_loader, digit=5)  # Try with digit 5

    # Part 3: Compare activations across different digits
    print("\nComparing activation patterns across different digits...")
    compare_digits_activations(model, test_loader)

    print("\nExercise Tasks:")
    print(
        "1. Analyze the visualized filters from conv1 and conv2. What patterns do you observe?"
    )
    print(
        "2. For the activation maps, identify which filters respond strongly to certain features."
    )
    print(
        "3. Compare how different digits activate different filters. Can you identify:"
    )
    print(
        "   - Filters that detect vertical lines? (Look at activations for digits with vertical strokes)"
    )
    print("   - Filters that detect curves? (Look at digits like 3, 8, etc.)")
    print(
        "4. Select two visually similar digits (e.g., 3 and 8) and analyze which filters best distinguish them."
    )
    print(
        "5. Take a specific digit and visualize its activations at each stage of the network."
    )
    print(
        "   What do you notice about how the representation changes from earlier to deeper layers?"
    )


if __name__ == "__main__":
    main()
