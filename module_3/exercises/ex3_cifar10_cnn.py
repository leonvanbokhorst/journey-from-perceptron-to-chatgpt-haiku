"""
Exercise 3.3: CNN for CIFAR-10 Image Classification

In this exercise, you will implement a CNN for the CIFAR-10 dataset, which contains
colored 32x32 images in 10 classes. This is more challenging than MNIST due to
the color channels, higher variability, and smaller, more detailed objects.

Learning Objectives:
1. Build and train a CNN for colored image classification
2. Implement techniques to improve CNN performance (data augmentation, dropout, etc.)
3. Evaluate model performance on a more challenging dataset
4. Experiment with different CNN architectures to find the best accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import DataLoader, random_split

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define class names for CIFAR-10
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class BasicCNN(nn.Module):
    """
    A basic CNN for CIFAR-10 classification
    """

    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class ImprovedCNN(nn.Module):
    """
    An improved CNN with regularization and more layers
    """

    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


def load_cifar10_data(batch_size=64, val_split=0.1, augment=False):
    """
    Load CIFAR-10 dataset with optional data augmentation
    """
    # Define transformations
    if augment:
        # With data augmentation for training
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        # Without data augmentation
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    # Test/validation transformations (no augmentation)
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load the training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )

    # Split into training and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Load the test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

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

        # Track statistics
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


def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    """Train the model and return training history"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )

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

        # Adjust learning rate
        scheduler.step(val_loss)

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
            f"Time: {epoch_time:.2f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
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


def visualize_samples(data_loader, num_samples=5):
    """Visualize sample images from the dataset"""
    # Get a batch of images
    images, labels = next(iter(data_loader))

    # Denormalize images for display
    images = images * 0.5 + 0.5  # Denormalize

    # Plot images
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(
            np.transpose(images[i].numpy(), (1, 2, 0))
        )  # Convert from (C,H,W) to (H,W,C)
        axes[i].set_title(f"{CIFAR10_CLASSES[labels[i]]}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def per_class_accuracy(model, test_loader):
    """Calculate and plot per-class accuracy"""
    model.eval()
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i, label in enumerate(labels):
                label = label.item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Calculate accuracy for each class
    class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(10)]

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.bar(CIFAR10_CLASSES, class_accuracy, color="skyblue")
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45)

    # Add percentage labels
    for i, accuracy in enumerate(class_accuracy):
        plt.text(i, accuracy + 1, f"{accuracy:.1f}%", ha="center")

    plt.tight_layout()
    plt.show()

    # Print results
    print("\nPer-Class Accuracy:")
    for i in range(10):
        print(f"{CIFAR10_CLASSES[i]}: {class_accuracy[i]:.1f}%")

    return class_accuracy


def visualize_predictions(model, test_loader, num_samples=5):
    """Visualize model predictions on sample images"""
    model.eval()

    # Get a batch of samples
    images, labels = next(iter(test_loader))

    # Make predictions
    images_to_display = images[:num_samples].to(device)
    with torch.no_grad():
        outputs = model(images_to_display)
        _, predicted = torch.max(outputs, 1)

    # Convert images back to displayable format
    images_to_display = images_to_display.cpu() * 0.5 + 0.5

    # Show predictions
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(np.transpose(images_to_display[i].numpy(), (1, 2, 0)))
        pred_class = CIFAR10_CLASSES[predicted[i].item()]
        true_class = CIFAR10_CLASSES[labels[i].item()]
        color = "green" if pred_class == true_class else "red"
        axes[i].set_title(f"Pred: {pred_class}\nTrue: {true_class}", color=color)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def compare_models():
    """Compare basic and improved CNN models with and without augmentation"""
    # Load data
    train_loader, val_loader, test_loader = load_cifar10_data(augment=False)
    train_loader_aug, val_loader_aug, _ = load_cifar10_data(augment=True)

    # Show some sample images
    print("Visualizing sample CIFAR-10 images:")
    visualize_samples(test_loader)

    # Train and evaluate models
    models = [
        {
            "name": "Basic CNN",
            "model": BasicCNN(),
            "train_loader": train_loader,
            "val_loader": val_loader,
            "epochs": 15,
        },
        {
            "name": "Basic CNN + Augmentation",
            "model": BasicCNN(),
            "train_loader": train_loader_aug,
            "val_loader": val_loader_aug,
            "epochs": 15,
        },
        {
            "name": "Improved CNN",
            "model": ImprovedCNN(),
            "train_loader": train_loader,
            "val_loader": val_loader,
            "epochs": 15,
        },
        {
            "name": "Improved CNN + Augmentation",
            "model": ImprovedCNN(),
            "train_loader": train_loader_aug,
            "val_loader": val_loader_aug,
            "epochs": 15,
        },
    ]

    results = {}

    for model_config in models:
        print(f"\nTraining {model_config['name']}...")

        # Train model
        history = train_model(
            model_config["model"],
            model_config["train_loader"],
            model_config["val_loader"],
            epochs=model_config["epochs"],
        )

        # Plot training history
        plot_history(history, title=f"Training History: {model_config['name']}")

        # Evaluate on test set
        test_loss, test_acc = evaluate(
            model_config["model"], test_loader, nn.CrossEntropyLoss(), device
        )
        print(f"Test accuracy for {model_config['name']}: {test_acc:.4f}")

        # Calculate per-class accuracy
        print(f"\nPer-class accuracy for {model_config['name']}:")
        class_acc = per_class_accuracy(model_config["model"], test_loader)

        # Visualize some predictions
        print(f"\nSample predictions from {model_config['name']}:")
        visualize_predictions(model_config["model"], test_loader)

        # Store results
        results[model_config["name"]] = {
            "history": history,
            "test_acc": test_acc,
            "class_acc": class_acc,
        }

    # Compare model test accuracies
    model_names = list(results.keys())
    test_accs = [results[name]["test_acc"] for name in model_names]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, test_accs, color="skyblue")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.title("Model Comparison")
    plt.xticks(rotation=45)

    # Add percentage labels
    for i, acc in enumerate(test_accs):
        plt.text(i, acc + 0.01, f"{acc*100:.1f}%", ha="center")

    plt.tight_layout()
    plt.show()


def main():
    print("Exercise 3.3: CNN for CIFAR-10 Classification\n")

    # Option 1: Run full model comparison (takes longer)
    compare_models()

    # Option 2: Just train the improved model with augmentation
    # (Uncomment to use this faster option instead)
    """
    print("Training Improved CNN with data augmentation...")
    train_loader, val_loader, test_loader = load_cifar10_data(augment=True)
    model = ImprovedCNN().to(device)
    history = train_model(model, train_loader, val_loader, epochs=15)
    plot_history(history, title="Training History: Improved CNN with Augmentation")
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"\nTest accuracy: {test_acc:.4f}")
    per_class_accuracy(model, test_loader)
    visualize_predictions(model, test_loader)
    """

    print("\nExercise Tasks:")
    print(
        "1. Analyze the results of the different models. Which performed best and why?"
    )
    print("2. How much improvement did data augmentation provide?")
    print("3. Which classes were easiest/hardest for the CNN to classify?")
    print(
        "4. Create your own custom CNN architecture to try to improve the results further."
    )
    print("   Suggestions to experiment with:")
    print("   - Try different numbers of filters or kernel sizes")
    print("   - Add more layers or use different pooling strategies")
    print("   - Adjust dropout rates or try other regularization techniques")
    print("   - Experiment with different learning rates or optimizers")
    print(
        "5. (Challenge) Implement a residual connection (skip connection) similar to ResNet."
    )


if __name__ == "__main__":
    main()
