import torch
import torch.nn as nn
import torch.nn.functional as F  # Functional provides activation functions
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# --- 1. Define the CNN Structure ---
class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network for MNIST classification.

    Consists of two convolutional layers followed by max pooling,
    and a final fully connected layer.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: [Batch, 1, 28, 28]
        # Layer 1: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        # After conv1: [Batch, 8, 28, 28]
        # After ReLU: [Batch, 8, 28, 28]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool: [Batch, 8, 14, 14]

        # Layer 2: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # After conv2: [Batch, 16, 14, 14]
        # After ReLU: [Batch, 16, 14, 14]
        # After pool: [Batch, 16, 7, 7]

        # Layer 3: Fully Connected for classification
        # Flatten the output from conv layers: 16 channels * 7 height * 7 width
        self.fc = nn.Linear(16 * 7 * 7, 10)  # 10 output classes

    def forward(self, x):
        """Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor (MNIST images) of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor (logits) of shape (batch_size, 10).
        """
        # Pass through Layer 1
        x = self.pool(F.relu(self.conv1(x)))
        # Pass through Layer 2
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Reshape to [Batch, 16*7*7]
        # Pass through final fully connected layer
        x = self.fc(x)
        return x


# --- 2. Set Hyperparameters ---
LEARNING_RATE = 0.001  # Often good for Adam with CNNs
BATCH_SIZE = 64
NUM_EPOCHS = 3  # Keep short for demo


# --- 3. Prepare MNIST Data ---
def prepare_mnist_data(batch_size):
    """Downloads, transforms, and prepares MNIST dataset loaders.

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        tuple: A tuple containing (train_loader, test_loader, train_dataset, test_dataset).
    """
    print("Preparing MNIST dataset...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST specific mean/std
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Dataset loaded. Train examples: {len(train_dataset)}, Test examples: {len(test_dataset)}"
    )
    return train_loader, test_loader, train_dataset, test_dataset


# --- 4. Instantiate Model, Loss, Optimizer ---
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nCNN Model Structure:")
print(model)


# --- 5. Training Loop (Identical structure to MLP) ---
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """Runs the training loop for the given model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion: The loss function.
        optimizer: The optimizer.
        num_epochs (int): Number of epochs to train for.
    """
    print(f"\nStarting CNN Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # images shape: [BATCH_SIZE, 1, 28, 28]
            # labels shape: [BATCH_SIZE]

            # 1. Clear previous gradients
            optimizer.zero_grad()
            # 2. Forward pass: get model predictions (logits)
            outputs = model(images)
            # 3. Calculate the loss
            loss = criterion(outputs, labels)
            # 4. Backward pass: compute gradients
            loss.backward()
            # 5. Update model parameters
            optimizer.step()

            running_loss += loss.item()

            # Print progress periodically
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_epoch_loss = running_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f} ---")

    print("Training Finished.")


# --- 6. Evaluation Loop (Identical structure to MLP) ---
def evaluate_model(model, test_loader):
    """Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test data.

    Returns:
        float: The accuracy of the model on the test set.
    """
    print("\nStarting CNN Evaluation...")
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed for evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the CNN on the {total} test images: {accuracy:.2f} %")
    print("Evaluation Finished.")
    return accuracy


# --- 7. Visualize Conv1 Filters ---
def visualize_filters(model):
    """Visualizes the filters from the first convolutional layer (conv1).

    Args:
        model (SimpleCNN): The trained CNN model.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        print("\nVisualizing filters from the first convolutional layer...")

        model.eval()  # Ensure model is in eval mode

        # Get the weights of the first convolutional layer (conv1)
        # Shape: [out_channels, in_channels, kernel_height, kernel_width]
        filters = model.conv1.weight.data.clone()

        # Normalize filters for better visualization
        # Rescale weights to 0-1 range per filter
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        # Move to CPU and convert to numpy
        filters = filters.cpu().numpy()

        # Number of filters to visualize
        num_filters = filters.shape[0]  # Should be 8 in our case
        # Determine grid size (e.g., 2x4 or 4x2 for 8 filters)
        grid_cols = 4
        grid_rows = int(np.ceil(num_filters / grid_cols))

        fig, axes = plt.subplots(
            grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2)
        )
        fig.suptitle("Learned Filters (conv1)")

        # Handle cases where axes is not a 2D array (e.g., single row/column)
        if grid_rows == 1 and grid_cols == 1:
            axes = np.array([[axes]])
        elif grid_rows == 1:
            axes = axes.reshape(1, -1)
        elif grid_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_filters):
            # Get the filter (it has shape [in_channels, H, W], here in_channels=1)
            filt = filters[i, 0, :, :]  # Select the first (and only) input channel
            row_idx = i // grid_cols
            col_idx = i % grid_cols
            ax = axes[row_idx, col_idx]
            ax.imshow(filt, cmap="gray")  # Display filter as grayscale
            ax.set_xticks([])  # Remove axis ticks
            ax.set_yticks([])
            ax.set_title(f"Filter {i+1}")

        # Hide unused subplots
        for i in range(num_filters, grid_rows * grid_cols):
            row_idx = i // grid_cols
            col_idx = i % grid_cols
            if row_idx < axes.shape[0] and col_idx < axes.shape[1]:
                axes[row_idx, col_idx].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
        plt.show()
        print("Filter visualization complete. Check the plot window.")

    except ImportError:
        print(
            "\nVisualization skipped: matplotlib not found. Install it with 'pip install matplotlib'"
        )


# --- Main Execution Logic ---
if __name__ == "__main__":
    """Main script execution: setup, train, evaluate, visualize."""
    # Setup
    train_loader, test_loader, _, _ = prepare_mnist_data(BATCH_SIZE)
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("\nCNN Model Structure:")
    print(model)

    # Train
    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)

    # Evaluate
    evaluate_model(model, test_loader)

    # Visualize
    visualize_filters(model)
