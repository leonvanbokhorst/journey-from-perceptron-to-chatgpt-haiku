import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# --- 1. Define the MLP Structure ---
class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron with one hidden layer.

    Args:
        input_size (int): Dimension of the input features (e.g., 784 for MNIST).
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of output classes (e.g., 10 for MNIST).
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28) for MNIST.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Flatten the image (Batch x 1 x 28 x 28) -> (Batch x 784)
        x = x.view(x.size(0), -1)
        hidden = self.activation(self.fc1(x))
        output = self.fc2(hidden)
        return output


# --- 2. Set Hyperparameters ---
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10  # 10 digits (0-9)
LEARNING_RATE = 0.01  # Adam often works well with smaller LRs
BATCH_SIZE = 64
NUM_EPOCHS = 3  # Keep it short for a quick demo

# MNIST specific
INPUT_SIZE = 28 * 28  # 784 pixels flattened


# --- 3. Prepare MNIST Data ---
def prepare_mnist_data(batch_size):
    """Downloads, transforms, and prepares MNIST dataset loaders.

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        tuple: A tuple containing (train_loader, test_loader, train_dataset, test_dataset).
    """
    print("Preparing MNIST dataset...")
    # Transformations: Convert images to PyTorch tensors and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST specific mean/std
        ]
    )

    # Download or load training data
    train_dataset = torchvision.datasets.MNIST(
        root="./data",  # Directory to save data
        train=True,  # Get the training split
        transform=transform,
        download=True,  # Download if not present!
    )

    # Download or load test data
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True,  # Get the test split
    )

    # Create DataLoaders for batching
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Dataset loaded. Train examples: {len(train_dataset)}, Test examples: {len(test_dataset)}"
    )
    return train_loader, test_loader, train_dataset, test_dataset


# --- 4. Instantiate Model, Loss, Optimizer ---
model = SimpleMLP(
    input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE
)
criterion = nn.CrossEntropyLoss()  # Includes LogSoftmax + NLLLoss
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)  # Adam is often faster than SGD

print("\nMLP Model Structure:")
print(model)


# --- 5. Training Loop ---
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """Runs the training loop for the given model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion: The loss function.
        optimizer: The optimizer.
        num_epochs (int): Number of epochs to train for.
    """
    print(f"\nStarting Training for {num_epochs} epochs...")
    model.train()  # Set model to training mode
    running_loss = 0.0
    for epoch in range(num_epochs):
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


# --- 6. Evaluation Loop ---
def evaluate_model(model, test_loader):
    """Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test data.

    Returns:
        float: The accuracy of the model on the test set.
    """
    print("\nStarting Evaluation...")
    model.eval()  # Set model to evaluation mode (important for dropout, batchnorm etc.)
    correct = 0
    total = 0
    # No need to track gradients during evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the {total} test images: {accuracy:.2f} %")
    print("Evaluation Finished.")
    return accuracy


# --- 7. Visualize First Layer Weights ---
def visualize_weights(model):
    """Visualizes the weights of the first layer of the MLP.

    Args:
        model (nn.Module): The trained MLP model (assumes fc1 is the first layer).
    """
    print("\nVisualizing first layer weights...")
    model.eval()  # Ensure model is in eval mode

    # Get the weights of the first fully connected layer
    weights = model.fc1.weight.data

    # We need to detach the weights from the computation graph and move to CPU if necessary
    weights = weights.cpu().numpy()

    # Determine grid size for plotting (e.g., plot the first 16 hidden unit weights)
    num_to_plot = min(weights.shape[0], 16)  # Plot up to 16 weights
    grid_size = int(np.ceil(np.sqrt(num_to_plot)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    fig.suptitle("First Layer Learned Weights (Reshaped to 28x28)")

    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            # Get the weights for the i-th hidden neuron and reshape to 28x28
            img = weights[i].reshape(28, 28)
            ax.imshow(img, cmap="gray")  # Display as grayscale image
            ax.set_title(f"Neuron {i+1}")
            ax.axis("off")  # Hide axes ticks
        else:
            # Hide unused subplots
            ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    # If running in an environment that doesn't automatically show plots:
    plt.show()
    print("Visualization complete. Check the plot window.")


# --- Main Execution Logic ---
if __name__ == "__main__":
    """Main script execution: setup, train, evaluate, visualize."""
    # Setup
    train_loader, test_loader, train_dataset, test_dataset = prepare_mnist_data(
        BATCH_SIZE
    )
    model = SimpleMLP(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("\nMLP Model Structure:")
    print(model)

    # Train
    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)

    # Evaluate
    evaluate_model(model, test_loader)

    # Visualize
    try:
        visualize_weights(model)
    except ImportError:
        print("\nVisualization skipped: matplotlib/numpy not found.")
