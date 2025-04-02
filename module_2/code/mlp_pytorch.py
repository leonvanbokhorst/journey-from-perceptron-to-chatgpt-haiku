"""
Multilayer Perceptron in PyTorch - Module 2
-------------------------------------------
Implementation of a multilayer perceptron using PyTorch,
demonstrating the power of hidden layers and backpropagation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)


class MLP(nn.Module):
    """
    A configurable multilayer perceptron with customizable architecture
    """

    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU()):
        """
        Initialize the multilayer perceptron

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output units
            activation: Activation function to use (default: ReLU)
        """
        super(MLP, self).__init__()

        # Store network architecture specs
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation

        # Create network layers
        layers = []

        # Input layer to first hidden layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation)
            prev_size = hidden_size

        # Last hidden layer to output layer
        layers.append(nn.Linear(prev_size, output_size))

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor

        Returns:
            Output predictions
        """
        return self.model(x)

    def get_num_parameters(self):
        """
        Count the total number of trainable parameters

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dataset():
    """
    Create a synthetic dataset (two moons) for binary classification

    Returns:
        Training and testing data and labels as PyTorch tensors
    """
    # Generate two moons dataset
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X, y


def train_model(
    model, X_train, y_train, X_test, y_test, epochs=1000, learning_rate=0.01
):
    """
    Train the MLP model

    Args:
        model: The MLP model to train
        X_train, y_train: Training data
        X_test, y_test: Testing data
        epochs: Number of training iterations
        learning_rate: Learning rate for optimizer

    Returns:
        Lists of training and testing losses
    """
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # For storing the training history
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record training loss
        train_losses.append(loss.item())

        # Calculate test loss
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}"
            )

    return train_losses, test_losses


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model

    Args:
        model: Trained MLP model
        X_test, y_test: Testing data

    Returns:
        Accuracy score
    """
    with torch.no_grad():
        # Get predictions
        outputs = model(X_test)
        predicted = (torch.sigmoid(outputs) > 0.5).float()

        # Calculate accuracy
        accuracy = (predicted == y_test).float().mean()

    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
    return accuracy.item()


def plot_decision_boundary(model, X, y):
    """
    Plot the decision boundary of the trained model

    Args:
        model: Trained MLP model
        X, y: Full dataset for visualization
    """
    # Define bounds of the plot
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create a mesh grid
    h = 0.01  # mesh step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict on all mesh grid points
    Z = np.zeros(xx.shape)

    # Convert grid to PyTorch tensor
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    with torch.no_grad():
        logits = model(grid)
        probs = torch.sigmoid(logits)
        Z = probs.numpy().reshape(xx.shape)

    # Plot decision boundary and data points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=2)

    # Plot the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Decision Boundary - MLP with {model.hidden_sizes} hidden units")
    plt.show()


def plot_training_history(train_losses, test_losses):
    """
    Plot the training and testing loss curves

    Args:
        train_losses: List of training losses
        test_losses: List of testing losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_architectures():
    """
    Compare different MLP architectures (number of hidden layers and units)
    """
    # Load dataset
    X_train, X_test, y_train, y_test, X, y = create_dataset()

    # Define architectures to compare
    architectures = [
        [3],  # 1 hidden layer with 3 units
        [10],  # 1 hidden layer with 10 units
        [5, 5],  # 2 hidden layers with 5 units each
        [10, 10],  # 2 hidden layers with 10 units each
        [10, 10, 10],  # 3 hidden layers with 10 units each
    ]

    # Store results
    results = []

    # Test each architecture
    for hidden_sizes in architectures:
        print(f"\nTraining MLP with hidden layers: {hidden_sizes}")

        # Create and train model
        model = MLP(input_size=2, hidden_sizes=hidden_sizes, output_size=1)
        train_losses, test_losses = train_model(
            model, X_train, y_train, X_test, y_test, epochs=1000
        )

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)

        # Store results
        results.append(
            {
                "architecture": hidden_sizes,
                "accuracy": accuracy,
                "parameters": model.get_num_parameters(),
            }
        )

        # Plot decision boundary
        plot_decision_boundary(model, X, y)

    # Display comparison
    print("\nArchitecture Comparison:")
    print("-----------------------")
    for result in results:
        arch_str = " -> ".join([str(x) for x in [2] + result["architecture"] + [1]])
        print(
            f"Architecture: {arch_str}, Accuracy: {result['accuracy']*100:.2f}%, Parameters: {result['parameters']}"
        )


def xor_example():
    """
    Demonstrate solving the XOR problem with an MLP
    """
    print("\nSolving the XOR Problem with MLP")
    print("-------------------------------")

    # Create XOR dataset
    X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.FloatTensor([[0], [1], [1], [0]])

    # Create model with one hidden layer
    model = MLP(input_size=2, hidden_sizes=[4], output_size=1)

    # Train model
    train_losses, _ = train_model(model, X, y, X, y, epochs=5000, learning_rate=0.01)

    # Evaluate model
    with torch.no_grad():
        outputs = model(X)
        predicted = (torch.sigmoid(outputs) > 0.5).float()

        print("\nXOR Truth Table Predictions:")
        for i in range(len(X)):
            print(
                f"{int(X[i][0].item())} XOR {int(X[i][1].item())} = {int(predicted[i].item())} (Expected: {int(y[i].item())})"
            )

    # Plot XOR decision boundary
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    with torch.no_grad():
        Z = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors="k", s=100)
    plt.title("XOR Decision Boundary - MLP with 1 Hidden Layer")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.show()

    # Plot training curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("XOR Training Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("Multilayer Perceptron in PyTorch")
    print("===============================")

    # Solve XOR problem
    xor_example()

    # Compare different architectures
    compare_architectures()

    # Haiku moment
    print("\nHaiku:")
    print("Hidden layers dance,")
    print("Weaving patterns from chaos,")
    print("Neurons enlightened.")
