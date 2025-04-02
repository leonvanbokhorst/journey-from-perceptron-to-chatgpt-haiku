"""
Perceptron in PyTorch - Module 1
--------------------------------
A PyTorch implementation of the perceptron algorithm.
Demonstrating the connection between the original algorithm and modern frameworks.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class PyTorchPerceptron(nn.Module):
    """
    A perceptron implemented using PyTorch's neural network module
    """

    def __init__(self, input_size):
        """
        Initialize the perceptron layer

        Args:
            input_size: Number of input features
        """
        super(PyTorchPerceptron, self).__init__()

        # Create a single linear layer (no activation function yet)
        self.layer = nn.Linear(input_size, 1)

        # Initialize weights with small random values
        nn.init.normal_(self.layer.weight, mean=0, std=0.1)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        """
        Forward pass through the perceptron

        Args:
            x: Input tensor

        Returns:
            Raw output from linear layer
        """
        return self.layer(x)

    def predict(self, x):
        """
        Make binary predictions

        Args:
            x: Input tensor

        Returns:
            Binary predictions (0 or 1)
        """
        with torch.no_grad():
            # Apply step function to raw outputs
            return (self.forward(x) > 0).float()


def train_perceptron(model, inputs, targets, learning_rate=0.1, epochs=100):
    """
    Train the perceptron using PyTorch

    Args:
        model: PyTorch perceptron model
        inputs: Training data
        targets: Target values
        learning_rate: Learning rate for updates
        epochs: Maximum number of training iterations

    Returns:
        List of errors per epoch
    """
    errors_history = []

    # Manually implement perceptron learning rule
    for epoch in range(epochs):
        error_count = 0

        for i in range(len(inputs)):
            # Forward pass to get prediction
            x = inputs[i]
            y_true = targets[i]

            # Get binary prediction
            y_pred = model.predict(x)

            # Update weights only if prediction is wrong
            if y_pred != y_true:
                error_count += 1

                # Manually update weights using perceptron learning rule
                error = y_true - y_pred

                with torch.no_grad():
                    # Update weights: w = w + learning_rate * error * x
                    model.layer.weight += learning_rate * error * x

                    # Update bias: b = b + learning_rate * error
                    model.layer.bias += learning_rate * error

        # Record errors for this epoch
        errors_history.append(error_count)

        # Check for convergence
        if error_count == 0:
            print(f"Converged after {epoch + 1} epochs!")
            break

    return errors_history


def plot_decision_boundary(model, inputs, targets):
    """
    Plot the decision boundary of the perceptron

    Args:
        model: Trained perceptron model
        inputs: Input tensor
        targets: Target tensor
    """
    # Convert to numpy for plotting
    inputs_np = inputs.numpy()
    targets_np = targets.numpy()

    # Create meshgrid for visualization
    x_min, x_max = inputs_np[:, 0].min() - 0.5, inputs_np[:, 0].max() + 0.5
    y_min, y_max = inputs_np[:, 1].min() - 0.5, inputs_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Get predictions for all grid points
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = model.predict(grid).numpy().reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(inputs_np[:, 0], inputs_np[:, 1], c=targets_np, edgecolors="k")

    # Get weights and bias for the line equation
    w = model.layer.weight.detach().numpy()
    b = model.layer.bias.detach().numpy()

    # Plot decision boundary line: w_1*x + w_2*y + b = 0
    # Solving for y: y = (-w_1*x - b) / w_2
    slope = -w[0, 0] / w[0, 1]
    intercept = -b[0] / w[0, 1]

    x_vals = np.array([x_min, x_max])
    y_vals = slope * x_vals + intercept

    plt.plot(x_vals, y_vals, "r-", label="Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("PyTorch Perceptron Decision Boundary")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Training a PyTorch perceptron on the OR logic gate...")
    print("Truth table for OR:")
    print("0 OR 0 = 0")
    print("0 OR 1 = 1")
    print("1 OR 0 = 1")
    print("1 OR 1 = 1")

    # Prepare training data as tensors
    or_inputs = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    or_targets = torch.FloatTensor([[0], [1], [1], [1]])

    # Create and train model
    model = PyTorchPerceptron(input_size=2)
    errors = train_perceptron(
        model, or_inputs, or_targets, learning_rate=0.1, epochs=20
    )

    # Test the model
    print("\nTesting the trained model:")
    for i in range(len(or_inputs)):
        x = or_inputs[i]
        pred = model.predict(x).item()
        print(f"{int(x[0].item())} OR {int(x[1].item())} → {int(pred)}")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(errors)
    plt.xlabel("Epoch")
    plt.ylabel("Number of Errors")
    plt.title("Training Progress")
    plt.grid(True)
    plt.show()

    # Plot decision boundary
    plot_decision_boundary(model, or_inputs, or_targets.squeeze())

    # Haiku moment
    print("\nFrameworks evolve but,")
    print("The humble perceptron's heart")
    print("Remains the same: learn.")
