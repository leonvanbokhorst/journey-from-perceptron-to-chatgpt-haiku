"""
Exercise 2.2: Manual Backpropagation Implementation

In this exercise, you will implement backpropagation manually for a simple 2-layer network
and compare your computed gradients with PyTorch's autograd. This will solidify your
understanding of the chain rule and how automatic differentiation works.

Learning Objectives:
1. Understand the mathematics behind backpropagation
2. Implement gradient computation manually using the chain rule
3. Verify your implementation against PyTorch's autograd
4. Gain intuition for how gradients flow through a neural network
"""

import torch
import torch.nn as nn
import numpy as np


class TwoLayerNetManual:
    """
    A simple 2-layer neural network implemented with manual forward/backward passes.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the network parameters.
        """
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

        # Store intermediate values for backprop
        self.x = None
        self.z1 = None  # Pre-activation at layer 1
        self.a1 = None  # Post-activation at layer 1
        self.z2 = None  # Pre-activation at layer 2
        self.a2 = None  # Output

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, x):
        """
        Forward pass computation.

        Args:
            x: Input data, shape (batch_size, input_size)

        Returns:
            Output after forward pass, shape (batch_size, output_size)
        """
        # Store input
        self.x = x

        # First layer
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Second layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, y):
        """
        Backward pass computation.

        Args:
            y: Ground truth labels, shape (batch_size, output_size)

        Returns:
            Dictionary containing gradients with respect to parameters
        """
        # Number of samples
        m = y.shape[0]

        # Gradient of loss with respect to output
        dz2 = self.a2 - y

        # Gradients for W2 and b2
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m

        # Gradient backpropagated to hidden layer
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.z1)

        # Gradients for W1 and b1
        dW1 = np.dot(self.x.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def compute_loss(self, y_pred, y_true):
        """
        Compute binary cross-entropy loss.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Average loss value
        """
        m = y_true.shape[0]
        loss = (
            -np.sum(
                y_true * np.log(y_pred + 1e-8)
                + (1 - y_true) * np.log(1 - y_pred + 1e-8)
            )
            / m
        )
        return loss


class TwoLayerNetPyTorch(nn.Module):
    """
    Same 2-layer neural network implemented with PyTorch for autograd comparison.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNetPyTorch, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid1(self.linear1(x))
        x = self.sigmoid2(self.linear2(x))
        return x


def compare_gradients(manual_net, torch_net, X, y):
    """
    Compare gradients computed manually vs. with PyTorch's autograd.

    Args:
        manual_net: Manual implementation network
        torch_net: PyTorch implementation network
        X: Input data
        y: Target outputs

    Returns:
        Dictionary of gradient differences
    """
    # Convert numpy data to PyTorch tensors
    X_torch = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor(y, dtype=torch.float32)

    # Copy weights from manual net to PyTorch net to ensure identical starting point
    with torch.no_grad():
        torch_net.linear1.weight.copy_(
            torch.tensor(manual_net.W1.T, dtype=torch.float32)
        )
        torch_net.linear1.bias.copy_(torch.tensor(manual_net.b1, dtype=torch.float32))
        torch_net.linear2.weight.copy_(
            torch.tensor(manual_net.W2.T, dtype=torch.float32)
        )
        torch_net.linear2.bias.copy_(torch.tensor(manual_net.b2, dtype=torch.float32))

    # Forward and backward pass with manual implementation
    manual_output = manual_net.forward(X)
    manual_loss = manual_net.compute_loss(manual_output, y)
    manual_grads = manual_net.backward(y)

    # Forward and backward pass with PyTorch
    torch_output = torch_net(X_torch)
    criterion = nn.BCELoss()
    torch_loss = criterion(torch_output, y_torch)
    torch_loss.backward()

    # Extract gradients from PyTorch model
    torch_grads = {
        "W1": torch_net.linear1.weight.grad.numpy().T,  # Transpose due to PyTorch's [out_features, in_features] ordering
        "b1": torch_net.linear1.bias.grad.numpy(),
        "W2": torch_net.linear2.weight.grad.numpy().T,
        "b2": torch_net.linear2.bias.grad.numpy(),
    }

    # Compute differences
    diff = {
        "W1": np.abs(manual_grads["W1"] - torch_grads["W1"]),
        "b1": np.abs(manual_grads["b1"] - torch_grads["b1"]),
        "W2": np.abs(manual_grads["W2"] - torch_grads["W2"]),
        "b2": np.abs(manual_grads["b2"] - torch_grads["b2"]),
    }

    print(f"Manual Loss: {manual_loss:.6f}, PyTorch Loss: {torch_loss.item():.6f}")
    print(f"Maximum gradient difference for W1: {np.max(diff['W1']):.6f}")
    print(f"Maximum gradient difference for b1: {np.max(diff['b1']):.6f}")
    print(f"Maximum gradient difference for W2: {np.max(diff['W2']):.6f}")
    print(f"Maximum gradient difference for b2: {np.max(diff['b2']):.6f}")

    return diff


def main():
    print("Exercise 2.2: Manual Backpropagation Implementation\n")

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Network parameters
    input_size = 2
    hidden_size = 3
    output_size = 1

    # Create simple dataset: XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize networks
    manual_net = TwoLayerNetManual(input_size, hidden_size, output_size)
    torch_net = TwoLayerNetPyTorch(input_size, hidden_size, output_size)

    # Compare gradients
    print("Comparing manually computed gradients with PyTorch autograd:")
    diff = compare_gradients(manual_net, torch_net, X, y)

    # Verify gradient computation
    if np.max([np.max(diff[key]) for key in diff]) < 1e-5:
        print(
            "\nSuccess! Your manual backpropagation implementation matches PyTorch's autograd."
        )
    else:
        print(
            "\nThere are discrepancies between your implementation and PyTorch's autograd."
        )
        print("Check your manual implementation of the backward pass.")

    print("\nExercise Tasks:")
    print("1. Review the manual implementation of backpropagation in this file")
    print(
        "2. Try to identify the specific mathematical steps of the chain rule in the code"
    )
    print(
        "3. Experiment with a different activation function (e.g., tanh) and update both implementations"
    )
    print(
        "4. Add a third layer to both networks and verify the gradient computation still matches"
    )
    print(
        "5. Extend to multi-class classification with softmax output and cross-entropy loss"
    )


if __name__ == "__main__":
    main()
