"""
Exercise 2.1: XOR Problem with Multi-Layer Perceptron

In this exercise, you will implement a Multi-Layer Perceptron (MLP) to solve the XOR problem,
which cannot be solved by a single-layer perceptron. You will experiment with different
numbers of hidden neurons to find the minimum required to solve this problem.

Learning Objectives:
1. Understand why the XOR problem requires a hidden layer
2. Implement a basic MLP in PyTorch
3. Experiment with model architecture to find minimal requirements
4. Visualize decision boundaries to understand what the network learns

The XOR (exclusive OR) truth table:
Input A | Input B | Output
   0    |    0    |   0
   0    |    1    |   1
   1    |    0    |   1
   1    |    1    |   0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Create the XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# Part 1: Single-layer perceptron (to demonstrate its failure)
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2 inputs, 1 output
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(x))


# Part 2: Multi-layer perceptron with configurable hidden neurons
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, hidden_size)
        self.activation = nn.ReLU()  # ReLU activation
        self.output = nn.Linear(hidden_size, 1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        return self.final_activation(self.output(x))


def train_model(model, X, y, epochs=5000, lr=0.01):
    """Train a model and return training history"""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return losses


def visualize_decision_boundary(model, title):
    """Visualize the decision boundary created by the model"""
    # Create a grid of points
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Make predictions on the grid
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    Z = model(grid).detach().numpy().reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.RdBu, alpha=0.6)

    # Plot the training data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors="k")
    plt.title(title)
    plt.xlabel("Input A")
    plt.ylabel("Input B")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_predictions(model, X, y):
    """Print model predictions compared to actual values"""
    predictions = model(X)
    print("\nPredictions vs Actual:")
    for i in range(len(X)):
        print(
            f"Input: {X[i].numpy()}, Predicted: {predictions[i].item():.4f}, Actual: {y[i].item()}"
        )


def main():
    print("Exercise 2.1: Solving the XOR Problem with MLPs\n")

    # Part 1: Demonstrate that a perceptron fails on XOR
    print("Part 1: Training a single-layer perceptron (should fail to learn XOR)")
    perceptron = Perceptron()
    perceptron_losses = train_model(perceptron, X, y)
    print_predictions(perceptron, X, y)
    visualize_decision_boundary(
        perceptron, "Single-Layer Perceptron (Expected to Fail)"
    )

    # Part 2: Find the minimum number of hidden neurons needed to solve XOR
    print("\nPart 2: Finding the minimum hidden neurons needed")

    for hidden_size in [1, 2, 3, 4]:
        print(f"\nTraining MLP with {hidden_size} hidden neurons:")
        model = MLP(hidden_size)
        losses = train_model(model, X, y)

        final_loss = losses[-1]
        print(f"Final loss with {hidden_size} hidden neurons: {final_loss:.6f}")

        if final_loss < 0.01:  # Threshold for "solving" XOR
            print(f"Success! MLP with {hidden_size} hidden neurons can solve XOR.")
            print_predictions(model, X, y)
            visualize_decision_boundary(model, f"MLP with {hidden_size} Hidden Neurons")

            # Plot the loss curve
            plt.figure(figsize=(8, 6))
            plt.plot(losses)
            plt.title(f"Training Loss for MLP with {hidden_size} Hidden Neurons")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.show()

            # Only break after showing the first successful case
            break
        else:
            print(f"MLP with {hidden_size} hidden neurons failed to solve XOR.")

    print("\nExercise Tasks:")
    print("1. Run this code and observe when the MLP successfully solves XOR")
    print("2. Explain why the minimum number of hidden neurons is what you found")
    print(
        "3. Try changing the activation function to tanh or sigmoid and observe differences"
    )
    print("4. Draw the decision regions created by each successful hidden layer size")
    print("5. Can you relate the number of hidden neurons to regions in input space?")


if __name__ == "__main__":
    main()
