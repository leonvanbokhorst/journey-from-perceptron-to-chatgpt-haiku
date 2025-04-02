"""
Exercise 2.3: Function Approximation with MLPs

In this exercise, you will build a two-layer MLP to approximate a sine function on [0, 2π].
This will demonstrate the universal approximation capabilities of neural networks.
You will also explore how well your network interpolates between training points and
extrapolates outside the training range.

Learning Objectives:
1. Implement an MLP for regression (continuous output)
2. Visualize how neural networks learn to approximate continuous functions
3. Study the effects of model architecture on approximation quality
4. Understand the limitations of neural networks in extrapolation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SineMLP(nn.Module):
    """
    Multi-layer perceptron for approximating the sine function
    """

    def __init__(self, hidden_size=20):
        super(SineMLP, self).__init__()
        self.hidden = nn.Linear(1, hidden_size)
        self.activation = nn.Tanh()  # Tanh often works well for function approximation
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.activation(self.hidden(x))
        return self.output(x)


def generate_data(n_samples=100, train_range=(0, 2 * np.pi), noise=0.0):
    """
    Generate sine function data with optional noise

    Args:
        n_samples: Number of samples to generate
        train_range: Range of x values as tuple (min, max)
        noise: Standard deviation of Gaussian noise to add

    Returns:
        x_data: Tensor of x values
        y_data: Tensor of sin(x) values with optional noise
    """
    # Generate evenly spaced points in the specified range
    x = np.linspace(train_range[0], train_range[1], n_samples)

    # Compute sine values
    y = np.sin(x)

    # Add noise if specified
    if noise > 0:
        y += np.random.normal(0, noise, size=y.shape)

    # Reshape x for model input (N, 1)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Convert to PyTorch tensors
    x_data = torch.tensor(x, dtype=torch.float32)
    y_data = torch.tensor(y, dtype=torch.float32)

    return x_data, y_data


def train_model(model, x_data, y_data, epochs=2000, lr=0.01):
    """
    Train the MLP model on sine function data

    Args:
        model: The MLP model to train
        x_data: Input data tensor
        y_data: Target data tensor
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        losses: List of loss values during training
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Print progress
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return losses


def evaluate_model(
    model, train_data=None, range_to_plot=(-0.5, 2 * np.pi + 0.5), n_points=200
):
    """
    Evaluate and visualize the model's approximation of the sine function

    Args:
        model: Trained MLP model
        train_data: Optional tuple of (x_train, y_train) to show training points
        range_to_plot: Range of x values to plot
        n_points: Number of points to evaluate
    """
    model.eval()

    # Generate evaluation points (including points outside training range)
    x_eval = np.linspace(range_to_plot[0], range_to_plot[1], n_points).reshape(-1, 1)
    x_eval_tensor = torch.tensor(x_eval, dtype=torch.float32)

    # Get model predictions
    with torch.no_grad():
        y_pred = model(x_eval_tensor).numpy()

    # Compute true sine values
    y_true = np.sin(x_eval)

    # Plot results
    plt.figure(figsize=(10, 6))

    # Plot true sine function
    plt.plot(x_eval, y_true, "b-", label="True sine function", linewidth=2)

    # Plot model predictions
    plt.plot(x_eval, y_pred, "r--", label="MLP approximation", linewidth=2)

    # Plot training points if provided
    if train_data is not None:
        x_train, y_train = train_data
        plt.scatter(
            x_train.numpy(), y_train.numpy(), c="g", label="Training points", alpha=0.5
        )

    # Add vertical lines to mark training range if known
    if train_data is not None:
        x_min, x_max = x_train.min().item(), x_train.max().item()
        plt.axvline(x=x_min, color="k", linestyle="--", alpha=0.3)
        plt.axvline(x=x_max, color="k", linestyle="--", alpha=0.3)
        plt.text(x_min, -1.5, "Train start", rotation=90)
        plt.text(x_max, -1.5, "Train end", rotation=90)

    plt.title("Sine Function Approximation with MLP")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    plt.grid(True)
    plt.ylim(-1.5, 1.5)
    plt.tight_layout()
    plt.show()


def plot_loss_curve(losses):
    """Plot the training loss curve"""
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def experiment_with_hidden_sizes():
    """
    Experiment with different hidden layer sizes and compare approximation quality
    """
    # Generate training data
    x_train, y_train = generate_data(n_samples=50)

    # Test different hidden layer sizes
    hidden_sizes = [5, 20, 100]

    plt.figure(figsize=(15, 5))

    for i, hidden_size in enumerate(hidden_sizes):
        # Create and train model
        model = SineMLP(hidden_size=hidden_size)
        losses = train_model(model, x_train, y_train, epochs=2000)

        # Generate evaluation points
        x_eval = np.linspace(-0.5, 2 * np.pi + 0.5, 200).reshape(-1, 1)
        x_eval_tensor = torch.tensor(x_eval, dtype=torch.float32)

        # Get model predictions
        with torch.no_grad():
            y_pred = model(x_eval_tensor).numpy()

        # Compute true sine values
        y_true = np.sin(x_eval)

        # Plot in subplot
        plt.subplot(1, 3, i + 1)
        plt.plot(x_eval, y_true, "b-", label="True sine")
        plt.plot(x_eval, y_pred, "r--", label=f"MLP (h={hidden_size})")
        plt.scatter(
            x_train.numpy(),
            y_train.numpy(),
            c="g",
            s=20,
            alpha=0.4,
            label="Training points",
        )

        # Add vertical lines to mark training range
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.axvline(x=2 * np.pi, color="k", linestyle="--", alpha=0.3)

        plt.title(f"Hidden Size = {hidden_size}")
        plt.xlabel("x")
        plt.ylabel("sin(x)")
        plt.grid(True)
        plt.ylim(-1.5, 1.5)
        plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    print("Exercise 2.3: Function Approximation with MLPs\n")

    # Part 1: Generate and visualize data
    print("Generating sine function data...")
    x_train, y_train = generate_data(n_samples=50)

    # Plot the training data
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train.numpy(), y_train.numpy(), c="b", label="Training data")
    plt.plot(
        np.linspace(0, 2 * np.pi, 100),
        np.sin(np.linspace(0, 2 * np.pi, 100)),
        "r-",
        label="True sine function",
    )
    plt.title("Sine Function Training Data")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Part 2: Train a basic MLP and visualize results
    print("\nTraining MLP to approximate sine function...")
    model = SineMLP(hidden_size=20)
    losses = train_model(model, x_train, y_train)

    # Plot loss curve
    plot_loss_curve(losses)

    # Evaluate model including extrapolation
    print("\nEvaluating model (including extrapolation)...")
    evaluate_model(model, train_data=(x_train, y_train))

    # Part 3: Experiment with different hidden layer sizes
    print("\nExperimenting with different hidden layer sizes...")
    experiment_with_hidden_sizes()

    print("\nExercise Tasks:")
    print(
        "1. Try increasing the number of training points. How does this affect approximation quality?"
    )
    print(
        "2. Add noise to the training data (use the noise parameter in generate_data). How robust is the model?"
    )
    print(
        "3. Change the activation function from Tanh to ReLU. How does this affect the smoothness of the approximation?"
    )
    print(
        "4. Add a second hidden layer to the model. Does this improve the approximation?"
    )
    print("5. Try approximating a more complex function like sin(x) + sin(2x) or x².")
    print(
        "6. Explain why the model performs worse in the extrapolation region (outside training range)."
    )


if __name__ == "__main__":
    main()
