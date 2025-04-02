"""
Perceptron Visualization - Module 1
-----------------------------------
Visual exploration of perceptron capabilities and limitations,
focusing on the XOR problem that drove the development of multilayer networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def create_logic_gate_data():
    """
    Create data for the basic logic gates: AND, OR, NAND, XOR

    Returns:
        Dictionary containing inputs and outputs for each gate
    """
    # Input data - same for all gates
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Output data for different gates
    and_outputs = np.array([0, 0, 0, 1])
    or_outputs = np.array([0, 1, 1, 1])
    nand_outputs = np.array([1, 1, 1, 0])
    xor_outputs = np.array([0, 1, 1, 0])

    return {
        "inputs": inputs,
        "AND": and_outputs,
        "OR": or_outputs,
        "NAND": nand_outputs,
        "XOR": xor_outputs,
    }


def plot_logic_gates_data():
    """
    Visualize the data points for different logic gates
    """
    data = create_logic_gate_data()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    gates = ["AND", "OR", "NAND", "XOR"]
    colors = ["blue", "green", "red", "purple"]

    for i, gate in enumerate(gates):
        ax = axs[i]
        inputs = data["inputs"]
        outputs = data[gate]

        # Plot 0s and 1s with different markers
        for j, output in enumerate(outputs):
            marker = "o" if output == 1 else "x"
            ax.scatter(
                inputs[j, 0],
                inputs[j, 1],
                marker=marker,
                s=100,
                color=colors[i],
                label=f"Output: {output}",
            )

        # Add labels and decorations
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Input 1")
        ax.set_ylabel("Input 2")
        ax.set_title(f"{gate} Gate", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add legend (only once)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(), by_label.keys(), loc="upper right", fontsize=12
            )

    plt.tight_layout()
    plt.suptitle("Visualizing Logic Gates in 2D Space", fontsize=16, y=1.02)
    plt.show()

    print("Observe that AND, OR, and NAND are linearly separable.")
    print("For XOR, no single straight line can separate the classes!")


def plot_3d_xor():
    """
    Create a 3D visualization of XOR, showing how adding a dimension
    makes the problem linearly separable
    """
    data = create_logic_gate_data()
    inputs = data["inputs"]
    xor_outputs = data["XOR"]

    # Create a 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the original points in 3D space
    # For XOR, we'll create a new feature z = x XOR y = x + y - 2*x*y
    z = inputs[:, 0] + inputs[:, 1] - 2 * inputs[:, 0] * inputs[:, 1]

    # Plot 0s and 1s with different colors
    for i, output in enumerate(xor_outputs):
        marker = "o" if output == 1 else "x"
        color = "green" if output == 1 else "red"
        ax.scatter(
            inputs[i, 0],
            inputs[i, 1],
            z[i],
            marker=marker,
            s=100,
            color=color,
            label=f"Output: {output}",
        )

    # Create the plane that separates the classes
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * np.ones(X.shape)  # Plane at z = 0.5

    # Plot the plane
    ax.plot_surface(X, Y, Z, alpha=0.3, color="blue")

    # Add labels and decoration
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_zlabel("New Feature (Input 1 XOR Input 2)")
    ax.set_title("XOR Problem in 3D Space", fontsize=14)

    # Add legend (only once)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.show()

    print("By adding a third dimension (a hidden layer in neural network terms),")
    print("the XOR problem becomes linearly separable!")
    print("This is why we need multilayer perceptrons for more complex problems.")


def attempt_xor_training():
    """
    Demonstrate that a single perceptron can't learn XOR
    """
    print("\nAttempting to train a single perceptron on XOR...")

    # Create PyTorch tensors for XOR
    data = create_logic_gate_data()
    x = torch.FloatTensor(data["inputs"])
    y = torch.FloatTensor(data["XOR"]).unsqueeze(1)

    # Create model, loss function, and optimizer
    model = nn.Linear(2, 1)  # Single perceptron
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Training loop
    num_epochs = 1000
    losses = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        losses.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss for XOR with Single Perceptron")
    plt.grid(True)
    plt.show()

    # Test the trained model
    with torch.no_grad():
        test_outputs = (torch.sigmoid(model(x)) > 0.5).float()
        accuracy = (test_outputs == y).float().mean()

        print(f"Final test accuracy: {accuracy.item() * 100:.2f}%")
        print("\nPredictions vs. Actual:")
        for i in range(len(x)):
            inputs = x[i].numpy()
            actual = y[i].item()
            pred = test_outputs[i].item()
            print(f"Input: {inputs}, Predicted: {int(pred)}, Actual: {int(actual)}")

    print("\nObservation: The single perceptron fails to learn XOR completely!")
    print("This limitation spurred the development of multilayer perceptrons,")
    print("which we'll explore in the next module.")


if __name__ == "__main__":
    print("Visualizing Logic Gates and the XOR Problem")
    print("===========================================")
    print("\nThis visualization demonstrates why the XOR problem")
    print("was so significant in neural network history.")

    plot_logic_gates_data()

    input("\nPress Enter to see the XOR problem in 3D space...")
    plot_3d_xor()

    input("\nPress Enter to see a perceptron attempt to learn XOR...")
    attempt_xor_training()

    # End with a haiku
    print("\nSingle line fails here")
    print("XOR defies perceptron")
    print("New layers await.")
