"""
Exercise 4.1: Testing Short-Term Memory in RNNs

In this exercise, you will implement a simple RNN to test its short-term memory capabilities.
You'll generate random sequences of 0s and 1s and train the network to predict the next
element in the sequence. By varying the sequence length, you can explore the limits
of a vanilla RNN's memory.

Learning Objectives:
1. Implement a basic RNN model in PyTorch
2. Understand how RNNs process sequential data
3. Observe how sequence length affects the RNN's memory capacity
4. Identify the limitations of vanilla RNNs for long-term dependencies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleRNN(nn.Module):
    """
    A simple RNN for binary sequence prediction.

    The network takes as input a sequence of binary values and predicts the next value
    in the sequence.
    """

    def __init__(self, input_size=1, hidden_size=20, output_size=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)

        # Forward pass through RNN
        out, hidden = self.rnn(x, hidden)

        # Apply output layer to all time steps
        out = self.fc(out)

        # Apply sigmoid activation
        out = self.sigmoid(out)

        return out, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden state with zeros"""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


def generate_binary_sequences(seq_length, batch_size=64, test=False):
    """
    Generate random binary sequences.

    Args:
        seq_length: Length of each sequence
        batch_size: Number of sequences to generate
        test: If True, generates test data with more samples

    Returns:
        X: Input sequences (batch_size, seq_length, 1)
        y: Target values (batch_size, seq_length, 1)
    """
    # For test data, generate more samples for better evaluation
    actual_batch_size = batch_size * 5 if test else batch_size

    # Generate random binary sequences
    sequences = np.random.randint(0, 2, (actual_batch_size, seq_length + 1)).astype(
        np.float32
    )

    # Input: sequences[:, :-1], targets: sequences[:, 1:]
    X = sequences[:, :-1].reshape(actual_batch_size, seq_length, 1)
    y = sequences[:, 1:].reshape(actual_batch_size, seq_length, 1)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


def generate_memorization_sequences(seq_length, batch_size=64, test=False):
    """
    Generate sequences where the target is to output the first element after a delay.

    This tests the RNN's ability to remember the first input for a specific number of time steps.

    Args:
        seq_length: Length of each sequence
        batch_size: Number of sequences to generate
        test: If True, generates test data with more samples

    Returns:
        X: Input sequences (batch_size, seq_length, 1)
        y: Target values (batch_size, seq_length, 1)
    """
    actual_batch_size = batch_size * 5 if test else batch_size

    # Initialize arrays
    X = np.zeros((actual_batch_size, seq_length, 1), dtype=np.float32)
    y = np.zeros((actual_batch_size, seq_length, 1), dtype=np.float32)

    # For each sequence
    for i in range(actual_batch_size):
        # Set first element to 0 or 1 randomly
        first_element = np.random.randint(0, 2)
        X[i, 0, 0] = first_element

        # Set the last element of y to match the first element of X
        # This creates a task where the network needs to remember the first input
        # for seq_length-1 time steps
        y[i, -1, 0] = first_element

        # Fill rest of the sequence with random values
        X[i, 1:, 0] = np.random.randint(0, 2, (seq_length - 1))

        # For y, we set all but the last element to whatever is convenient
        # (e.g., matching inputs) since we'll only evaluate on the last prediction
        y[i, :-1, 0] = X[i, 1:, 0]

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


def train_rnn(model, X, y, epochs=100, lr=0.01, task="next_element"):
    """
    Train the RNN model.

    Args:
        model: The RNN model to train
        X: Input sequences
        y: Target sequences
        epochs: Number of training epochs
        lr: Learning rate
        task: Task type ('next_element' or 'memorization')

    Returns:
        model: Trained model
        losses: List of training losses
        accuracies: List of training accuracies
    """
    model = model.to(device)
    X, y = X.to(device), y.to(device)

    # Binary cross-entropy loss
    criterion = nn.BCELoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For tracking progress
    losses = []
    accuracies = []

    for epoch in range(epochs):
        # Initialize hidden state
        hidden = model.init_hidden(X.size(0))

        # Forward pass
        outputs, hidden = model(X, hidden)

        # For memorization task, we only care about the last output
        if task == "memorization":
            loss = criterion(outputs[:, -1], y[:, -1])

            # Calculate accuracy (for the last element only)
            predictions = (outputs[:, -1] > 0.5).float()
            accuracy = (predictions == y[:, -1]).float().mean().item()
        else:
            # For next element prediction, we care about all outputs
            loss = criterion(outputs, y)

            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y).float().mean().item()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss and accuracy
        losses.append(loss.item())
        accuracies.append(accuracy)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
            )

    return model, losses, accuracies


def evaluate_rnn(model, X, y, task="next_element"):
    """
    Evaluate the RNN model.

    Args:
        model: The RNN model to evaluate
        X: Input sequences
        y: Target sequences
        task: Task type ('next_element' or 'memorization')

    Returns:
        accuracy: Overall accuracy
        per_position_accuracy: Accuracy at each position in the sequence (for next_element task)
    """
    model.eval()
    X, y = X.to(device), y.to(device)

    with torch.no_grad():
        # Initialize hidden state
        hidden = model.init_hidden(X.size(0))

        # Forward pass
        outputs, _ = model(X, hidden)

        # Convert outputs to binary predictions
        predictions = (outputs > 0.5).float()

        if task == "memorization":
            # For memorization task, we only care about the last prediction
            accuracy = (predictions[:, -1] == y[:, -1]).float().mean().item()
            per_position_accuracy = None
        else:
            # Overall accuracy
            accuracy = (predictions == y).float().mean().item()

            # Calculate accuracy at each position
            per_position_accuracy = []
            for i in range(y.size(1)):
                pos_acc = (predictions[:, i] == y[:, i]).float().mean().item()
                per_position_accuracy.append(pos_acc)

    return accuracy, per_position_accuracy


def experiment_with_sequence_lengths(task="next_element"):
    """
    Run experiments with different sequence lengths to test RNN memory.

    Args:
        task: Task type ('next_element' or 'memorization')

    Returns:
        results: Dictionary of results
    """
    sequence_lengths = [5, 10, 20, 50, 100]
    results = {
        "lengths": sequence_lengths,
        "train_accuracies": [],
        "test_accuracies": [],
        "per_position_accuracies": [],
    }

    for seq_length in sequence_lengths:
        print(f"\nExperiment with sequence length: {seq_length}")

        # Generate data
        if task == "memorization":
            X_train, y_train = generate_memorization_sequences(seq_length)
            X_test, y_test = generate_memorization_sequences(seq_length, test=True)
        else:
            X_train, y_train = generate_binary_sequences(seq_length)
            X_test, y_test = generate_binary_sequences(seq_length, test=True)

        # Create and train model
        model = SimpleRNN(input_size=1, hidden_size=50, output_size=1)
        model, losses, train_accuracies = train_rnn(
            model, X_train, y_train, epochs=100, lr=0.01, task=task
        )

        # Evaluate model
        test_accuracy, per_position_accuracy = evaluate_rnn(
            model, X_test, y_test, task=task
        )

        # Store results
        results["train_accuracies"].append(train_accuracies[-1])
        results["test_accuracies"].append(test_accuracy)
        if per_position_accuracy:
            results["per_position_accuracies"].append(per_position_accuracy)

        print(f"Final train accuracy: {train_accuracies[-1]:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

    # Plot results
    plot_results(results, task)

    return results


def plot_results(results, task):
    """
    Plot the results of the experiments.

    Args:
        results: Dictionary of results
        task: Task type ('next_element' or 'memorization')
    """
    # Plot overall accuracy vs sequence length
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["lengths"], results["train_accuracies"], "o-", label="Train Accuracy"
    )
    plt.plot(
        results["lengths"], results["test_accuracies"], "o-", label="Test Accuracy"
    )
    plt.xlabel("Sequence Length")
    plt.ylabel("Accuracy")
    plt.title(
        f'RNN Accuracy vs Sequence Length ({task.replace("_", " ").title()} Task)'
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # For next_element task, plot accuracy at each position for longest sequence
    if task == "next_element" and len(results["per_position_accuracies"]) > 0:
        longest_seq = results["lengths"][-1]
        pos_accuracies = results["per_position_accuracies"][-1]

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, longest_seq + 1), pos_accuracies, "o-")
        plt.xlabel("Position in Sequence")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy at Each Position (Sequence Length {longest_seq})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def demonstrate_memory_capacity():
    """
    Demonstrate the memory capacity of the RNN by visualizing its predictions
    on specific sequences.
    """
    # Create a fixed test sequence where the first element needs to be remembered
    seq_length = 20
    batch_size = 1

    # Create a sequence where the model should remember the first element
    X = torch.zeros((batch_size, seq_length, 1))
    y = torch.zeros((batch_size, seq_length, 1))

    # Set first element to 1
    X[0, 0, 0] = 1.0
    # The target for the last element is the same as the first input
    y[0, -1, 0] = 1.0

    # Train a model on this specific pattern
    model = SimpleRNN(input_size=1, hidden_size=20, output_size=1)
    model, losses, accuracies = train_rnn(
        model, X, y, epochs=500, lr=0.01, task="memorization"
    )

    # Now visualize the hidden state activity and predictions
    X = X.to(device)
    model.eval()

    with torch.no_grad():
        hidden = model.init_hidden(batch_size)
        outputs = []
        hidden_states = []

        # Collect outputs and hidden states for each time step
        for t in range(seq_length):
            x_t = X[:, t : t + 1, :]  # Shape: (batch_size, 1, 1)
            output, hidden = model.rnn(x_t, hidden)
            outputs.append(model.sigmoid(model.fc(output)).item())
            hidden_states.append(hidden[0, 0, :].cpu().numpy())

    # Plot the input, outputs, and hidden activity
    plt.figure(figsize=(15, 6))

    # Plot input and output
    plt.subplot(2, 1, 1)
    plt.plot(X[0, :, 0].cpu().numpy(), label="Input", drawstyle="steps-post")
    plt.plot(outputs, label="Predicted Output", marker="o")
    plt.plot(y[0, :, 0].cpu().numpy(), label="Target", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Input and Output over Time")
    plt.legend()
    plt.grid(True)

    # Plot hidden state activity (first 5 units)
    plt.subplot(2, 1, 2)
    hidden_array = np.array(hidden_states)
    for i in range(min(5, model.hidden_size)):
        plt.plot(hidden_array[:, i], label=f"Hidden Unit {i+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Activation")
    plt.title("Hidden State Activity (First 5 Units)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    print("Exercise 4.1: Testing Short-Term Memory in RNNs\n")

    # Part 1: Run experiments with different sequence lengths for next element prediction
    print("Part 1: Testing RNN memory with next element prediction task")
    next_element_results = experiment_with_sequence_lengths(task="next_element")

    # Part 2: Run experiments with different sequence lengths for memorization task
    print("\nPart 2: Testing RNN memory with memorization task")
    memorization_results = experiment_with_sequence_lengths(task="memorization")

    # Part 3: Demonstrate memory capacity visually
    print("\nPart 3: Visualizing RNN memory capacity")
    demonstrate_memory_capacity()

    print("\nExercise Tasks:")
    print("1. Analyze the results. At what sequence length does the RNN start to fail?")
    print(
        "2. For the next element prediction task, does accuracy decrease uniformly across positions?"
    )
    print(
        "3. For the memorization task, how well can the RNN remember the first element as sequence length increases?"
    )
    print(
        "4. What happens if you increase the hidden size? Try modifying the code to experiment."
    )
    print(
        "5. Try adding more layers to the RNN. Does this improve its memory capacity?"
    )


if __name__ == "__main__":
    main()
