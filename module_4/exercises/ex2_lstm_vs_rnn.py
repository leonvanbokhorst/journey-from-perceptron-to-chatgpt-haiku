"""
Exercise 4.2: Comparing Vanilla RNNs and LSTMs for Long-Term Dependencies

This exercise compares the performance of vanilla RNNs and LSTMs on tasks
that require remembering information over long sequences.

Learning Objectives:
1. Implement both RNN and LSTM models in PyTorch
2. Compare their performance on tasks with long-term dependencies
3. Visualize and understand the vanishing gradient problem
4. Appreciate why LSTMs were developed to address RNN limitations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class VanillaRNN(nn.Module):
    """Simple RNN model implementing vanilla recurrent architecture"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid for binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0))

        # Forward propagate through RNN
        out, hidden = self.rnn(x, hidden)

        # Pass through fully connected layer
        out = self.fc(out)

        # Apply sigmoid activation
        out = self.sigmoid(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class LSTM(nn.Module):
    """LSTM model for sequence prediction"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid for binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        # Initialize hidden and cell states if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0))

        # Forward propagate through LSTM
        out, hidden = self.lstm(x, hidden)

        # Pass through fully connected layer
        out = self.fc(out)

        # Apply sigmoid activation
        out = self.sigmoid(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)


def generate_add_problem_data(seq_length, batch_size, test=False):
    """
    Generate data for the adding problem.

    The adding problem: Given a sequence of random numbers and a binary marker,
    sum the two numbers where the marker is 1.

    Args:
        seq_length: Length of each sequence
        batch_size: Number of sequences to generate
        test: If True, generates test data with more samples

    Returns:
        X: Input with shape (batch_size, seq_length, 2)
            X[:, :, 0] contains the random numbers
            X[:, :, 1] contains the binary markers
        y: Target with shape (batch_size, 1)
            The sum of the two marked numbers in each sequence
    """
    # Increase batch size for test data for better evaluation
    actual_batch_size = batch_size * 5 if test else batch_size

    # Random numbers between 0 and 1
    numbers = np.random.uniform(0, 1, (actual_batch_size, seq_length))

    # Initialize markers with zeros
    markers = np.zeros((actual_batch_size, seq_length))

    # Place two 1s in each sequence at random positions
    for i in range(actual_batch_size):
        # Choose two distinct positions
        positions = np.random.choice(seq_length, size=2, replace=False)
        markers[i, positions] = 1

    # Compute the target: sum of the two marked numbers
    y = np.sum(numbers * markers, axis=1, keepdims=True)

    # Stack numbers and markers to create input
    X = np.stack([numbers, markers], axis=2).astype(np.float32)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


def train_model(
    model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01, task="add_problem"
):
    """
    Train the given model.

    Args:
        model: The model to train (RNN or LSTM)
        X_train: Training input
        y_train: Training targets
        X_val: Validation input
        y_val: Validation targets
        epochs: Number of training epochs
        lr: Learning rate
        task: Task type

    Returns:
        model: Trained model
        results: Dictionary containing training statistics
    """
    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    # For regression task (adding problem)
    criterion = nn.MSELoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For tracking progress
    train_losses = []
    val_losses = []

    # For tracking gradients
    gradient_norms = {"input_layer": [], "hidden_layer": [], "output_layer": []}

    # Time the training
    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        # Initialize hidden state
        if isinstance(model, LSTM):
            hidden = (
                model.init_hidden(X_train.size(0))[0].detach(),
                model.init_hidden(X_train.size(0))[1].detach(),
            )
        else:
            hidden = model.init_hidden(X_train.size(0)).detach()

        # Forward pass
        outputs, hidden = model(X_train, hidden)

        # For adding problem, only the last output matters
        loss = criterion(outputs[:, -1, 0], y_train.squeeze())

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()

        # Track gradient norms
        if epoch % 10 == 0:
            # Track input layer gradients
            if hasattr(model, "rnn"):
                for name, param in model.rnn.named_parameters():
                    if "weight_ih" in name:
                        gradient_norms["input_layer"].append(param.grad.norm().item())
                    elif "weight_hh" in name:
                        gradient_norms["hidden_layer"].append(param.grad.norm().item())
            elif hasattr(model, "lstm"):
                for name, param in model.lstm.named_parameters():
                    if "weight_ih" in name:
                        gradient_norms["input_layer"].append(param.grad.norm().item())
                    elif "weight_hh" in name:
                        gradient_norms["hidden_layer"].append(param.grad.norm().item())

            # Track output layer gradients
            for name, param in model.fc.named_parameters():
                if "weight" in name:
                    gradient_norms["output_layer"].append(param.grad.norm().item())

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            if isinstance(model, LSTM):
                val_hidden = (
                    model.init_hidden(X_val.size(0))[0],
                    model.init_hidden(X_val.size(0))[1],
                )
            else:
                val_hidden = model.init_hidden(X_val.size(0))

            val_outputs, _ = model(X_val, val_hidden)
            val_loss = criterion(val_outputs[:, -1, 0], y_val.squeeze())

        # Record losses
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
            )

    # Calculate training time
    training_time = time.time() - start_time

    # Prepare results
    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "gradient_norms": gradient_norms,
        "training_time": training_time,
    }

    return model, results


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.

    Args:
        model: The trained model
        X_test: Test input
        y_test: Test targets

    Returns:
        mse: Mean squared error
        predictions: Model predictions
    """
    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.no_grad():
        if isinstance(model, LSTM):
            hidden = (
                model.init_hidden(X_test.size(0))[0],
                model.init_hidden(X_test.size(0))[1],
            )
        else:
            hidden = model.init_hidden(X_test.size(0))

        outputs, _ = model(X_test, hidden)

        # For adding problem, only the last output matters
        predictions = outputs[:, -1, 0].cpu().numpy()
        targets = y_test.squeeze().cpu().numpy()

        # Calculate mean squared error
        mse = np.mean((predictions - targets) ** 2)

    return mse, predictions


def plot_results(rnn_results, lstm_results, sequence_lengths):
    """
    Plot the results of the experiments.

    Args:
        rnn_results: Results from RNN model
        lstm_results: Results from LSTM model
        sequence_lengths: List of sequence lengths used in experiments
    """
    # Plot training curves for one sequence length
    seq_length = sequence_lengths[1]  # Use the second sequence length for visualization

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rnn_results[seq_length]["train_losses"], label="Train Loss (RNN)")
    plt.plot(rnn_results[seq_length]["val_losses"], label="Val Loss (RNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"RNN Training Curve (Seq Length = {seq_length})")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(lstm_results[seq_length]["train_losses"], label="Train Loss (LSTM)")
    plt.plot(lstm_results[seq_length]["val_losses"], label="Val Loss (LSTM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"LSTM Training Curve (Seq Length = {seq_length})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot final MSE vs sequence length
    rnn_mses = [rnn_results[length]["test_mse"] for length in sequence_lengths]
    lstm_mses = [lstm_results[length]["test_mse"] for length in sequence_lengths]

    plt.figure(figsize=(10, 6))
    plt.plot(sequence_lengths, rnn_mses, "o-", label="RNN")
    plt.plot(sequence_lengths, lstm_mses, "o-", label="LSTM")
    plt.xlabel("Sequence Length")
    plt.ylabel("Test MSE")
    plt.title("Model Performance vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")  # Log scale for better visualization
    plt.tight_layout()
    plt.show()

    # Plot gradient norms for longest sequence
    longest_seq = sequence_lengths[-1]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    for layer, norms in rnn_results[longest_seq]["gradient_norms"].items():
        if norms:  # Only plot if there are values
            epochs = np.arange(0, len(norms) * 10, 10)
            plt.plot(epochs, norms, "o-", label=f"{layer}")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title(f"RNN Gradient Norms (Seq Length = {longest_seq})")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")  # Log scale for better visualization

    plt.subplot(1, 2, 2)
    for layer, norms in lstm_results[longest_seq]["gradient_norms"].items():
        if norms:  # Only plot if there are values
            epochs = np.arange(0, len(norms) * 10, 10)
            plt.plot(epochs, norms, "o-", label=f"{layer}")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title(f"LSTM Gradient Norms (Seq Length = {longest_seq})")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")  # Log scale for better visualization

    plt.tight_layout()
    plt.show()

    # Plot training time vs sequence length
    rnn_times = [rnn_results[length]["training_time"] for length in sequence_lengths]
    lstm_times = [lstm_results[length]["training_time"] for length in sequence_lengths]

    plt.figure(figsize=(10, 6))
    plt.plot(sequence_lengths, rnn_times, "o-", label="RNN")
    plt.plot(sequence_lengths, lstm_times, "o-", label="LSTM")
    plt.xlabel("Sequence Length")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predictions(rnn_model, lstm_model, seq_length=100, num_samples=50):
    """
    Plot predictions from both models against targets for visual comparison.

    Args:
        rnn_model: Trained RNN model
        lstm_model: Trained LSTM model
        seq_length: Sequence length to use
        num_samples: Number of test samples to plot
    """
    # Generate test data
    X_test, y_test = generate_add_problem_data(seq_length, num_samples, test=True)

    # Get predictions from both models
    rnn_mse, rnn_preds = evaluate_model(rnn_model, X_test, y_test)
    lstm_mse, lstm_preds = evaluate_model(lstm_model, X_test, y_test)

    # Plot predictions vs targets
    plt.figure(figsize=(12, 6))

    # Sort by target value for better visualization
    targets = y_test.squeeze().numpy()
    sorted_indices = np.argsort(targets)
    targets = targets[sorted_indices]
    rnn_preds = rnn_preds[sorted_indices]
    lstm_preds = lstm_preds[sorted_indices]

    plt.plot(targets, "o-", label="Targets")
    plt.plot(rnn_preds, "x-", label=f"RNN (MSE={rnn_mse:.4f})")
    plt.plot(lstm_preds, "d-", label=f"LSTM (MSE={lstm_mse:.4f})")
    plt.xlabel("Sample Index (sorted by target value)")
    plt.ylabel("Value")
    plt.title(f"Model Predictions vs Targets (Seq Length = {seq_length})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_experiments():
    """
    Run experiments comparing RNN and LSTM on the adding problem with various sequence lengths.

    Returns:
        rnn_results: Dictionary of RNN results for each sequence length
        lstm_results: Dictionary of LSTM results for each sequence length
        sequence_lengths: List of sequence lengths used
    """
    # Hyperparameters
    hidden_size = 64
    batch_size = 128
    epochs = 100
    lr = 0.01

    # Sequence lengths to test
    sequence_lengths = [10, 50, 100, 200, 500]

    # Dictionaries to store results
    rnn_results = {}
    lstm_results = {}

    # For each sequence length
    for seq_length in sequence_lengths:
        print(f"\n--- Experiments with sequence length: {seq_length} ---")

        # Generate data
        X_train, y_train = generate_add_problem_data(seq_length, batch_size)
        X_val, y_val = generate_add_problem_data(seq_length, batch_size // 2)
        X_test, y_test = generate_add_problem_data(
            seq_length, batch_size * 2, test=True
        )

        print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")

        # Train and evaluate RNN
        print("\nTraining Vanilla RNN...")
        rnn_model = VanillaRNN(input_size=2, hidden_size=hidden_size, output_size=1)
        rnn_model, rnn_train_results = train_model(
            rnn_model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr
        )
        rnn_mse, _ = evaluate_model(rnn_model, X_test, y_test)
        print(f"RNN Test MSE: {rnn_mse:.4f}")

        # Train and evaluate LSTM
        print("\nTraining LSTM...")
        lstm_model = LSTM(input_size=2, hidden_size=hidden_size, output_size=1)
        lstm_model, lstm_train_results = train_model(
            lstm_model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr
        )
        lstm_mse, _ = evaluate_model(lstm_model, X_test, y_test)
        print(f"LSTM Test MSE: {lstm_mse:.4f}")

        # Store results
        rnn_results[seq_length] = {
            **rnn_train_results,
            "test_mse": rnn_mse,
            "model": rnn_model,
        }

        lstm_results[seq_length] = {
            **lstm_train_results,
            "test_mse": lstm_mse,
            "model": lstm_model,
        }

        # For mid-length sequence, plot predictions
        if seq_length == 100:
            plot_predictions(rnn_model, lstm_model, seq_length)

    # Plot combined results
    plot_results(rnn_results, lstm_results, sequence_lengths)

    return rnn_results, lstm_results, sequence_lengths


def main():
    print("Exercise 4.2: Comparing Vanilla RNNs and LSTMs for Long-Term Dependencies\n")

    print(
        "This exercise compares the performance of RNNs and LSTMs on the adding problem,"
    )
    print("which requires remembering information over long sequences.\n")

    # Run experiments
    rnn_results, lstm_results, sequence_lengths = run_experiments()

    print("\nExperiments completed!")
    print("\nExercise Tasks:")
    print(
        "1. Analyze the results. How does sequence length affect the performance of RNNs vs LSTMs?"
    )
    print(
        "2. Look at the gradient norms. Do you observe the vanishing gradient problem in RNNs?"
    )
    print(
        "3. For what sequence lengths does the RNN fail to learn the task effectively?"
    )
    print(
        "4. Why do LSTMs perform better than RNNs on tasks with long-term dependencies?"
    )
    print(
        "5. Try increasing the hidden size or using multiple layers. Does this help RNNs overcome their limitations?"
    )


if __name__ == "__main__":
    main()
