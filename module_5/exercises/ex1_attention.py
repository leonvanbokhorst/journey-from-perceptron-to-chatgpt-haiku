"""
Exercise 5.1: Understanding Attention Mechanisms

In this exercise, you will implement and visualize different attention mechanisms
to better understand how they work. You'll see how attention helps models focus
on relevant parts of the input sequences and improves performance on sequence tasks.

Learning Objectives:
1. Implement dot-product and additive attention
2. Visualize attention weights to understand the model's focus
3. Apply attention to a sequence classification task
4. Compare different attention mechanisms
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    Attention(Q, K, V) = softmax(Q·K^T/sqrt(d_k))·V
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, key, value):
        """
        Args:
            query: Tensor of shape [batch_size, query_len, d_k]
            key: Tensor of shape [batch_size, key_len, d_k]
            value: Tensor of shape [batch_size, key_len, d_v]

        Returns:
            context: Tensor of shape [batch_size, query_len, d_v]
            attention: Tensor of shape [batch_size, query_len, key_len]
        """
        # Calculate attention scores
        # [batch_size, query_len, d_k] x [batch_size, d_k, key_len] = [batch_size, query_len, key_len]
        scores = torch.bmm(query, key.transpose(1, 2)) / self.temperature

        # Apply softmax to get attention weights
        attention = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        # [batch_size, query_len, key_len] x [batch_size, key_len, d_v] = [batch_size, query_len, d_v]
        context = torch.bmm(attention, value)

        return context, attention


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) Attention mechanism.

    score(s_t, h_i) = v^T * tanh(W_1 * s_t + W_2 * h_i)
    """

    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(key_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, key, value):
        """
        Args:
            query: Tensor of shape [batch_size, query_len, query_dim]
            key: Tensor of shape [batch_size, key_len, key_dim]
            value: Tensor of shape [batch_size, key_len, value_dim]

        Returns:
            context: Tensor of shape [batch_size, query_len, value_dim]
            attention: Tensor of shape [batch_size, query_len, key_len]
        """
        # Reshape for broadcasting
        # [batch_size, query_len, query_dim] -> [batch_size, query_len, hidden_dim]
        query_proj = self.query_proj(query)

        # [batch_size, key_len, key_dim] -> [batch_size, key_len, hidden_dim]
        key_proj = self.key_proj(key)

        # Expand dimensions for broadcasting
        # [batch_size, query_len, 1, hidden_dim]
        query_proj_expanded = query_proj.unsqueeze(2)

        # [batch_size, 1, key_len, hidden_dim]
        key_proj_expanded = key_proj.unsqueeze(1)

        # Broadcast and combine
        # [batch_size, query_len, key_len, hidden_dim]
        combined = torch.tanh(query_proj_expanded + key_proj_expanded)

        # Calculate attention scores
        # [batch_size, query_len, key_len, hidden_dim] -> [batch_size, query_len, key_len, 1]
        scores = self.score(combined)

        # Remove the last dimension
        # [batch_size, query_len, key_len, 1] -> [batch_size, query_len, key_len]
        scores = scores.squeeze(-1)

        # Apply softmax to get attention weights
        attention = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        # [batch_size, query_len, key_len] x [batch_size, key_len, value_dim] = [batch_size, query_len, value_dim]
        context = torch.bmm(attention, value)

        return context, attention


class SelfAttentionClassifier(nn.Module):
    """
    Sequence classifier using self-attention mechanism.
    """

    def __init__(
        self, input_dim, hidden_dim, num_classes, attention_type="dot", dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Attention mechanism
        if attention_type == "dot":
            self.attention = DotProductAttention(temperature=np.sqrt(hidden_dim))
        elif attention_type == "additive":
            self.attention = AdditiveAttention(hidden_dim, hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            output: Class scores of shape [batch_size, num_classes]
            attention: Attention weights of shape [batch_size, seq_len, seq_len]
        """
        # Apply embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, hidden_dim]

        # Self-attention: use embedded as query, key, and value
        context, attention = self.attention(embedded, embedded, embedded)

        # Global max pooling
        pooled, _ = torch.max(context, dim=1)  # [batch_size, hidden_dim]

        # Apply dropout and output layer
        output = self.fc(self.dropout(pooled))  # [batch_size, num_classes]

        return output, attention


def generate_sequence_data(num_samples=1000, seq_len=10, num_classes=2, num_features=5):
    """
    Generate synthetic sequence data with class-specific patterns.

    Args:
        num_samples: Number of samples to generate
        seq_len: Length of each sequence
        num_classes: Number of classes
        num_features: Number of features for each element in the sequence

    Returns:
        X: Input sequences of shape [num_samples, seq_len, num_features]
        y: Class labels of shape [num_samples]
    """
    X = np.random.randn(num_samples, seq_len, num_features)
    y = np.random.randint(0, num_classes, size=num_samples)

    # Add class-specific pattern in random positions of each sequence
    for i in range(num_samples):
        class_id = y[i]

        # Generate a pattern specific to this class
        pattern = np.random.randn(3, num_features) + (class_id * 2 - 1) * 3

        # Place the pattern at random positions in the sequence
        start_pos = np.random.randint(0, seq_len - 3 + 1)
        X[i, start_pos : start_pos + 3, :] = pattern

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train_model(
    model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, lr=0.001
):
    """
    Train the sequence classifier.

    Args:
        model: The model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate

    Returns:
        model: The trained model
        history: Dictionary containing training history
    """
    # Move data to device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Number of batches
    n_batches = len(X_train) // batch_size

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0

        # Shuffle indices
        indices = torch.randperm(len(X_train))

        # Process each batch
        for i in range(n_batches):
            # Get batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            batch_indices = indices[start_idx:end_idx]

            # Get batch data
            X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

            # Forward pass
            optimizer.zero_grad()
            outputs, _ = model(X_batch)

            # Calculate loss
            loss = criterion(outputs, y_batch)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            train_loss += loss.item() * len(X_batch)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y_batch).sum().item()

        # Calculate training statistics
        train_loss /= len(X_train)
        train_acc = train_correct / len(X_train)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            outputs, _ = model(X_val)
            val_loss = criterion(outputs, y_val).item()
            _, predicted = torch.max(outputs, 1)
            val_correct = (predicted == y_val).sum().item()
            val_acc = val_correct / len(X_val)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print progress
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

    return model, history


def visualize_attention(model, X, y, num_samples=3):
    """
    Visualize attention weights for some samples.

    Args:
        model: The trained model
        X: Input sequences
        y: Class labels
        num_samples: Number of samples to visualize
    """
    model.eval()
    X_device = X.to(device)

    with torch.no_grad():
        # Forward pass to get attention weights
        _, attention = model(X_device)

    # Convert attention weights to numpy for visualization
    attention = attention.cpu().numpy()

    # Visualize attention weights for some samples
    plt.figure(figsize=(15, 4 * num_samples))

    for i in range(min(num_samples, len(X))):
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(attention[i], cmap="viridis")
        plt.colorbar()
        plt.title(f"Sample {i}, Class {y[i].item()}")
        plt.xlabel("Key position")
        plt.ylabel("Query position")

    plt.tight_layout()
    plt.show()


def compare_attention_mechanisms(X_train, y_train, X_val, y_val):
    """
    Compare dot-product and additive attention mechanisms.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        dot_model: Model with dot-product attention
        add_model: Model with additive attention
        dot_history: Training history for dot-product attention
        add_history: Training history for additive attention
    """
    # Get dimensions from data
    _, seq_len, input_dim = X_train.shape
    num_classes = len(torch.unique(y_train))
    hidden_dim = 64

    # Create models
    dot_model = SelfAttentionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        attention_type="dot",
    ).to(device)

    add_model = SelfAttentionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        attention_type="additive",
    ).to(device)

    # Train models
    print("Training model with dot-product attention:")
    dot_model, dot_history = train_model(dot_model, X_train, y_train, X_val, y_val)

    print("\nTraining model with additive attention:")
    add_model, add_history = train_model(add_model, X_train, y_train, X_val, y_val)

    # Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(dot_history["train_acc"], label="Dot-Train")
    plt.plot(dot_history["val_acc"], label="Dot-Val")
    plt.plot(add_history["train_acc"], label="Add-Train")
    plt.plot(add_history["val_acc"], label="Add-Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dot_history["train_loss"], label="Dot-Train")
    plt.plot(dot_history["val_loss"], label="Dot-Val")
    plt.plot(add_history["train_loss"], label="Add-Train")
    plt.plot(add_history["val_loss"], label="Add-Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return dot_model, add_model, dot_history, add_history


def main():
    """Main function to run the exercise."""
    print("Exercise 5.1: Understanding Attention Mechanisms\n")

    # Parameters
    num_samples = 1000
    seq_len = 15
    num_features = 8
    num_classes = 3

    # Generate data
    print("Generating sequence data...")
    X, y = generate_sequence_data(
        num_samples=num_samples,
        seq_len=seq_len,
        num_features=num_features,
        num_classes=num_classes,
    )

    # Split data
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"             X_val: {X_val.shape}, y_val: {y_val.shape}")

    # Compare attention mechanisms
    print("\nComparing attention mechanisms...")
    dot_model, add_model, dot_history, add_history = compare_attention_mechanisms(
        X_train, y_train, X_val, y_val
    )

    # Visualize attention weights
    print("\nVisualizing attention weights for dot-product attention:")
    visualize_attention(dot_model, X_val[:5], y_val[:5])

    print("\nVisualizing attention weights for additive attention:")
    visualize_attention(add_model, X_val[:5], y_val[:5])

    print("\nExercise Tasks:")
    print("1. Analyze the training curves. Which attention mechanism performs better?")
    print(
        "2. Examine the attention visualizations. Can you identify where the models focus?"
    )
    print(
        "3. Modify the code to use residual connections with attention. Does performance improve?"
    )
    print(
        "4. Experiment with different temperature values for dot-product attention. How does it affect the results?"
    )
    print(
        "5. Implement multi-head attention and compare its performance to single-head attention."
    )


if __name__ == "__main__":
    main()
