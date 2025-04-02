"""
Exercise 5.2: Implementing Multi-Head Attention

In this exercise, you will implement the multi-head attention mechanism,
a key component of Transformer models. Multi-head attention allows the model
to jointly attend to information from different representation subspaces,
enabling it to capture complex patterns and relationships in the data.

Learning Objectives:
1. Understand the structure and purpose of multi-head attention
2. Implement multi-head attention from scratch
3. Visualize attention patterns across different heads
4. Compare multi-head attention with single-head attention
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    Attention(Q, K, V) = softmax(Q·K^T/sqrt(d_k))·V
    """

    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Tensor of shape [batch_size, num_heads, query_len, d_k]
            key: Tensor of shape [batch_size, num_heads, key_len, d_k]
            value: Tensor of shape [batch_size, num_heads, value_len, d_v]
            mask: Optional mask tensor of shape [batch_size, query_len, key_len]

        Returns:
            context: Tensor of shape [batch_size, num_heads, query_len, d_v]
            attention: Tensor of shape [batch_size, num_heads, query_len, key_len]
        """
        # Get dimensions
        batch_size, num_heads, query_len, d_k = query.size()
        key_len = key.size(2)

        # Calculate attention scores
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        )  # [batch_size, num_heads, query_len, key_len]

        # Apply scaling
        if self.scale is not None:
            scores = scores / self.scale
        else:
            scores = scores / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting across num_heads dimension
            mask = mask.unsqueeze(1)  # [batch_size, 1, query_len, key_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        context = torch.matmul(
            attention, value
        )  # [batch_size, num_heads, query_len, d_v]

        return context, attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Module.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
    where head_i = Attention(Q·W_q_i, K·W_k_i, V·W_v_i)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of each head

        # Linear projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention()

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Tensor of shape [batch_size, query_len, d_model]
            key: Tensor of shape [batch_size, key_len, d_model]
            value: Tensor of shape [batch_size, value_len, d_model]
            mask: Optional mask tensor of shape [batch_size, query_len, key_len]

        Returns:
            output: Tensor of shape [batch_size, query_len, d_model]
            attention: Tensor of shape [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.size(0)

        # Linear projections
        query = self.query_proj(query)  # [batch_size, query_len, d_model]
        key = self.key_proj(key)  # [batch_size, key_len, d_model]
        value = self.value_proj(value)  # [batch_size, value_len, d_model]

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [batch_size, num_heads, query_len, d_k]
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [batch_size, num_heads, key_len, d_k]
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [batch_size, num_heads, value_len, d_k]

        # Apply scaled dot-product attention
        context, attention = self.attention(
            query, key, value, mask
        )  # [batch_size, num_heads, query_len, d_k]

        # Concatenate heads and put through final linear layer
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )  # [batch_size, query_len, d_model]
        output = self.output_proj(context)  # [batch_size, query_len, d_model]

        # Apply dropout
        output = self.dropout(output)

        return output, attention


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            output: Tensor of shape [batch_size, seq_len, d_model]
        """
        output = self.w_2(torch.relu(self.w_1(x)))
        output = self.dropout(output)
        return output


class MultiHeadAttentionLayer(nn.Module):
    """
    A single layer of the Transformer encoder.

    Layer consists of:
    1. Multi-head attention
    2. Add & Norm
    3. Position-wise feed-forward
    4. Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor of shape [batch_size, seq_len, seq_len]

        Returns:
            output: Tensor of shape [batch_size, seq_len, d_model]
            attention: Tensor of shape [batch_size, num_heads, seq_len, seq_len]
        """
        # Self-attention
        attn_output, attention = self.self_attention(x, x, x, mask)

        # Add & Norm
        x = self.norm1(x + self.dropout(attn_output))

        # Position-wise feed-forward
        ff_output = self.feed_forward(x)

        # Add & Norm
        output = self.norm2(x + self.dropout(ff_output))

        return output, attention


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the Transformer.

    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, d_model, max_seq_len=1000, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer (not a parameter but should be saved and loaded with the model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            output: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SequenceClassifier(nn.Module):
    """
    Sequence classifier using multi-head attention.
    """

    def __init__(
        self,
        input_dim,
        d_model,
        num_heads,
        d_ff,
        num_classes,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output layer
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len, seq_len]

        Returns:
            output: Class scores of shape [batch_size, num_classes]
            attention_weights: List of attention weights from each layer,
                              each of shape [batch_size, num_heads, seq_len, seq_len]
        """
        # Input embedding
        x = self.embedding(x)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]

        # Store attention weights from each layer
        attention_weights = []

        # Apply transformer layers
        for layer in self.transformer_layers:
            x, attention = layer(x, mask)  # [batch_size, seq_len, d_model]
            attention_weights.append(attention)

        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, d_model]

        # Output layer
        output = self.fc(x)  # [batch_size, num_classes]

        return output, attention_weights


def generate_sequence_data(
    num_samples=1000, seq_len=20, num_classes=3, num_features=10
):
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

    # Add class-specific patterns in random positions
    for i in range(num_samples):
        class_id = y[i]

        # Add a primary pattern
        pattern1 = np.random.randn(3, num_features) + (class_id * 2 - 1) * 3
        pos1 = np.random.randint(0, seq_len - 3 + 1)
        X[i, pos1 : pos1 + 3, :] = pattern1

        # Add a secondary pattern
        pattern2 = np.random.randn(2, num_features) + (class_id * 2 - 1) * 2
        pos2 = np.random.randint(0, seq_len - 2 + 1)
        while pos2 >= pos1 and pos2 < pos1 + 3:  # Ensure no overlap
            pos2 = np.random.randint(0, seq_len - 2 + 1)
        X[i, pos2 : pos2 + 2, :] = pattern2

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train_model(
    model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, lr=0.0005
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )

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

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

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


def visualize_attention(model, X, y, num_samples=2, layer_idx=0):
    """
    Visualize attention weights for different heads.

    Args:
        model: The trained model
        X: Input sequences
        y: Class labels
        num_samples: Number of samples to visualize
        layer_idx: Index of the transformer layer to visualize
    """
    model.eval()
    X_device = X.to(device)

    with torch.no_grad():
        # Forward pass to get attention weights
        _, attention_weights = model(X_device)

    # Get attention weights from the specified layer
    attention = attention_weights[
        layer_idx
    ]  # [batch_size, num_heads, seq_len, seq_len]

    # Convert attention weights to numpy for visualization
    attention = attention.cpu().numpy()

    # Get number of heads
    num_heads = attention.shape[1]

    # Visualize attention weights for some samples
    for i in range(min(num_samples, len(X))):
        plt.figure(figsize=(15, 4 * num_heads))

        for h in range(num_heads):
            plt.subplot(num_heads, 1, h + 1)
            plt.imshow(attention[i, h], cmap="viridis")
            plt.colorbar()
            plt.title(f"Sample {i}, Class {y[i].item()}, Head {h+1}")
            plt.xlabel("Key position")
            plt.ylabel("Query position")

        plt.tight_layout()
        plt.show()


def compare_num_heads(X_train, y_train, X_val, y_val):
    """
    Compare models with different numbers of attention heads.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        models: Dictionary of trained models with different numbers of heads
        histories: Dictionary of training histories
    """
    # Get dimensions from data
    _, seq_len, input_dim = X_train.shape
    num_classes = len(torch.unique(y_train))

    # Model parameters
    d_model = 128
    d_ff = 256
    num_layers = 1
    dropout = 0.1

    # Different numbers of heads to compare
    head_configs = [1, 2, 4, 8]

    # Dictionaries to store models and histories
    models = {}
    histories = {}

    # Train models with different numbers of heads
    for num_heads in head_configs:
        print(f"\nTraining model with {num_heads} attention heads:")

        # Create model
        model = SequenceClassifier(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        # Train model
        model, history = train_model(model, X_train, y_train, X_val, y_val)

        # Store model and history
        models[num_heads] = model
        histories[num_heads] = history

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    for num_heads in head_configs:
        plt.plot(histories[num_heads]["train_acc"], label=f"{num_heads}-head train")
        plt.plot(histories[num_heads]["val_acc"], label=f"{num_heads}-head val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()

    plt.subplot(1, 2, 2)
    for num_heads in head_configs:
        plt.plot(histories[num_heads]["train_loss"], label=f"{num_heads}-head train")
        plt.plot(histories[num_heads]["val_loss"], label=f"{num_heads}-head val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return models, histories


def main():
    """Main function to run the exercise."""
    print("Exercise 5.2: Implementing Multi-Head Attention\n")

    # Parameters
    num_samples = 1000
    seq_len = 20
    num_features = 10
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

    # Compare models with different numbers of heads
    print("\nComparing models with different numbers of attention heads...")
    models, histories = compare_num_heads(X_train, y_train, X_val, y_val)

    # Visualize attention weights for the 4-head model
    print("\nVisualizing attention weights for model with 4 heads:")
    visualize_attention(models[4], X_val[:3], y_val[:3])

    print("\nExercise Tasks:")
    print(
        "1. Analyze the results. How does the number of attention heads affect model performance?"
    )
    print(
        "2. Examine the attention visualizations. Do different heads focus on different patterns?"
    )
    print(
        "3. Modify the code to implement the transformer with multiple layers. How does depth affect performance?"
    )
    print(
        "4. Experiment with adding residual connections and layer normalization. How do they affect training stability?"
    )
    print(
        "5. Try different positional encoding schemes (learned vs. fixed). Which works better for this task?"
    )


if __name__ == "__main__":
    main()
