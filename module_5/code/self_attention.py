"""
Self-Attention Implementation

This module implements self-attention mechanisms as used in the Transformer architecture.
It includes implementations of:
1. Self-Attention
2. Multi-Head Self-Attention
3. Visualization of attention patterns

The module demonstrates how self-attention allows elements in a sequence to attend
to all other elements, regardless of their distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class SelfAttention(nn.Module):
    """
    Self-Attention layer implementation.

    This is a simplified version of the attention mechanism used in the Transformer
    architecture. It allows elements in a sequence to attend to all other elements,
    capturing dependencies regardless of distance.
    """

    def __init__(self, d_model):
        """
        Initialize the self-attention layer.

        Args:
            d_model: Dimension of the input features
        """
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.scaling_factor = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

        # Linear projections for query, key, value
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Forward pass of the self-attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
        """
        # Project inputs to queries, keys, values
        Q = self.query_projection(x)  # (batch_size, seq_len, d_model)
        K = self.key_projection(x)  # (batch_size, seq_len, d_model)
        V = self.value_projection(x)  # (batch_size, seq_len, d_model)

        # Calculate attention scores
        attention_scores = torch.bmm(
            Q, K.transpose(1, 2)
        )  # (batch_size, seq_len, seq_len)

        # Apply scaling
        attention_scores = attention_scores / self.scaling_factor

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(
            attention_scores, dim=2
        )  # (batch_size, seq_len, seq_len)

        # Apply attention weights to values
        output = torch.bmm(attention_weights, V)  # (batch_size, seq_len, d_model)

        # Project output
        output = self.output_projection(output)  # (batch_size, seq_len, d_model)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation.

    Multi-head attention consists of multiple self-attention heads operating in parallel,
    allowing the model to jointly attend to information from different representation subspaces.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize the multi-head attention layer.

        Args:
            d_model: Dimension of the input features
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Linear projections for query, key, value
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, head_dim).

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def combine_heads(self, x):
        """
        Combine the heads back to the original shape.

        Args:
            x: Tensor of shape (batch_size, num_heads, seq_len, head_dim)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, x, mask=None):
        """
        Forward pass of the multi-head attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # Project inputs to queries, keys, values
        Q = self.query_projection(x)  # (batch_size, seq_len, d_model)
        K = self.key_projection(x)  # (batch_size, seq_len, d_model)
        V = self.value_projection(x)  # (batch_size, seq_len, d_model)

        # Split heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, head_dim)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len, head_dim)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len, head_dim)

        # Calculate attention scores for each head
        attention_scores = torch.matmul(
            Q, K.transpose(2, 3)
        )  # (batch_size, num_heads, seq_len, seq_len)

        # Apply scaling
        attention_scores = attention_scores / (self.head_dim**0.5)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention_scores shape
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(
            attention_scores, dim=3
        )  # (batch_size, num_heads, seq_len, seq_len)

        # Apply attention weights to values
        output = torch.matmul(
            attention_weights, V
        )  # (batch_size, num_heads, seq_len, head_dim)

        # Combine heads
        output = self.combine_heads(output)  # (batch_size, seq_len, d_model)

        # Project output
        output = self.output_projection(output)  # (batch_size, seq_len, d_model)

        return output, attention_weights


def visualize_attention(
    attention_weights, tokens, head=0, title="Attention Visualization"
):
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len) or (batch_size, seq_len, seq_len)
        tokens: List of tokens that correspond to the sequence
        head: Head index to visualize (only for multi-head attention)
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))

    # Extract attention weights for visualization
    if attention_weights.dim() > 3:
        # For multi-head attention, extract specific head
        attention_map = attention_weights[0, head].detach().cpu().numpy()
    else:
        # For single-head attention
        attention_map = attention_weights[0].detach().cpu().numpy()

    # Create heatmap
    sns.heatmap(attention_map, xticklabels=tokens, yticklabels=tokens, cmap="viridis")

    plt.title(f"{title} (Head {head})" if attention_weights.dim() > 3 else title)
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.tight_layout()
    plt.show()


def visualize_all_heads(
    attention_weights, tokens, title="Multi-Head Attention Visualization"
):
    """
    Visualize all attention heads in a multi-head attention mechanism.

    Args:
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        tokens: List of tokens that correspond to the sequence
        title: Title for the plot
    """
    _, num_heads, _, _ = attention_weights.size()

    # Calculate grid dimensions
    n_cols = min(4, num_heads)
    n_rows = (num_heads + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    for i in range(num_heads):
        if i < len(axs):
            attention_map = attention_weights[0, i].detach().cpu().numpy()
            sns.heatmap(
                attention_map,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="viridis",
                ax=axs[i],
            )
            axs[i].set_title(f"Head {i+1}")
            axs[i].set_xlabel("Tokens")
            axs[i].set_ylabel("Tokens")

    # Hide empty subplots
    for i in range(num_heads, len(axs)):
        axs[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def demo_self_attention():
    """
    Demonstrate self-attention on a simple example.
    """
    # Setup dimensions
    batch_size = 1
    seq_len = 5
    d_model = 8

    # Create random inputs
    x = torch.randn(batch_size, seq_len, d_model)

    # Create attention module
    attention = SelfAttention(d_model)

    # Forward pass
    output, attention_weights = attention(x)

    # Print results
    print("Self-Attention Demo:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Visualize attention weights
    tokens = ["Token 1", "Token 2", "Token 3", "Token 4", "Token 5"]
    visualize_attention(attention_weights, tokens)


def demo_multi_head_attention():
    """
    Demonstrate multi-head attention on a simple example.
    """
    # Setup dimensions
    batch_size = 1
    seq_len = 5
    d_model = 64
    num_heads = 8

    # Create random inputs
    x = torch.randn(batch_size, seq_len, d_model)

    # Create attention module
    attention = MultiHeadAttention(d_model, num_heads)

    # Forward pass
    output, attention_weights = attention(x)

    # Print results
    print("\nMulti-Head Attention Demo:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Visualize attention weights for a single head
    tokens = ["Token 1", "Token 2", "Token 3", "Token 4", "Token 5"]
    visualize_attention(attention_weights, tokens, head=0)

    # Visualize all heads
    visualize_all_heads(attention_weights, tokens)


def haiku_example():
    """
    Demonstrate self-attention on a haiku example.
    """
    # Define a simple haiku
    haiku = "old pond / frog jumps in / water sound"
    tokens = haiku.split()

    # Create embeddings for each token (in a real scenario these would be learned)
    batch_size = 1
    seq_len = len(tokens)
    d_model = 32

    # Create random embeddings
    embeddings = torch.randn(batch_size, seq_len, d_model)

    # Create self-attention module
    attention = SelfAttention(d_model)

    # Forward pass
    output, attention_weights = attention(embeddings)

    # Visualize attention weights
    print("\nHaiku Self-Attention Example:")
    print(f"Haiku: {haiku}")
    print(f"Attention weights shape: {attention_weights.shape}")

    visualize_attention(attention_weights, tokens, title="Haiku Self-Attention")

    # Now try with multi-head attention
    multi_head = MultiHeadAttention(d_model, num_heads=4)
    output_mh, attention_weights_mh = multi_head(embeddings)

    print("\nHaiku Multi-Head Attention Example:")
    print(f"Attention weights shape: {attention_weights_mh.shape}")

    visualize_all_heads(
        attention_weights_mh, tokens, title="Haiku Multi-Head Attention"
    )

    print(
        "\nThis visualization shows how each token in the haiku attends to every other token."
    )
    print("In a trained model, related words would have stronger attention patterns.")
    print(
        "Multi-head attention allows the model to focus on different aspects of the relationships."
    )


if __name__ == "__main__":
    print("Self-Attention Mechanisms Demo")
    print("=" * 40)

    demo_self_attention()
    demo_multi_head_attention()
    haiku_example()

    print("\nSelf-attention haiku:")
    print("Words speak to each word,")
    print("Patterns emerge from chaos,")
    print("Context flows like streams.")
