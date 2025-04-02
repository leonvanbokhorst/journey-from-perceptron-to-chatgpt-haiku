"""
Basic Attention Mechanisms in PyTorch

This module demonstrates different types of attention mechanisms:
1. Additive (Bahdanau) Attention
2. Multiplicative (Dot-Product) Attention
3. Scaled Dot-Product Attention

Each implementation includes a forward pass and visualization capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.

    Uses a feedforward neural network to compute alignment scores.
    """

    def __init__(self, query_dim, key_dim, attention_dim):
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(key_dim, attention_dim, bias=False)
        self.energy = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, query, keys, values, mask=None):
        """
        Forward pass for additive attention.

        Args:
            query: Tensor of shape (batch_size, query_dim)
            keys: Tensor of shape (batch_size, seq_len, key_dim)
            values: Tensor of shape (batch_size, seq_len, value_dim)
            mask: Optional mask of shape (batch_size, seq_len)

        Returns:
            context: Tensor of shape (batch_size, value_dim)
            attention_weights: Tensor of shape (batch_size, seq_len)
        """
        # Reshape query to (batch_size, 1, attention_dim)
        query = self.query_proj(query).unsqueeze(1)

        # Reshape keys to (batch_size, seq_len, attention_dim)
        keys = self.key_proj(keys)

        # Calculate energies (batch_size, seq_len, 1)
        energies = self.energy(torch.tanh(query + keys))
        energies = energies.squeeze(2)

        # Apply mask if provided
        if mask is not None:
            energies = energies.masked_fill(mask == 0, -1e9)

        # Calculate attention weights (batch_size, seq_len)
        attention_weights = F.softmax(energies, dim=1)

        # Calculate context vector (batch_size, value_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return context, attention_weights


class MultiplicativeAttention(nn.Module):
    """
    Multiplicative (Luong) attention mechanism.

    Uses dot product between query and keys to compute alignment scores.
    """

    def __init__(self, query_dim, key_dim):
        super(MultiplicativeAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, key_dim, bias=False)

    def forward(self, query, keys, values, mask=None):
        """
        Forward pass for multiplicative attention.

        Args:
            query: Tensor of shape (batch_size, query_dim)
            keys: Tensor of shape (batch_size, seq_len, key_dim)
            values: Tensor of shape (batch_size, seq_len, value_dim)
            mask: Optional mask of shape (batch_size, seq_len)

        Returns:
            context: Tensor of shape (batch_size, value_dim)
            attention_weights: Tensor of shape (batch_size, seq_len)
        """
        # Project query to key dimension (batch_size, key_dim)
        query = self.query_proj(query)

        # Calculate energies (batch_size, seq_len)
        energies = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)

        # Apply mask if provided
        if mask is not None:
            energies = energies.masked_fill(mask == 0, -1e9)

        # Calculate attention weights (batch_size, seq_len)
        attention_weights = F.softmax(energies, dim=1)

        # Calculate context vector (batch_size, value_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return context, attention_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as proposed in "Attention Is All You Need".

    Scales the dot product by sqrt(d_k) to stabilize gradients.
    """

    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.scaling_factor = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

    def forward(self, queries, keys, values, mask=None):
        """
        Forward pass for scaled dot-product attention.

        Args:
            queries: Tensor of shape (batch_size, num_queries, d_model)
            keys: Tensor of shape (batch_size, seq_len, d_model)
            values: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask of shape (batch_size, num_queries, seq_len)

        Returns:
            outputs: Tensor of shape (batch_size, num_queries, d_model)
            attention_weights: Tensor of shape (batch_size, num_queries, seq_len)
        """
        # Calculate attention scores (batch_size, num_queries, seq_len)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))

        # Scale attention scores
        attention_scores = attention_scores / self.scaling_factor

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Calculate attention weights (batch_size, num_queries, seq_len)
        attention_weights = F.softmax(attention_scores, dim=2)

        # Calculate outputs (batch_size, num_queries, d_model)
        outputs = torch.bmm(attention_weights, values)

        return outputs, attention_weights


def visualize_attention(attention_weights, input_tokens, output_tokens=None):
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights: Tensor of shape (seq_len, seq_len) or (batch_size, seq_len)
        input_tokens: List of input tokens
        output_tokens: Optional list of output tokens
    """
    plt.figure(figsize=(10, 8))

    if attention_weights.dim() > 2:
        attention_weights = attention_weights.squeeze(0)  # Remove batch dimension

    if output_tokens is not None:
        # For encoder-decoder attention
        sns.heatmap(
            attention_weights.detach().cpu().numpy(),
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            cmap="viridis",
        )
        plt.xlabel("Input tokens")
        plt.ylabel("Output tokens")
    else:
        # For self-attention
        sns.heatmap(
            attention_weights.detach().cpu().numpy(),
            xticklabels=input_tokens,
            yticklabels=input_tokens if len(attention_weights.shape) > 1 else None,
            cmap="viridis",
        )
        plt.xlabel("Tokens")
        plt.ylabel(
            "Attention weights" if len(attention_weights.shape) == 1 else "Tokens"
        )

    plt.title("Attention Visualization")
    plt.tight_layout()
    plt.show()


def demo_additive_attention():
    """
    Demonstrate additive attention with a simple example.
    """
    # Setup dimensions
    batch_size = 1
    seq_len = 5
    query_dim = 8
    key_dim = 8
    value_dim = 8
    attention_dim = 10

    # Create random inputs
    query = torch.randn(batch_size, query_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, value_dim)

    # Create attention module
    attention = AdditiveAttention(query_dim, key_dim, attention_dim)

    # Forward pass
    context, attention_weights = attention(query, keys, values)

    # Print results
    print("Additive Attention:")
    print(f"Context vector shape: {context.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights: {attention_weights}")

    # Visualize attention weights
    input_tokens = ["Token 1", "Token 2", "Token 3", "Token 4", "Token 5"]
    visualize_attention(attention_weights, input_tokens)


def demo_multiplicative_attention():
    """
    Demonstrate multiplicative attention with a simple example.
    """
    # Setup dimensions
    batch_size = 1
    seq_len = 5
    query_dim = 8
    key_dim = 8
    value_dim = 8

    # Create random inputs
    query = torch.randn(batch_size, query_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, value_dim)

    # Create attention module
    attention = MultiplicativeAttention(query_dim, key_dim)

    # Forward pass
    context, attention_weights = attention(query, keys, values)

    # Print results
    print("\nMultiplicative Attention:")
    print(f"Context vector shape: {context.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights: {attention_weights}")

    # Visualize attention weights
    input_tokens = ["Token 1", "Token 2", "Token 3", "Token 4", "Token 5"]
    visualize_attention(attention_weights, input_tokens)


def demo_scaled_dot_product_attention():
    """
    Demonstrate scaled dot-product attention with a simple example.
    """
    # Setup dimensions
    batch_size = 1
    seq_len = 5
    num_queries = 3
    d_model = 8

    # Create random inputs
    queries = torch.randn(batch_size, num_queries, d_model)
    keys = torch.randn(batch_size, seq_len, d_model)
    values = torch.randn(batch_size, seq_len, d_model)

    # Create attention module
    attention = ScaledDotProductAttention(d_model)

    # Forward pass
    outputs, attention_weights = attention(queries, keys, values)

    # Print results
    print("\nScaled Dot-Product Attention:")
    print(f"Outputs shape: {outputs.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Visualize attention weights for the first query
    input_tokens = ["Token 1", "Token 2", "Token 3", "Token 4", "Token 5"]
    output_tokens = ["Query 1", "Query 2", "Query 3"]
    visualize_attention(attention_weights, input_tokens, output_tokens)


def haiku_example():
    """
    Demonstrate attention on a haiku example, showing how words attend to each other.
    """
    # Define a simple haiku and tokenize it
    haiku = "old pond / frog jumps in / water sound"
    tokens = haiku.split()

    # Create embeddings for each token (simplified as random vectors)
    batch_size = 1
    seq_len = len(tokens)
    d_model = 8

    # Create embeddings
    embeddings = torch.randn(batch_size, seq_len, d_model)

    # Use self-attention to see how words relate to each other
    attention = ScaledDotProductAttention(d_model)
    outputs, attention_weights = attention(embeddings, embeddings, embeddings)

    # Print and visualize
    print("\nHaiku Self-Attention Example:")
    print(f"Haiku: {haiku}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Visualize the self-attention matrix
    visualize_attention(attention_weights[0], tokens, tokens)

    print(
        "\nThis visualization shows how each word in the haiku attends to every other word."
    )
    print("Brighter colors indicate stronger attention weights.")
    print("In a trained model, this would reveal semantic relationships between words.")


if __name__ == "__main__":
    print("Basic Attention Mechanisms Demo")
    print("=" * 40)

    demo_additive_attention()
    demo_multiplicative_attention()
    demo_scaled_dot_product_attention()
    haiku_example()

    print("\nAttention haiku:")
    print("Focus shifts gently,")
    print("Important words stand out bright,")
    print("Context emerges.")
