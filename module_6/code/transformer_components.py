"""
Transformer Components Implementation

This module implements the core building blocks of the Transformer architecture:
1. Positional Encoding
2. Multi-Head Attention
3. Position-wise Feed-Forward Networks
4. Layer Normalization with Residual Connections

These components can be assembled to create complete Transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in "Attention Is All You Need".

    Adds positional information to the input embeddings since the Transformer
    has no recurrence or convolution to capture sequence order.
    """

    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        """
        Initialize the positional encoding.

        Args:
            d_model: The embedding dimension
            max_seq_length: Maximum sequence length to pre-compute
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, seq_length, d_model)

        Returns:
            Output with positional encoding added, same shape as input
        """
        # Add positional encoding (sliced to match sequence length)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def visualize_positional_encoding(d_model=64, max_seq_length=100):
    """
    Visualize the positional encoding patterns.

    Args:
        d_model: Embedding dimension
        max_seq_length: Number of positions to visualize
    """
    # Create positional encoding
    pe = PositionalEncoding(d_model, max_seq_length)
    sample_input = torch.zeros(1, max_seq_length, d_model)
    pos_encoding = pe(sample_input)[0].detach().cpu().numpy()

    # Plot heatmap of positional encodings
    plt.figure(figsize=(10, 8))
    plt.imshow(pos_encoding, cmap="viridis", aspect="auto")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position in Sequence")
    plt.title("Positional Encoding Patterns")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.show()

    # Plot specific dimensions
    plt.figure(figsize=(10, 6))
    dim_indices = [0, 1, 4, 8, 16, 32]
    for i, dim in enumerate(dim_indices):
        if dim < d_model:
            plt.plot(pos_encoding[:, dim], label=f"Dimension {dim}")

    plt.xlabel("Position in Sequence")
    plt.ylabel("Encoding Value")
    plt.title("Positional Encoding Values for Different Dimensions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class ScaledDotProductAttention(nn.Module):
    """
    Implements scaled dot-product attention as described in "Attention Is All You Need".

    Computes attention weights using query, key, value tensors with scaling to stabilize gradients.
    """

    def __init__(self, dropout=0.1):
        """
        Initialize scaled dot-product attention.

        Args:
            dropout: Dropout probability for attention weights
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_length, d_k)
            key: Key tensor of shape (batch_size, num_heads, seq_length, d_k)
            value: Value tensor of shape (batch_size, num_heads, seq_length, d_v)
            mask: Optional mask of shape (batch_size, 1, 1, seq_length) or (batch_size, 1, seq_length, seq_length)

        Returns:
            output: Tensor of shape (batch_size, num_heads, seq_length, d_v)
            attention_weights: Attention weight matrix
        """
        d_k = query.size(-1)

        # Compute attention scores: (batch_size, num_heads, seq_length, seq_length)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention as described in "Attention Is All You Need".

    Allows the model to jointly attend to information from different representation subspaces.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model's embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of key/query
        self.d_v = d_model // num_heads  # Dimension of value

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, depth).

        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        """
        Combine the heads back into original shape.

        Args:
            x: Tensor of shape (batch_size, num_heads, seq_length, d_k)

        Returns:
            Tensor of shape (batch_size, seq_length, d_model)
        """
        batch_size, _, seq_length, _ = x.size()
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.d_model)

    def forward(self, query, key, value, mask=None, residual=None):
        """
        Compute multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_length, d_model)
            key: Key tensor of shape (batch_size, seq_length, d_model)
            value: Value tensor of shape (batch_size, seq_length, d_model)
            mask: Optional mask of shape (batch_size, 1, seq_length) or (batch_size, seq_length, seq_length)
            residual: Optional residual tensor, if None, query is used

        Returns:
            output: Tensor of shape (batch_size, seq_length, d_model)
            attention_weights: Attention weight matrices for each head
        """
        if residual is None:
            residual = query

        batch_size, seq_length, _ = query.size()

        # Linear projections
        q = self.W_q(query)  # (batch_size, seq_length, d_model)
        k = self.W_k(key)  # (batch_size, seq_length, d_model)
        v = self.W_v(value)  # (batch_size, seq_length, d_model)

        # Split heads
        q = self.split_heads(q)  # (batch_size, num_heads, seq_length, d_k)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_length, d_k)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_length, d_v)

        # Adjust mask for multi-head attention if needed
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_length, seq_length)

        # Scaled dot-product attention
        output, attention_weights = self.attention(
            q, k, v, mask
        )  # (batch_size, num_heads, seq_length, d_v)

        # Combine heads
        output = self.combine_heads(output)  # (batch_size, seq_length, d_model)

        # Final linear projection
        output = self.W_o(output)  # (batch_size, seq_length, d_model)

        # Apply dropout and residual connection
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network of the Transformer.

    Consists of two linear transformations with a ReLU activation in between.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize position-wise feed-forward network.

        Args:
            d_model: Model's embedding dimension
            d_ff: Hidden layer dimension
            dropout: Dropout probability
        """
        super(PositionWiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Apply position-wise feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_length, d_model)
        """
        residual = x

        # Apply feed-forward network
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        # Apply dropout and residual connection
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x


class EncoderLayer(nn.Module):
    """
    Implements a single layer of the Transformer encoder.

    Consists of multi-head attention and position-wise feed-forward networks
    with residual connections and layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize encoder layer.

        Args:
            d_model: Model's embedding dimension
            num_heads: Number of attention heads
            d_ff: Hidden layer dimension in feed-forward network
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        """
        Apply encoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            mask: Optional mask for padding

        Returns:
            output: Tensor of shape (batch_size, seq_length, d_model)
            attention_weights: Self-attention weights
        """
        # Self-attention with residual connection
        attention_output, attention_weights = self.self_attention(x, x, x, mask)

        # Feed-forward network
        output = self.feed_forward(attention_output)

        return output, attention_weights


class DecoderLayer(nn.Module):
    """
    Implements a single layer of the Transformer decoder.

    Consists of masked multi-head attention, multi-head attention over encoder output,
    and position-wise feed-forward networks, with residual connections and layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize decoder layer.

        Args:
            d_model: Model's embedding dimension
            num_heads: Number of attention heads
            d_ff: Hidden layer dimension in feed-forward network
            dropout: Dropout probability
        """
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        Apply decoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            encoder_output: Output from encoder of shape (batch_size, seq_length_src, d_model)
            self_attn_mask: Mask for self-attention (prevents attending to future tokens)
            cross_attn_mask: Mask for cross-attention (usually for padding)

        Returns:
            output: Tensor of shape (batch_size, seq_length, d_model)
            self_attention_weights: Self-attention weights
            cross_attention_weights: Cross-attention weights over encoder output
        """
        # Self-attention with residual connection
        self_attention_output, self_attention_weights = self.self_attention(
            x, x, x, self_attn_mask
        )

        # Cross-attention with encoder output
        cross_attention_output, cross_attention_weights = self.cross_attention(
            self_attention_output, encoder_output, encoder_output, cross_attn_mask
        )

        # Feed-forward network
        output = self.feed_forward(cross_attention_output)

        return output, self_attention_weights, cross_attention_weights


def create_padding_mask(seq, pad_idx=0):
    """
    Create mask for padding tokens.

    Args:
        seq: Input sequence of shape (batch_size, seq_length)
        pad_idx: Index used for padding

    Returns:
        mask: Mask of shape (batch_size, 1, 1, seq_length)
              with 0s for padding tokens and 1s elsewhere
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(seq_length):
    """
    Create mask to prevent attending to future tokens.

    Args:
        seq_length: Length of the sequence

    Returns:
        mask: Upper triangular mask of shape (1, 1, seq_length, seq_length)
              with 0s in the upper triangle and 1s elsewhere
    """
    mask = (
        torch.triu(torch.ones((seq_length, seq_length)), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(1)
    )
    return mask == 0


def demo_components():
    """
    Demonstrate the usage of Transformer components.
    """
    # Parameters
    batch_size = 2
    seq_length = 10
    d_model = 64
    num_heads = 8
    d_ff = 256

    # Sample input
    x = torch.randn(batch_size, seq_length, d_model)

    # Positional encoding
    print("Testing Positional Encoding...")
    pos_encoding = PositionalEncoding(d_model)
    pos_encoded_x = pos_encoding(x)
    print(f"Input shape: {x.shape}")
    print(f"After positional encoding: {pos_encoded_x.shape}")
    print(f"Values affected? {(x != pos_encoded_x).any().item()}")

    # Multi-head attention
    print("\nTesting Multi-Head Attention...")
    mha = MultiHeadAttention(d_model, num_heads)
    mha_output, attention_weights = mha(x, x, x)
    print(f"Multi-head attention output shape: {mha_output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Position-wise feed-forward network
    print("\nTesting Position-Wise Feed-Forward Network...")
    ffn = PositionWiseFeedForward(d_model, d_ff)
    ffn_output = ffn(x)
    print(f"Feed-forward output shape: {ffn_output.shape}")

    # Encoder layer
    print("\nTesting Encoder Layer...")
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
    encoder_output, attn_weights = encoder_layer(x)
    print(f"Encoder layer output shape: {encoder_output.shape}")

    # Decoder layer with encoder output
    print("\nTesting Decoder Layer...")
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
    decoder_output, self_attn, cross_attn = decoder_layer(x, encoder_output)
    print(f"Decoder layer output shape: {decoder_output.shape}")
    print(f"Self-attention weights shape: {self_attn.shape}")
    print(f"Cross-attention weights shape: {cross_attn.shape}")

    # Masking
    print("\nTesting Masks...")
    seq = torch.randint(0, 10, (batch_size, seq_length))
    seq[:, -2:] = 0  # Add some padding
    padding_mask = create_padding_mask(seq)
    look_ahead_mask = create_look_ahead_mask(seq_length)
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Look-ahead mask shape: {look_ahead_mask.shape}")

    print("\nTransformer components tested successfully!")


def haiku_example():
    """
    Generate a haiku about transformers.
    """
    print("Transformer Haiku:")
    print("Layers of meaning,")
    print("Self-attention connects all,")
    print("Knowledge transforms us.")


if __name__ == "__main__":
    print("Transformer Components Module")
    print("=" * 40)

    # Demonstrate the components
    demo_components()

    # Visualize positional encoding
    visualize_positional_encoding()

    # Share a haiku
    haiku_example()
