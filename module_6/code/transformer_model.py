"""
Complete Transformer Model Implementation

This module implements the full Transformer architecture as introduced in
"Attention Is All You Need" (Vaswoski et al., 2017).

The implementation includes:
1. Transformer Encoder
2. Transformer Decoder
3. Complete Transformer Model
4. Utilities for training and inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# Import components from transformer_components
from transformer_components import (
    PositionalEncoding,
    EncoderLayer,
    DecoderLayer,
    create_padding_mask,
    create_look_ahead_mask,
)


class TransformerEncoder(nn.Module):
    """
    Implements the Transformer encoder stack.

    Consists of N identical layers with multi-head attention and feed-forward networks,
    preceded by an embedding layer and positional encoding.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        dropout=0.1,
        max_seq_length=5000,
    ):
        """
        Initialize the Transformer encoder.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Model's embedding dimension
            num_heads: Number of attention heads
            d_ff: Hidden layer dimension in feed-forward network
            num_layers: Number of encoder layers
            dropout: Dropout probability
            max_seq_length: Maximum sequence length for positional encoding
        """
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Create a stack of encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        """
        Apply the Transformer encoder.

        Args:
            x: Input token indices of shape (batch_size, seq_length)
            mask: Optional mask for padding

        Returns:
            output: Encoded representation of shape (batch_size, seq_length, d_model)
            attention_weights: List of attention weights from each layer
        """
        # Convert to embeddings and scale
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Store attention weights from each layer
        attention_weights = []

        # Apply encoder layers
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, mask)
            attention_weights.append(attn_weights)

        return x, attention_weights


class TransformerDecoder(nn.Module):
    """
    Implements the Transformer decoder stack.

    Consists of N identical layers with masked multi-head attention, multi-head attention
    over encoder output, and feed-forward networks, preceded by an embedding layer and
    positional encoding.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        dropout=0.1,
        max_seq_length=5000,
    ):
        """
        Initialize the Transformer decoder.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Model's embedding dimension
            num_heads: Number of attention heads
            d_ff: Hidden layer dimension in feed-forward network
            num_layers: Number of decoder layers
            dropout: Dropout probability
            max_seq_length: Maximum sequence length for positional encoding
        """
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Create a stack of decoder layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        """
        Apply the Transformer decoder.

        Args:
            x: Input token indices of shape (batch_size, seq_length)
            encoder_output: Output from encoder of shape (batch_size, seq_length_src, d_model)
            look_ahead_mask: Mask to prevent attending to future tokens
            padding_mask: Mask for padding in encoder output

        Returns:
            output: Decoded representation of shape (batch_size, seq_length, d_model)
            self_attention_weights: List of self-attention weights from each layer
            cross_attention_weights: List of cross-attention weights from each layer
        """
        seq_length = x.size(1)

        # Convert to embeddings and scale
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Store attention weights from each layer
        self_attention_weights = []
        cross_attention_weights = []

        # Apply decoder layers
        for decoder_layer in self.decoder_layers:
            x, self_attn, cross_attn = decoder_layer(
                x, encoder_output, look_ahead_mask, padding_mask
            )

            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)

        return x, self_attention_weights, cross_attention_weights


class Transformer(nn.Module):
    """
    Implements the complete Transformer model.

    Combines the encoder and decoder stacks with a final linear layer to predict output tokens.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        dropout=0.1,
        max_seq_length=5000,
    ):
        """
        Initialize the Transformer model.

        Args:
            src_vocab_size: Size of the source vocabulary
            tgt_vocab_size: Size of the target vocabulary
            d_model: Model's embedding dimension
            num_heads: Number of attention heads
            d_ff: Hidden layer dimension in feed-forward network
            num_layers: Number of encoder/decoder layers
            dropout: Dropout probability
            max_seq_length: Maximum sequence length for positional encoding
        """
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_layers,
            dropout,
            max_seq_length,
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_layers,
            dropout,
            max_seq_length,
        )

        # Final linear layer to predict target tokens
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        """
        Apply the Transformer model.

        Args:
            src: Source token indices of shape (batch_size, src_seq_length)
            tgt: Target token indices of shape (batch_size, tgt_seq_length)
            src_mask: Mask for padding in source sequence
            tgt_mask: Mask for padding and to prevent attending to future tokens in target sequence
            src_padding_mask: Padding mask for source sequence
            tgt_padding_mask: Padding mask for target sequence

        Returns:
            output: Logits for target token prediction of shape (batch_size, tgt_seq_length, tgt_vocab_size)
            encoder_attention: Attention weights from the encoder
            decoder_self_attention: Self-attention weights from the decoder
            decoder_cross_attention: Cross-attention weights from the decoder
        """
        # Create masks if not provided
        if src_mask is None and src is not None:
            src_mask = create_padding_mask(src)

        if tgt_mask is None and tgt is not None:
            tgt_seq_length = tgt.size(1)
            tgt_mask = create_look_ahead_mask(tgt_seq_length)

            if tgt_padding_mask is not None:
                # Combine the look-ahead mask with the padding mask
                tgt_mask = tgt_mask & tgt_padding_mask

        # Encode source sequence
        encoder_output, encoder_attention = self.encoder(src, src_mask)

        # Decode target sequence with encoder output
        decoder_output, decoder_self_attention, decoder_cross_attention = self.decoder(
            tgt, encoder_output, tgt_mask, src_mask
        )

        # Apply final linear layer to predict target tokens
        output = self.final_layer(decoder_output)

        return (
            output,
            encoder_attention,
            decoder_self_attention,
            decoder_cross_attention,
        )

    def encode(self, src, src_mask=None):
        """
        Encode source sequence.

        Args:
            src: Source token indices of shape (batch_size, src_seq_length)
            src_mask: Mask for padding in source sequence

        Returns:
            encoder_output: Encoded representation
            encoder_attention: Attention weights from the encoder
        """
        if src_mask is None:
            src_mask = create_padding_mask(src)

        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """
        Decode target sequence with encoder output.

        Args:
            tgt: Target token indices of shape (batch_size, tgt_seq_length)
            encoder_output: Output from encoder
            tgt_mask: Mask for padding and to prevent attending to future tokens in target sequence
            src_mask: Mask for padding in source sequence

        Returns:
            output: Logits for target token prediction
            decoder_self_attention: Self-attention weights from the decoder
            decoder_cross_attention: Cross-attention weights from the decoder
        """
        if tgt_mask is None:
            tgt_seq_length = tgt.size(1)
            tgt_mask = create_look_ahead_mask(tgt_seq_length)

        decoder_output, decoder_self_attention, decoder_cross_attention = self.decoder(
            tgt, encoder_output, tgt_mask, src_mask
        )

        output = self.final_layer(decoder_output)

        return output, decoder_self_attention, decoder_cross_attention

    def generate(self, src, max_length=50, sos_idx=1, eos_idx=2, temperature=1.0):
        """
        Generate target sequence from source sequence.

        Args:
            src: Source token indices of shape (batch_size, src_seq_length)
            max_length: Maximum length of generated sequence
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            temperature: Sampling temperature (1.0 = greedy, >1.0 = more random)

        Returns:
            generated: Generated token indices
            attention_weights: Attention weights used during generation
        """
        batch_size = src.size(0)
        device = src.device

        # Encode source sequence
        encoder_output, _ = self.encode(src)

        # Initialize target sequence with SOS token
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

        # Store attention weights
        all_cross_attention = []

        # Generate tokens auto-regressively
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            # Decode current target sequence
            output, _, cross_attention = self.decode(tgt, encoder_output)

            # Get the last token prediction
            logits = output[:, -1, :] / temperature

            # Sample next token (or use argmax for greedy decoding)
            if temperature == 1.0:
                next_token = torch.argmax(logits, dim=-1)
            else:
                # Sample from softmax distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Store cross-attention for visualization
            all_cross_attention.append(cross_attention[-1][:, :, -1, :].detach())

            # Append next token to target sequence
            tgt = torch.cat([tgt, next_token.unsqueeze(-1)], dim=-1)

            # Check if generation is finished
            finished = finished | (next_token == eos_idx)
            if finished.all():
                break

        return tgt, all_cross_attention


def create_masks(src, tgt, src_pad_idx=0, tgt_pad_idx=0):
    """
    Create all necessary masks for training.

    Args:
        src: Source token indices of shape (batch_size, src_seq_length)
        tgt: Target token indices of shape (batch_size, tgt_seq_length)
        src_pad_idx: Padding token index for source
        tgt_pad_idx: Padding token index for target

    Returns:
        src_mask: Source padding mask
        tgt_mask: Target padding and look-ahead mask
    """
    # Source padding mask
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    # Target padding mask
    tgt_padding_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(2)

    # Target look-ahead mask
    tgt_seq_length = tgt.size(1)
    tgt_look_ahead_mask = torch.triu(
        torch.ones((tgt_seq_length, tgt_seq_length)), diagonal=1
    ).eq(0)
    tgt_look_ahead_mask = tgt_look_ahead_mask.unsqueeze(0).unsqueeze(1)

    # Combine padding and look-ahead masks
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask.to(tgt_padding_mask.device)

    return src_mask, tgt_mask


class LabelSmoothing(nn.Module):
    """
    Implements label smoothing as described in "Rethinking the Inception Architecture for Computer Vision".

    Smooths the target distribution to prevent the model from being too confident.
    """

    def __init__(self, eps=0.1, ignore_index=0):
        """
        Initialize label smoothing.

        Args:
            eps: Smoothing factor (0 = no smoothing, 1 = maximum smoothing)
            ignore_index: Index to ignore in loss calculation (e.g., padding)
        """
        super(LabelSmoothing, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        Apply label smoothing to cross-entropy loss.

        Args:
            output: Model output logits of shape (batch_size, seq_length, vocab_size)
            target: Target indices of shape (batch_size, seq_length)

        Returns:
            loss: Smoothed loss value
        """
        batch_size, seq_length, vocab_size = output.size()

        # Create smoothed target distribution
        smoothed_target = torch.zeros_like(output).fill_(self.eps / (vocab_size - 1))

        # Fill in the true class with 1 - eps
        smoothed_target.scatter_(2, target.unsqueeze(-1), 1 - self.eps)

        # Create mask for padding tokens
        mask = (target != self.ignore_index).float().unsqueeze(-1)

        # Calculate loss with smoothed targets
        log_probs = F.log_softmax(output, dim=-1)
        loss = -(smoothed_target * log_probs).sum(dim=-1) * mask.squeeze(-1)

        # Return mean loss over non-padding tokens
        return loss.sum() / mask.sum()


def subsequent_mask(size):
    """
    Create a mask to prevent attending to future tokens.

    Args:
        size: Sequence length

    Returns:
        mask: Lower triangular mask
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1) == 0
    return mask


def visualize_attention(
    attention_weights, src_tokens, tgt_tokens, layer_idx=0, head_idx=0
):
    """
    Visualize attention patterns in the Transformer.

    Args:
        attention_weights: Attention weights from the model
        src_tokens: Source tokens for labeling
        tgt_tokens: Target tokens for labeling
        layer_idx: Layer index to visualize
        head_idx: Head index to visualize
    """
    # Extract attention weights from the specified layer and head
    attn = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    cax = ax.matshow(attn, cmap="viridis")

    # Add colorbar
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90)
    ax.set_yticklabels(tgt_tokens)

    ax.set_xlabel("Source")
    ax.set_ylabel("Target")
    ax.set_title(f"Layer {layer_idx+1}, Head {head_idx+1} Attention")

    plt.tight_layout()
    plt.show()


def demo_transformer():
    """
    Demonstrate how to use the Transformer model.
    """
    # Define model parameters
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    dropout = 0.1

    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Generate sample data
    batch_size = 2
    src_seq_length = 10
    tgt_seq_length = 8

    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_length))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_length))

    # Create masks
    src_mask, tgt_mask = create_masks(src, tgt)

    # Forward pass
    print("Testing forward pass...")
    output, enc_attn, dec_self_attn, dec_cross_attn = model(
        src, tgt, src_mask, tgt_mask
    )

    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Encoder attention shape: {enc_attn[0].shape}")
    print(f"Decoder self-attention shape: {dec_self_attn[0].shape}")
    print(f"Decoder cross-attention shape: {dec_cross_attn[0].shape}")

    # Test generation
    print("\nTesting generation...")
    generated, gen_attn = model.generate(src, max_length=15)

    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated attention shape: {len(gen_attn)}")

    print("\nTransformer model tested successfully!")


def haiku_example():
    """
    Generate a haiku about transformers.
    """
    print("Transformer Model Haiku:")
    print("Data flows through gates,")
    print("Attention weaves connections,")
    print("Meaning emerges.")


if __name__ == "__main__":
    print("Transformer Model Module")
    print("=" * 40)

    # Demonstrate the model
    demo_transformer()

    # Share a haiku
    haiku_example()
