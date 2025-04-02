"""
Observes the output of a Multi-Head Self-Attention layer in PyTorch.

Demonstrates the structure and properties of the attention weight matrix
calculated using `torch.nn.MultiheadAttention` on a dummy sequence.
Corresponds to the hands-on example in Module 6 of the curriculum.
"""

import torch
import torch.nn as nn
import numpy as np  # For printing clarity

print("--- Muppet Labs Presents: Observing Multi-Head Self-Attention ---")

# --- 1. Parameters ---
seq_len = 3  # Words in our sequence
embed_dim = 4  # Vector dimension per word (must be divisible by num_heads)
num_heads = 2  # Number of attention heads
batch_size = 1  # We'll process one sequence

print(f"\nParameters:")
print(f"  Sequence Length (L): {seq_len}")
print(f"  Embedding Dimension (E): {embed_dim}")
print(f"  Number of Heads: {num_heads}")
print(f"  Batch Size (N): {batch_size}")

# --- 2. Prepare Dummy Input ---
# Shape requires [L, N, E] for batch_first=False (default)
dummy_input = torch.randn(seq_len, batch_size, embed_dim)
print(f"\nGenerated Dummy Input Shape: {dummy_input.shape}")
# print("Input Tensor:\n", dummy_input)

# --- 3. Instantiate Attention Layer ---
# Ensure embed_dim is divisible by num_heads
assert (
    embed_dim % num_heads == 0
), "Embedding dimension must be divisible by number of heads"

attention_layer = nn.MultiheadAttention(
    embed_dim=embed_dim, num_heads=num_heads, batch_first=False
)
print("\nInstantiated nn.MultiheadAttention Layer:")
# print(attention_layer) # Can be verbose

# --- 4. Calculate Self-Attention ---
print("\nCalculating self-attention (Input -> Query, Key, Value)...")
# For self-attention, the input sequence serves as Q, K, and V
# We are primarily interested in the attention weights here
attn_output, attn_output_weights = attention_layer(
    dummy_input, dummy_input, dummy_input
)

print(f"\nAttention Output Shape: {attn_output.shape}")  # Expected: [L, N, E]
print(f"Attention Weights Shape: {attn_output_weights.shape}")  # Expected: [N, L, L]

# --- 5. Examine Attention Weights ---
print("\n--- Examining Calculated Attention Weights --- ")
# Detach from graph, move to CPU (if needed), convert to numpy for printing
weights_matrix = attn_output_weights.squeeze(0).detach().cpu().numpy()

# Set print options for readability
np.set_printoptions(precision=3, suppress=True)

print(f"Attention Matrix (Shape: {weights_matrix.shape}):\n{weights_matrix}")

print("\nInterpretation (How much each word attends to others):")
for i in range(seq_len):
    row_sum = weights_matrix[i, :].sum()
    print(f"  Input Word {i+1} (Query) attended to:")
    for j in range(seq_len):
        print(f"    - Word {j+1} (Key): {weights_matrix[i, j]:.3f}")
    # Weights for each query word should sum to 1 due to softmax
    print(f"  (Row {i+1} Sum: {row_sum:.3f}) <-- Should be ~1.0")

print(
    "\nExperiment Complete! Observe how weights distribute attention across the sequence."
)
print(
    "(Note: Weights are based on random input and untrained layer; they lack specific meaning here.)"
)
