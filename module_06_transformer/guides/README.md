# Module 6 Guide: Attention and Transformers – Sequence Learning Revolutionized

This guide provides context for the code example in Module 6, introducing the Attention mechanism and the Transformer architecture. For the full theoretical background (Self-Attention, Multi-Head Attention, Positional Encoding, architecture details), historical context, and exercises, please refer to **Module 6** in the main `../../curriculum.md` file.

## Objectives Recap

- Explain the concept of Attention in sequence models.
- Introduce the Transformer architecture based solely on attention.
- Understand key components: self-attention, multi-head attention, positional encoding.
- Observe the output of an attention layer in PyTorch.

## Code Example: `observe_attention.py`

The script `../observe_attention.py` provides a focused look at the core **Multi-Head Self-Attention** mechanism, as implemented in `torch.nn.MultiheadAttention`.

**Key Steps in the Code:**

1.  **Parameters:** Defines basic parameters for a toy example: sequence length (`seq_len`), embedding dimension (`embed_dim`), number of attention heads (`num_heads`), and batch size (`batch_size`).
2.  **Dummy Input:** Creates a random tensor representing a sequence of embeddings.
3.  **Instantiate Layer:** Creates an instance of `nn.MultiheadAttention`.
4.  **Calculate Self-Attention:** Calls the attention layer with the `dummy_input` serving as the Query, Key, and Value. This performs self-attention.
    - `attn_output, attn_output_weights = attention_layer(dummy_input, dummy_input, dummy_input)`
5.  **Examine Weights:** Extracts the `attn_output_weights` tensor.
    - This tensor has shape `[N, L, L]` (Batch Size, Sequence Length, Sequence Length).
    - Each element `weights[b, i, j]` represents how much the `i`-th element of the sequence (Query) attends to the `j`-th element (Key) in batch `b`.
    - The script prints this matrix and iterates through its rows, confirming that the weights in each row (attention paid _by_ one element _to_ all others) sum to approximately 1.0 due to the internal softmax operation.

**Running the Code:**

Navigate to the module directory and run the script:

```bash
cd module_06_transformer
python observe_attention.py
```

**Expected Outcome:**

The script will print the shapes of the input, attention output, and attention weights. It will then display the calculated attention weight matrix (filled with values between 0 and 1). Because the input and the attention layer are random/untrained, the specific weight values lack inherent meaning, but the _structure_ and the _row-wise sum-to-one property_ demonstrate the core output of a self-attention mechanism.

This exercise isolates the attention calculation, paving the way for understanding how it's used within the full Transformer architecture (discussed in the curriculum and implemented in Module 7).

## Exercises & Further Exploration

Refer back to **Module 6** in `../../curriculum.md` for:

- **Exercise 6.1:** Manually computing attention for a tiny example.
- **Exercise 6.2:** Experimenting with the effect of removing positional encoding.
- **Exercise 6.3 (Project):** Training a small transformer on a toy sequence-to-sequence task.
- **Reflection 6.4:** Relating model attention to human selective attention.

This guide clarifies how the code provides a practical glimpse into the attention mechanism that powers Transformers.
