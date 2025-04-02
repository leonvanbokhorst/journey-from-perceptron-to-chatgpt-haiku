### Experiment: Witnessing the Focused Gaze (Observing Attention)

This experiment isolates the core `MultiheadAttention` mechanism used in Transformers. We apply it to a dummy sequence to directly observe the calculated attention weights, illustrating how each element can attend to all other elements.

**A Muppet Labs Guide to Observing Multi-Head Self-Attention**

1.  **Gather the Focusing Tools (Imports):**
    We need `torch` and `torch.nn`. `numpy` is helpful for clearer printing of the results.

    ```python
    import torch
    import torch.nn as nn
    import numpy as np
    ```

2.  **Prepare a Simple Sequence (Dummy Data):**

    - Define parameters: `seq_len` (e.g., 3), `embed_dim` (e.g., 4), `num_heads` (e.g., 2). Ensure `embed_dim` is divisible by `num_heads`.
    - Create a random tensor representing our input sequence with shape `[seq_len, batch_size, embed_dim]`. We use `batch_size=1`.

    ```python
    seq_len = 3
    embed_dim = 4
    num_heads = 2
    batch_size = 1

    dummy_input = torch.randn(seq_len, batch_size, embed_dim)
    ```

3.  **Instantiate the Attention Mechanism:**

    - Create an instance of `nn.MultiheadAttention`, passing `embed_dim` and `num_heads`. We use `batch_first=False` (the default) which matches our input tensor shape `[L, N, E]`.

    ```python
    attention_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)
    ```

4.  **Perform the Attentive Calculation (Self-Attention):**

    - Call the `attention_layer`, passing the `dummy_input` as the query, key, and value arguments. This configures it for self-attention.
    - Capture the two outputs: `attn_output` (the sequence elements updated by attention) and `attn_output_weights` (the attention scores).

    ```python
    attn_output, attn_output_weights = attention_layer(dummy_input, dummy_input, dummy_input)
    ```

5.  **Gaze Upon the Weights (Examine Results):**

    - The `attn_output_weights` tensor has the shape `[batch_size, seq_len, seq_len]` or `[N, L, L]`.
    - Extract the weights matrix for our single batch item (`squeeze(0)`).
    - Print this matrix. Each row `i` shows how much attention element `i` (the query) paid to element `j` (the key) across all columns `j`.
    - Verify that each row sums to approximately 1.0, confirming the softmax normalization.

    ```python
    # Detach, move to CPU, convert to numpy
    weights_matrix = attn_output_weights.squeeze(0).detach().cpu().numpy()
    np.set_printoptions(precision=3, suppress=True)
    print(f"Attention Matrix:\n{weights_matrix}")

    # Print interpretation loop (as in script)
    for i in range(seq_len):
        # ... print attention from word i to word j ...
        print(f"  (Row {i+1} Sum: {weights_matrix[i, :].sum():.3f}) <-- Should be ~1.0")
    ```

**Running the Code & Interpretation:**
Execute `observe_attention.py`. It will print the shape of the input, the attention output, and the crucial attention weights matrix. Examine the matrix: see how each row distributes values across the columns. Although the specific values are random here (due to random input and untrained weights), the _structure_ demonstrates the mechanism: every element calculates a relationship score with every other element (including itself), and these scores determine how information is aggregated.

**Reflection:**
This direct calculation of relationship scores between all pairs of elements simultaneously is the heart of the Transformer's power. It allows parallel processing and enables the model to directly capture long-range dependencies without information needing to pass sequentially through many steps like in an RNN or LSTM. This fundamental difference paved the way for training much larger and more capable sequence models.
