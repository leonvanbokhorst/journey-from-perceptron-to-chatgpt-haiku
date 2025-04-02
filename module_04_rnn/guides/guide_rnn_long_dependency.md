### Experiment: Testing the Limits (RNN Long Dependency)

This experiment directly tests the vanilla RNN's ability to handle longer-range dependencies, a known weakness due to the vanishing gradient problem. We create a synthetic task where the output at time `t` depends on the input from time `t - DELAY` (e.g., `t-5`).

**A Guide to Demonstrating RNN Limitations**

1.  **RNN Structure (Imports & Class Definition):**
    We reuse the exact same `SimpleRNN` class definition and necessary imports (`torch`, `nn`, `optim`) from the `rnn_hello.py` experiment. The underlying architecture is unchanged.

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random

    class SimpleRNN(nn.Module):
        # ... (Same RNN class as before) ...
    ```

2.  **Craft the Challenge (Generate Synthetic Data):**

    - Define a longer sequence length (`SEQ_LEN`, e.g., 15) and a delay (`DELAY`, e.g., 5).
    - Define a larger vocabulary (`VOCAB`).
    - Generate a random sequence of characters (`input_chars`).
    - Create the target sequence: This should be the input sequence, but shifted forward by `DELAY` steps. The first `DELAY` steps of the target have no corresponding input from the past, so we mark them with a special padding index (`pad_idx = -100`) that the loss function will ignore.
    - Convert input and target character sequences to index tensors.

    ```python
    SEQ_LEN = 15
    DELAY = 5
    VOCAB = list("abcdefghijklmnopqrstuvwxyz")
    vocab_size = len(VOCAB)
    # ... (char_to_idx, idx_to_char)

    input_chars = [random.choice(VOCAB) for _ in range(SEQ_LEN)]
    pad_idx = -100
    target_chars_indices = [pad_idx] * DELAY + [char_to_idx[c] for c in input_chars[:-DELAY]]

    input_indices = torch.tensor([[char_to_idx[c] for c in input_chars]])
    target_indices = torch.tensor(target_chars_indices).long()
    ```

3.  **Set the Stage (Hyperparameters & Instantiation):**

    - Define `EMBEDDING_DIM` and `HIDDEN_SIZE`. The hidden size might need to be larger than in the "hello" example to store information longer.
    - Set `LEARNING_RATE` and `NUM_EPOCHS`. This task might require more epochs to learn.
    - Instantiate the `SimpleRNN` model.
    - Crucially, define the `CrossEntropyLoss` with `ignore_index=pad_idx` so it doesn't penalize predictions for the initial padded steps.
    - Instantiate the `Adam` optimizer.

    ```python
    EMBEDDING_DIM = 16
    HIDDEN_SIZE = 32
    LEARNING_RATE = 0.005
    NUM_EPOCHS = 500

    model = SimpleRNN(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ```

4.  **The Training Ritual (Training Loop):**
    The training loop structure is identical to the `rnn_hello` experiment.

    - Loop epochs.
    - Initialize and detach hidden state.
    - Zero gradients.
    - Forward pass.
    - Calculate loss (using `ignore_index`).
    - Backward pass and optimizer step.
    - Print loss periodically.

    ```python
    print(f"\nStarting Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        # ... (Training loop logic as in script) ...
    print("Training Finished.")
    ```

5.  **Observe the Struggle (Evaluation):**

    - Set model to evaluation mode.
    - Pass the input sequence through the model.
    - Get the predicted character indices for each step.
    - **Important:** Compare only the _valid_ predictions (from step `DELAY` onwards) with the corresponding target characters.
    - Calculate the accuracy on these valid steps.
    - Print the input context, the target sequence, and the predicted sequence for comparison.
    - Include a message interpreting the accuracy (e.g., low accuracy indicates the expected struggle).

    ```python
    print("\nEvaluating the trained model...")
    model.eval()
    # ... (Evaluation logic as in script) ...
    predicted_valid_chars = "".join([...])
    target_valid_chars = "".join([...])
    input_context_chars = "".join([...])
    # ... (Print input context, target, prediction) ...
    accuracy = 100 * correct_predictions / valid_length
    print(f"\nAccuracy on valid predictions ({valid_length} steps): {accuracy:.2f} %")
    # ... (Interpretation message based on accuracy) ...
    ```

**Running the Code:**
Execute `rnn_long_dependency.py`. It trains the RNN on the synthetic delay task. Observe the loss during training – it might decrease initially but potentially plateau at a higher level than in the "hello" task.
Pay close attention to the final accuracy reported for the valid prediction steps. It is expected that the vanilla RNN will perform poorly (significantly less than 100% accuracy) because retaining information reliably over 5 steps is challenging for it.

**Reflection:**
This experiment highlights the practical difficulty vanilla RNNs face with longer-term dependencies. The hidden state, updated at every step, struggles to preserve information from many steps prior without it being corrupted or washed out. This observed limitation directly motivates the development of more sophisticated recurrent architectures like LSTMs and GRUs (Module 5), which employ gating mechanisms to better control information flow and memory retention.
