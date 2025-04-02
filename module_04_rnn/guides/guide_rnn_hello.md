### Experiment: Teaching the Echo (RNN Learns "hello")

This experiment focuses on the core mechanics of an RNN by training it on a very simple, specific task: predicting the next character in the sequence "hello". This avoids the complexity of large text datasets while demonstrating sequence processing.

**A Guide to Training an RNN for Sequence Prediction**

1.  **Gather the Sequence Tools (Imports):**
    We need `torch`, `torch.nn`, and `torch.optim`.

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np # Used indirectly via list indexing
    ```

2.  **Define the Echo Chamber (Define the SimpleRNN Class):**
    This RNN structure includes:

    - `nn.Embedding`: Converts character indices to vectors.
    - `nn.RNN`: The core recurrent layer (`batch_first=True`).
    - `nn.Linear`: Maps the hidden state output to scores for each character in the vocabulary.
    - `forward`: Defines the data flow (Index -> Embed -> RNN -> Linear).
    - `init_hidden`: Helper to create a starting hidden state (zeros).

    ```python
    class SimpleRNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_size):
           # ... (constructor as in script) ...
        def forward(self, x, h):
           # ... (forward logic as in script) ...
           return output_logits, h_new
        def init_hidden(self, batch_size):
           # ... (init_hidden logic as in script) ...
           return torch.zeros(1, batch_size, self.hidden_size)
    ```

3.  **Prepare the Tiny Song ("hello" Data):**

    - Define the vocabulary based only on the unique characters in "hello" (`h`, `e`, `l`, `o`).
    - Create mappings `char_to_idx` and `idx_to_char`.
    - Define the input sequence: "hell" (indices).
    - Define the target sequence: "ello" (indices) - this is what the RNN should predict at each step corresponding to the input.
    - Convert these into PyTorch tensors. The input needs a batch dimension `(1, seq_len)`. The target needs to be a LongTensor `(seq_len)`.

    ```python
    vocab = list("helo")
    vocab_size = len(vocab)
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}

    input_str = "hell"
    target_str = "ello"

    input_indices = torch.tensor([[char_to_idx[c] for c in input_str]])
    target_indices = torch.tensor([char_to_idx[c] for c in target_str]).long()
    ```

4.  **Set the Stage (Hyperparameters & Instantiation):**

    - Define `EMBEDDING_DIM`, `HIDDEN_SIZE` (memory size), `LEARNING_RATE`, and `NUM_EPOCHS`.
    - Instantiate the `SimpleRNN` model.
    - Use `nn.CrossEntropyLoss` for classification (predicting the next character class).
    - Use `optim.Adam` as the optimizer.

    ```python
    EMBEDDING_DIM = 10
    HIDDEN_SIZE = 16
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 100

    model = SimpleRNN(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ```

5.  **The Training Ritual (Training Loop):**

    - Loop through epochs.
    - **Crucially:** Initialize the hidden state `hidden = model.init_hidden(1)` _before_ processing the sequence in each epoch.
    - **Detach** the hidden state (`hidden = hidden.detach()`) to prevent gradients from flowing across epoch boundaries (important for longer sequences, good practice here).
    - Zero gradients.
    - Perform the forward pass: `outputs_logits, hidden_new = model(input_indices, hidden)`.
    - **Reshape** the `outputs_logits` from `(batch, seq_len, vocab_size)` to `(batch*seq_len, vocab_size)` to match the `CrossEntropyLoss` expectation.
    - Calculate the loss between the reshaped logits and the flat `target_indices`.
    - Backward pass and optimizer step.
    - Print loss periodically.

    ```python
    print(f"\nStarting Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        hidden = model.init_hidden(1).detach()
        outputs_logits, hidden_new = model(input_indices, hidden)
        loss = criterion(outputs_logits.view(-1, vocab_size), target_indices)
        loss.backward()
        optimizer.step()
        # ... (printing logic as in script) ...
    print("Training Finished.")
    ```

6.  **Check the Echo (Testing):**

    - Set model to evaluation mode (`model.eval()`).
    - Initialize hidden state.
    - Pass the input sequence "hell" through the model (`with torch.no_grad():`).
    - Get the output logits.
    - For each time step, find the character index with the highest score using `torch.max(..., dim=2)`.
    - Convert the predicted indices back to characters.
    - Compare the predicted string to the target string "ello".

    ```python
    print("\nTesting the trained model...")
    model.eval()
    # ... (testing logic as in script) ...
    predicted_str = "".join([...])
    print(f"Input string:  '{input_str}'")
    print(f"Target string: '{target_str}'")
    print(f"Predicted string (next chars): '{predicted_str}'")
    # ... (success/fail message) ...
    ```

**Running the Code:**
Execute `rnn_hello.py`. It requires no external data. Watch the loss decrease during training. The final output will show if the RNN successfully learned to predict 'e' after 'h', 'l' after 'e', 'l' after 'l', and 'o' after 'l'.

**Reflection:**
This simple example demonstrates how an RNN processes sequence information step-by-step, using its hidden state to remember context (like the previous character) to predict the next output. Even learning "hello" requires remembering the sequence order. This forms the basis for understanding how RNNs handle more complex sequential tasks.
