### Experiment: LSTM Tackles the Long Dependency

This experiment directly compares the performance of an LSTM against the SimpleRNN from Module 4 on the challenging long-dependency task (predicting the character from `DELAY` steps ago). We hypothesize the LSTM's gating mechanisms will enable superior performance.

**A Muppet Labs Procedure for LSTM Evaluation**

1.  **Assemble the Apparatus (Imports):**
    Standard PyTorch imports (`torch`, `nn`, `optim`) plus `random` and `numpy`.

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random
    import numpy as np
    ```

2.  **Construct the LSTM Unit (Define the LSTM Model):**

    - The model architecture (`SimpleLSTM`) mirrors the `SimpleRNN` but crucially replaces `nn.RNN` with `nn.LSTM`.
    - The `forward` method now handles the `(hidden_state, cell_state)` tuple characteristic of LSTMs.
    - `init_hidden_cell` initializes both states to zeros.

    ```python
    class SimpleLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_size):
            # ... (constructor as in script) ...
        def forward(self, x, hidden_cell):
            # ... (forward logic as in script) ...
            return output_logits, (h_new, c_new)
        def init_hidden_cell(self, batch_size):
            # ... (init logic as in script) ...
            return (h0, c0)
    ```

3.  **Prepare the Test Subject (Data Generation - Identical):**

    - Crucially, we use the _exact same_ setup as in `module_04_rnn/rnn_long_dependency.py` to generate the input sequence and the delayed target sequence (`SEQ_LEN`, `DELAY`, `VOCAB`, `pad_idx`). This ensures a fair comparison.

    ```python
    SEQ_LEN = 15
    DELAY = 5
    VOCAB = list("abcdefghijklmnopqrstuvwxyz")
    # ... (char_to_idx, idx_to_char, pad_idx)
    # ... (Code to generate input_chars, target_chars_indices)
    input_indices = torch.tensor([[...]])
    target_indices = torch.tensor([...]).long()
    print("Generated identical synthetic data for LSTM test.")
    ```

4.  **Calibrate Instruments (Hyperparameters & Instantiation):**

    - We use the same `EMBEDDING_DIM`, `HIDDEN_SIZE`, `LEARNING_RATE`, and `NUM_EPOCHS` as the corresponding SimpleRNN experiment for direct comparison.
    - Instantiate the `SimpleLSTM` model.
    - Use `CrossEntropyLoss(ignore_index=pad_idx)` to handle padding.
    - Use `Adam` optimizer.

    ```python
    EMBEDDING_DIM = 16
    HIDDEN_SIZE = 32
    LEARNING_RATE = 0.005
    NUM_EPOCHS = 500

    model = SimpleLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("\nLSTM Model Structure:")
    print(model)
    ```

5.  **Initiate Experiment Sequence (Training Loop):**

    - The loop structure mirrors the SimpleRNN training.
    - **Key Difference:** Initialize _both_ hidden and cell states using `model.init_hidden_cell(1)`.
    - Detach the _tuple_ `(h0.detach(), c0.detach())` before the forward pass in each epoch.
    - The forward pass uses and returns the `(h, c)` tuple.

    ```python
    print(f"\nStarting LSTM Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        h0, c0 = model.init_hidden_cell(1)
        hidden_cell = (h0.detach(), c0.detach())
        outputs_logits, hidden_cell_new = model(input_indices, hidden_cell)
        loss = criterion(outputs_logits.view(-1, vocab_size), target_indices)
        loss.backward()
        optimizer.step()
        # ... (printing logic) ...
    print("Training Finished.")
    ```

6.  **Analyze Results (Evaluation):**

    - Evaluation logic is identical to the SimpleRNN test.
    - Initialize the `(h, c)` tuple for evaluation.
    - Get predictions using `torch.no_grad()`.
    - Compare the valid (non-padded) predictions against the target.
    - Calculate and print the accuracy.
    - Print an interpretation comparing the expected LSTM performance (high accuracy) to the SimpleRNN's likely poor performance on this task.

    ```python
    print("\nEvaluating the trained LSTM model...")
    model.eval()
    hidden_cell_test = model.init_hidden_cell(1)
    with torch.no_grad():
        # ... (prediction logic) ...
    # ... (comparison logic for valid parts) ...
    accuracy = 100 * correct_predictions / valid_length
    print(f"\nLSTM Accuracy on valid predictions ({valid_length} steps): {accuracy:.2f} %")
    # ... (Interpretation message)
    print("\n(Compare this accuracy to the result from module_04_rnn/rnn_long_dependency.py)")
    ```

**Running the Code & Expected Outcome:**
Execute `lstm_long_dependency.py`. The training process will run. Observe the final accuracy. It is strongly expected that the LSTM will achieve significantly higher accuracy (likely >90-95%) compared to the SimpleRNN on this identical task. This difference clearly demonstrates the effectiveness of the LSTM's gating mechanisms in preserving information over longer time intervals, overcoming the vanishing gradient problem that hindered the SimpleRNN.

**Reflection:**
The gates within the LSTM act like intelligent regulators, allowing the network to selectively forget irrelevant information and retain crucial details in its cell state across many time steps. This experiment provides concrete evidence of this capability and motivates the use of LSTMs (or similar architectures like GRUs) for tasks requiring robust handling of sequential data with long-range dependencies.
