# Module 5 Guide: Long Short-Term Memory (LSTM) Networks – Overcoming Forgetfulness

This guide provides context for the code example in Module 5, introducing Long Short-Term Memory (LSTM) networks. For the full theoretical background (including gates: forget, input, output), comparison with GRUs, and exercises, please refer to **Module 5** in the main `../../curriculum.md` file.

## Objectives Recap

- Present LSTMs as a solution to the vanishing gradient problem in RNNs.
- Understand the LSTM's gating mechanisms and the role of the cell state.
- Implement or use an LSTM in PyTorch (`nn.LSTM`).
- Observe the improved performance of LSTMs on tasks requiring longer memory.

## Code Example: `lstm_long_dependency.py`

The script `../lstm_long_dependency.py` directly addresses the limitations shown in Module 4's `rnn_long_dependency.py`. It uses an LSTM network on the _exact same_ long-term dependency task (predicting a character from `DELAY` steps ago).

**Key Components:**

1.  **`SimpleLSTM` Class (inherits from `nn.Module`):**
    - `__init__`: Defines the layers, critically replacing `nn.RNN` with `nn.LSTM`.
    - `forward`: Defines the data flow. Note that `nn.LSTM` takes the previous hidden state _and_ cell state `(h_prev, c_prev)` as input and returns the output sequence plus the _new_ hidden and cell states `(h_new, c_new)`.
    - `init_hidden_cell`: Initializes _both_ the hidden state `h0` and the cell state `c0` to zeros.
2.  **Data Generation:** Identical to the setup in `module_04_rnn/rnn_long_dependency.py` to ensure a fair comparison.
3.  **Training Loop:**
    - Similar structure to the RNN training loop.
    - Crucially, initializes **both** hidden and cell states using `model.init_hidden_cell()`.
    - Detaches **both** states `(h0.detach(), c0.detach())` before the forward pass.
    - Passes the `hidden_cell` tuple to the model's forward method.
    - Loss calculation and backpropagation proceed as before, but gradients flow more effectively through the LSTM's structure due to the gates and cell state.
4.  **Evaluation:** Identical setup to the RNN evaluation, calculating accuracy on the valid (non-padded) part of the sequence.

**Running the Code:**

Navigate to the module directory and run the script:

```bash
cd module_05_lstm
python lstm_long_dependency.py
```

**Expected Outcome:**

You should observe significantly higher accuracy compared to the result from `module_04_rnn/rnn_long_dependency.py` when run with the same parameters (sequence length, delay, epochs). This demonstrates the LSTM's superior ability to capture longer-range dependencies in the sequence data, effectively mitigating the vanishing gradient problem that hinders the simple RNN.

The script includes print statements interpreting the accuracy and explicitly prompting the comparison with the Module 4 results.

## Exercises & Further Exploration

Refer back to **Module 5** in `../../curriculum.md` for:

- **Exercise 5.1:** Comparing RNN vs. LSTM on a synthetic copy task.
- **Exercise 5.2:** Inspecting LSTM gate activations during processing.
- **Exercise 5.3:** Implementing an LSTM cell manually from equations.
- **Reflection 5.4:** Considering parallels between LSTM memory and human memory/identity.

This guide emphasizes how the LSTM code directly demonstrates the solution to the problems raised in the previous module.
