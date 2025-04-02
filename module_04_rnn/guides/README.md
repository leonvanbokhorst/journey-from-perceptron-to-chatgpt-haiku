# Module 4 Guide: Recurrent Neural Networks – Modeling Sequence and Memory

This guide provides context for the code examples in Module 4, introducing Recurrent Neural Networks (RNNs). For the full theoretical background, discussion of vanishing/exploding gradients, and exercises, please refer to **Module 4** in the main `../../curriculum.md` file.

## Objectives Recap

- Introduce RNNs for processing sequence data.
- Understand the concept of a hidden state carrying memory across time steps.
- Explain the challenges RNNs face with long sequences (vanishing gradients).
- Implement simple RNNs in PyTorch using `nn.Embedding` and `nn.RNN`.

## Code Examples

This module includes two scripts:

1.  `../rnn_hello.py`: A basic example learning a very short sequence.
2.  `../rnn_long_dependency.py`: An example designed to illustrate RNN limitations.

### 1. `rnn_hello.py`

This script trains a simple RNN to predict the next character in the sequence "hello" (input "hell", target "ello").

**Key Components:**

- **`SimpleRNN` Class:** Defines the network using `nn.Embedding` (to convert character indices to vectors), `nn.RNN` (the recurrent layer), and `nn.Linear` (to map hidden states to output logits).
- **Data Preparation:** Creates a small vocabulary (`h, e, l, o`) and converts the input/target strings into corresponding index tensors.
- **Training Loop:**
  - Initializes the hidden state (`model.init_hidden`).
  - **Detaches hidden state:** `hidden = hidden.detach()` prevents gradients from flowing across iterations/epochs.
  - Performs a forward pass through the RNN for the entire sequence.
  - Calculates `nn.CrossEntropyLoss` between the output logits (reshaped) and the target indices.
  - Performs backpropagation and optimizer step.
- **Testing:** After training, feeds the input "hell" again and checks if the predicted next characters match "ello".

**Running the Code:**

```bash
cd module_04_rnn
python rnn_hello.py
```

With enough epochs, this simple model should successfully learn the short "hello" sequence.

### 2. `rnn_long_dependency.py`

This script highlights the difficulty vanilla RNNs have with **long-term dependencies**, a consequence of the vanishing gradient problem.

**Key Components:**

- **Task Setup:** Creates a synthetic task where the network must predict the character from `DELAY` steps ago (e.g., 5 steps). The input is a random sequence of characters.
- **Padding:** Uses `ignore_index=-100` in `nn.CrossEntropyLoss` so the model isn't penalized for predictions during the initial `DELAY` steps where the target is undefined.
- **Model:** Uses the _same_ `SimpleRNN` architecture as `rnn_hello.py`.
- **Training:** Trains the RNN on this more challenging sequence prediction task.
- **Evaluation:** Compares the predicted characters (after the initial delay) with the actual target characters and calculates accuracy.

**Running the Code:**

```bash
cd module_04_rnn
python rnn_long_dependency.py
```

You will likely observe that the accuracy is significantly lower than 100%, especially if `SEQ_LEN` is increased or `DELAY` is larger. The script includes comments interpreting this expected outcome, demonstrating the RNN's struggle. This motivates the need for architectures like LSTMs (Module 5).

## Exercises & Further Exploration

Refer back to **Module 4** in `../../curriculum.md` for:

- **Exercise 4.1:** Implementing a sequence memorization task.
- **Exercise 4.2:** Training a char-RNN on actual haiku text (expect limited results).
- **Exercise 4.3:** Deriving Backpropagation Through Time (BPTT) equations.
- **Reflection 4.4:** Comparing RNN memory to human memory.

This guide clarifies the two RNN examples, illustrating both basic operation and inherent limitations.
