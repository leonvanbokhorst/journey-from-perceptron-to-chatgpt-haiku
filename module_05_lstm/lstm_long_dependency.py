"""
Demonstrates a Long Short-Term Memory (LSTM) network's ability
to handle long-term dependencies in sequence data, contrasting
with the limitations of a simple RNN.
Trains an LSTM on the same character prediction task (predict N steps back)
as in Module 4's `rnn_long_dependency.py`.
Corresponds to Module 5 of the curriculum.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


# --- 1. Define the LSTM Structure ---
class SimpleLSTM(nn.Module):
    """A simple LSTM model with an embedding layer, one LSTM layer, and a linear output layer.

    Designed to show improvement over SimpleRNN on long-dependency tasks.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the character embeddings.
        hidden_size (int): The number of features in the hidden state of the LSTM.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # Replacing nn.RNN with nn.LSTM!
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_cell):
        """Defines the forward pass of the LSTM.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len).
            hidden_cell (tuple[torch.Tensor, torch.Tensor]): A tuple containing the initial
                hidden state (h_0) and cell state (c_0). Each has shape
                (num_layers*num_directions, batch_size, hidden_size).

        Returns:
            tuple: A tuple containing:
                - output_logits (torch.Tensor): Logits for each character in the sequence, shape (batch_size, seq_len, vocab_size).
                - hidden_cell_new (tuple[torch.Tensor, torch.Tensor]): The final hidden state (h_n) and cell state (c_n).
        """
        # hidden_cell is a tuple (h_prev, c_prev)
        embedded = self.embed(x)  # -> (batch, seq_len, embedding_dim)
        # LSTM returns output, and the new (h_n, c_n) tuple
        out, (h_new, c_new) = self.lstm(embedded, hidden_cell)
        # out: (batch, seq_len, hidden_size)
        output_logits = self.fc(out)  # -> (batch, seq_len, vocab_size)
        return output_logits, (h_new, c_new)

    def init_hidden_cell(self, batch_size):
        """Initializes the hidden state and cell state with zeros.

        Args:
            batch_size (int): The batch size.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The initial hidden state (h_0) and cell state (c_0) tuple.
        """
        # Initialize both hidden state and cell state
        h0 = torch.zeros(
            1, batch_size, self.hidden_size
        )  # (num_layers*directions, batch, hidden)
        c0 = torch.zeros(1, batch_size, self.hidden_size)
        return (h0, c0)


# --- 2. Generate Synthetic Data (Identical to Module 4 experiment) ---
SEQ_LEN = 15
DELAY = 5
VOCAB = list("abcdefghijklmnopqrstuvwxyz")
vocab_size = len(VOCAB)
char_to_idx = {c: i for i, c in enumerate(VOCAB)}
idx_to_char = {i: c for i, c in enumerate(VOCAB)}
pad_idx = -100  # Padding index for CrossEntropyLoss ignore_index

# Create one sequence
input_chars = [random.choice(VOCAB) for _ in range(SEQ_LEN)]
target_chars_indices = [pad_idx] * DELAY + [
    char_to_idx[c] for c in input_chars[:-DELAY]
]

# Convert input to indices
input_indices = torch.tensor(
    [[char_to_idx[c] for c in input_chars]]
)  # Shape (1, SEQ_LEN)
target_indices = torch.tensor(target_chars_indices).long()  # Shape (SEQ_LEN)

print(f"--- Data Generation ---")
print(f"Input Sequence Chars:  {' '.join(input_chars)}")
print(
    f"Target Sequence Chars: {' '.join([idx_to_char.get(i, '_') for i in target_indices.tolist()])}"
)
print(f"Target Indices: {target_indices.tolist()} (Padding: {pad_idx})")

# --- 3. Set Hyperparameters & Instantiate ---
EMBEDDING_DIM = 16
HIDDEN_SIZE = 32  # Keeping same size as SimpleRNN test
LEARNING_RATE = 0.005
NUM_EPOCHS = 500  # Keeping same number of epochs

model = SimpleLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n--- Model Setup --- ")
print(model)

# --- 4. Training Loop ---
print(f"\n--- Starting LSTM Training for {NUM_EPOCHS} epochs ---")
for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()

    # Initialize *both* hidden and cell states
    hidden_cell = model.init_hidden_cell(batch_size=1)
    # Detach the tuple
    h0, c0 = hidden_cell
    hidden_cell = (h0.detach(), c0.detach())

    outputs_logits, hidden_cell_new = model(input_indices, hidden_cell)

    # Loss calculation: Reshape output to (N*SEQ_LEN, C), target is (N*SEQ_LEN)
    loss = criterion(outputs_logits.view(-1, vocab_size), target_indices)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

print("Training Finished.")

# --- 5. Evaluate the Trained Model ---
print("\n--- Evaluating the trained LSTM model ---")
model.eval()
# Initialize hidden and cell state for test
hidden_cell_test = model.init_hidden_cell(1)

with torch.no_grad():
    output_logits_test, _ = model(input_indices, hidden_cell_test)
    # output_logits_test shape: (1, SEQ_LEN, vocab_size)
    _, predicted_indices = torch.max(output_logits_test, dim=2)
    # predicted_indices shape: (1, SEQ_LEN)

# Compare only the valid (non-padded) parts
valid_length = SEQ_LEN - DELAY
predicted_valid_indices = predicted_indices.squeeze()[DELAY:]
target_valid_indices = target_indices[DELAY:]

# Ensure indices are within vocab bounds before converting back to chars
predicted_valid_chars = "".join(
    [idx_to_char.get(idx.item(), "?") for idx in predicted_valid_indices]
)
target_valid_chars = "".join(
    [idx_to_char.get(idx.item(), "?") for idx in target_valid_indices]
)
input_context_chars = "".join(
    input_chars[:-DELAY]
)  # The chars it should have predicted

print(f"Input context (used for prediction): '{input_context_chars}'")
print(f"Target sequence (delayed input):    '{target_valid_chars}'")
print(f"Predicted sequence:                 '{predicted_valid_chars}'")

correct_predictions = (predicted_valid_indices == target_valid_indices).sum().item()
accuracy = 100 * correct_predictions / valid_length
print(f"\nLSTM Accuracy on valid predictions ({valid_length} steps): {accuracy:.2f} %")

# Interpretation
if accuracy > 95:
    print("Excellent! The LSTM successfully learned the long dependency.")
elif accuracy > 75:
    print("Good performance. The LSTM clearly outperformed the SimpleRNN (likely). ")
else:
    print(
        "Hmm, LSTM performance is lower than ideal. The task is non-trivial, but this likely still beats the SimpleRNN."
    )

print(
    "\n(Compare this accuracy to the result from module_04_rnn/rnn_long_dependency.py)"
)

if __name__ == "__main__":
    """Main execution logic for the LSTM long-dependency task."""
    # --- Data Generation ---
    # ... (rest of the __main__ block remains the same)
