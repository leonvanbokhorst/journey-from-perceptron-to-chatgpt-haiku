"""
Demonstrates a basic Recurrent Neural Network (RNN) for sequence learning.
Trains a simple RNN to predict the next character in the sequence "hello".
Corresponds to Module 4 of the curriculum.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# --- 1. Define the RNN Structure (Simplified CharRNN) ---
class SimpleRNN(nn.Module):
    """A simple RNN model with an embedding layer, one RNN layer, and a linear output layer.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the character embeddings.
        hidden_size (int): The number of features in the hidden state of the RNN.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # Embedding layer: converts index to vector
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # RNN layer: processes sequence
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        # Output layer: maps hidden state to vocab scores
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        """Defines the forward pass of the RNN.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len).
            h (torch.Tensor): Initial hidden state tensor of shape (num_layers*num_directions, batch_size, hidden_size).

        Returns:
            tuple: A tuple containing:
                - output_logits (torch.Tensor): Logits for each character in the sequence, shape (batch_size, seq_len, vocab_size).
                - h_new (torch.Tensor): The final hidden state, shape (num_layers*num_directions, batch_size, hidden_size).
        """
        # x: (batch, seq_len) -> LongTensor
        # h: (num_layers*num_directions, batch, hidden_size)
        embedded = self.embed(x)  # -> (batch, seq_len, embedding_dim)
        out, h_new = self.rnn(embedded, h)
        # out: (batch, seq_len, hidden_size)
        output_logits = self.fc(out)  # -> (batch, seq_len, vocab_size)
        return output_logits, h_new

    def init_hidden(self, batch_size):
        """Initializes the hidden state with zeros.

        Args:
            batch_size (int): The batch size.

        Returns:
            torch.Tensor: The initial hidden state tensor.
        """
        # Initial hidden state (zeros)
        return torch.zeros(1, batch_size, self.hidden_size)  # (1 layer, 1 direction)


# --- 2. Prepare Data for "hello" ---
# Vocabulary: h, e, l, o
vocab = list("helo")  # Unique characters
vocab_size = len(vocab)
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}

# Sequence: input 'hell', target 'ello'
input_str = "hell"
target_str = "ello"

# Convert strings to index tensors
input_indices = torch.tensor([[char_to_idx[c] for c in input_str]])  # Shape (1, 4)
target_indices = torch.tensor([char_to_idx[c] for c in target_str])  # Shape (4)
# Note: Target for CrossEntropyLoss should be shape (N) or (N, d1, ...)
# For sequence loss, it's often (N*seq_len) where N is batch size.
# Here N=1, so target should be shape (seq_len) = (4).
target_indices = target_indices.long()  # Ensure LongTensor for loss function

print(f"Vocabulary: {vocab}")
print(f"Input sequence (indices): {input_indices.tolist()}")
print(f"Target sequence (indices): {target_indices.tolist()}")

# --- 3. Set Hyperparameters & Instantiate ---
EMBEDDING_DIM = 10
HIDDEN_SIZE = 16  # Size of RNN memory
LEARNING_RATE = 0.01
NUM_EPOCHS = 100

model = SimpleRNN(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE)
criterion = nn.CrossEntropyLoss()  # Handles logit -> probability and loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nRNN Model Structure:")
print(model)

# --- 4. Training Loop ---
print(f"\nStarting Training for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()

    # Initialize hidden state for the batch (batch size is 1 here)
    hidden = model.init_hidden(batch_size=1)

    # --- Important: Detach hidden state ---
    # We detach the hidden state from the computation graph history of the previous iteration.
    # This prevents gradients from flowing back across epoch/batch boundaries,
    # which is standard practice for stateful RNN training.
    hidden = hidden.detach()

    # Forward pass
    # Input shape: (1, 4), Hidden shape: (1, 1, hidden_size)
    outputs_logits, hidden_new = model(input_indices, hidden)
    # outputs_logits shape: (1, 4, vocab_size)

    # Calculate loss
    # CrossEntropyLoss expects logits as (N, C) or (N, C, d1...)
    # and targets as (N) or (N, d1...)
    # Here N=batch*seq_len, C=vocab_size. We treat each time step prediction independently.
    # Reshape output: (1, 4, vocab_size) -> (1*4, vocab_size) = (4, vocab_size)
    # Target shape is already (4)
    loss = criterion(outputs_logits.view(-1, vocab_size), target_indices)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

print("Training Finished.")

# --- 5. Test the Trained Model ---
print("\nTesting the trained model...")
model.eval()

# Use the same input "hell"
input_test = input_indices
hidden_test = model.init_hidden(1)

with torch.no_grad():
    output_logits_test, _ = model(input_test, hidden_test)
    # Get prediction for each step
    # output_logits_test shape: (1, 4, vocab_size)
    # Find the index with the highest logit value for each step in the sequence
    _, predicted_indices = torch.max(output_logits_test, dim=2)
    # predicted_indices shape: (1, 4)

predicted_str = "".join(
    [idx_to_char[idx.item()] for idx in predicted_indices.squeeze()]
)

print(f"Input string:  '{input_str}'")
print(f"Target string: '{target_str}'")
print(f"Predicted string (next chars): '{predicted_str}'")

if predicted_str == target_str:
    print("Success! The RNN learned the sequence.")
else:
    print("The RNN did not perfectly learn the sequence.")

if __name__ == "__main__":
    """Main execution logic for the SimpleRNN 'hello' example."""
    # --- Data Setup ---
    # ... (rest of the __main__ block remains the same)
