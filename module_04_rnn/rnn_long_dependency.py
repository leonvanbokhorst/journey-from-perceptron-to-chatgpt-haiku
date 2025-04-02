"""
Demonstrates the difficulty vanilla RNNs face with long-term dependencies.
Trains a simple RNN to predict a character from several steps prior in a sequence.
This illustrates the vanishing gradient problem discussed in Module 4.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random


# --- 1. Define the RNN Structure (Same as rnn_hello.py) ---
class SimpleRNN(nn.Module):
    """A simple RNN model (identical structure to rnn_hello.py).

    Used here to demonstrate limitations on long-dependency tasks.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the character embeddings.
        hidden_size (int): The number of features in the hidden state of the RNN.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
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
        embedded = self.embed(x)
        out, h_new = self.rnn(embedded, h)
        output_logits = self.fc(out)
        return output_logits, h_new

    def init_hidden(self, batch_size):
        """Initializes the hidden state with zeros.

        Args:
            batch_size (int): The batch size.

        Returns:
            torch.Tensor: The initial hidden state tensor.
        """
        return torch.zeros(1, batch_size, self.hidden_size)


# --- 2. Generate Synthetic Data for Long Dependency Task ---
# Task: Predict the character from 5 steps ago.
# Example: Input:  a b c d e f g h i j
#          Target: _ _ _ _ _ a b c d e (_ = padding/ignore)

SEQ_LEN = 15
DELAY = 5
VOCAB = list("abcdefghijklmnopqrstuvwxyz")  # Larger vocab
vocab_size = len(VOCAB)
char_to_idx = {c: i for i, c in enumerate(VOCAB)}
idx_to_char = {i: c for i, c in enumerate(VOCAB)}

# Create one sequence
input_chars = [random.choice(VOCAB) for _ in range(SEQ_LEN)]
# Target is the input sequence shifted by DELAY
# Use a special padding index (e.g., -100) for initial steps where target is undefined
# CrossEntropyLoss ignores targets with index -100 by default
pad_idx = -100
target_chars_indices = [pad_idx] * DELAY + [
    char_to_idx[c] for c in input_chars[:-DELAY]
]

# Convert input to indices
input_indices = torch.tensor(
    [[char_to_idx[c] for c in input_chars]]
)  # Shape (1, SEQ_LEN)
target_indices = torch.tensor(target_chars_indices).long()  # Shape (SEQ_LEN)

print(f"Input Sequence Chars:  {' '.join(input_chars)}")
print(
    f"Target Sequence Chars: {' '.join([idx_to_char.get(i, '_') for i in target_indices.tolist()])}"
)  # Show padding as _
print(f"Input Indices:  {input_indices.tolist()}")
print(f"Target Indices: {target_indices.tolist()}")

# --- 3. Set Hyperparameters & Instantiate ---
EMBEDDING_DIM = 16
HIDDEN_SIZE = 32  # Might need more memory than 'hello'
LEARNING_RATE = 0.005
NUM_EPOCHS = 500  # Needs more training potentially

model = SimpleRNN(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE)
# Use ignore_index for padding
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nRNN Model Structure:")
print(model)

# --- 4. Training Loop ---
print(f"\nStarting Training for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    hidden = model.init_hidden(batch_size=1).detach()

    outputs_logits, hidden_new = model(input_indices, hidden)
    # outputs_logits shape: (1, SEQ_LEN, vocab_size)

    # Loss calculation: Reshape output to (N*SEQ_LEN, C), target is (N*SEQ_LEN)
    loss = criterion(outputs_logits.view(-1, vocab_size), target_indices)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

print("Training Finished.")

# --- 5. Evaluate the Trained Model ---
print("\nEvaluating the trained model...")
model.eval()
hidden_test = model.init_hidden(1)

with torch.no_grad():
    output_logits_test, _ = model(input_indices, hidden_test)
    # output_logits_test shape: (1, SEQ_LEN, vocab_size)
    _, predicted_indices = torch.max(output_logits_test, dim=2)
    # predicted_indices shape: (1, SEQ_LEN)

# Compare only the valid (non-padded) parts
valid_length = SEQ_LEN - DELAY
predicted_valid_indices = predicted_indices.squeeze()[DELAY:]
target_valid_indices = target_indices[DELAY:]

predicted_valid_chars = "".join(
    [idx_to_char[idx.item()] for idx in predicted_valid_indices]
)
target_valid_chars = "".join([idx_to_char[idx.item()] for idx in target_valid_indices])
input_context_chars = "".join(
    input_chars[:-DELAY]
)  # The chars it should have predicted

print(f"Input context (used for prediction): '{input_context_chars}'")
print(f"Target sequence (delayed input):    '{target_valid_chars}'")
print(f"Predicted sequence:                 '{predicted_valid_chars}'")

correct_predictions = (predicted_valid_indices == target_valid_indices).sum().item()
accuracy = 100 * correct_predictions / valid_length

print(f"\nAccuracy on valid predictions ({valid_length} steps): {accuracy:.2f} %")

if accuracy < 75:  # Set a threshold for struggle
    print("As expected, the vanilla RNN struggles with this longer dependency.")
elif accuracy < 95:
    print("The RNN learned some of the dependency, but not perfectly.")
else:
    print("Surprisingly, the RNN learned the dependency well in this run!")

if __name__ == "__main__":
    """Main execution logic for the RNN long-dependency task."""
    # --- Data Generation ---
    # ... (rest of the __main__ block remains the same)
