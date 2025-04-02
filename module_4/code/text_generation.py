"""
Character-level Text Generation with LSTM - Module 4
---------------------------------------------------
Implementing a character-level language model using LSTM
for generating haiku-inspired text.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import string

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Sample haiku corpus for training
HAIKU_CORPUS = """
ancient pond
a frog leaps in
water's sound

the first cold shower
even the monkey seems to want
a little coat of straw

over-ripe sushi
the master
is full of regret

in the cicada's cry
no sign can foretell
how soon it must die

calm and serene
the sound of a cicada
penetrates the rock

clouds appear
and bring to men a chance to rest
from looking at the moon

the light of a candle
is transferred to another candle
spring twilight

no one travels
along this way but I
this autumn evening

the crow has flown away
swaying in the evening sun
a leafless tree

winter seclusion
listening, that evening
to the rain in the mountain

an old silent pond
a frog jumps into the pond
splash! silence again

lighting one candle
with another candle
spring evening

temple bells die out
the fragrant blossoms remain
a perfect evening

in the twilight rain
these brilliant-hued hibiscus
a lovely sunset

the wind of Mt. Fuji
I've brought on my fan
a gift from Edo

fish shop
how they lie
with their white bellies

for love and for hate
I swat a fly and offer it
to an ant

don't weep, insects
lovers, stars themselves
must part

the stars on the pond
float in the bowl
of dark water

old dark sleepy pool
quick unexpected frog
goes plop! watersplash

the falling flower
I saw drift back to the branch
was a butterfly
"""


class CharDataset(Dataset):
    """
    Dataset for character-level language modeling
    """

    def __init__(self, text, seq_length):
        """
        Initialize the dataset

        Args:
            text: The text corpus to use
            seq_length: Length of sequences for training
        """
        self.text = text
        self.seq_length = seq_length

        # Create character to integer mapping
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.n_chars = len(chars)

        # Create sequences
        self.data = []
        self.targets = []
        for i in range(0, len(text) - seq_length):
            self.data.append(text[i : i + seq_length])
            self.targets.append(text[i + 1 : i + seq_length + 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert characters to indices
        data_indices = [self.char_to_idx[ch] for ch in self.data[idx]]
        target_indices = [self.char_to_idx[ch] for ch in self.targets[idx]]

        # Convert to PyTorch tensors
        return (
            torch.tensor(data_indices, dtype=torch.long),
            torch.tensor(target_indices, dtype=torch.long),
        )

    def get_vocab_size(self):
        return self.n_chars

    def decode(self, indices):
        """Convert indices back to characters"""
        return "".join([self.idx_to_char[idx.item()] for idx in indices])


class CharLSTM(nn.Module):
    """
    Character-level LSTM model for text generation
    """

    def __init__(
        self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0.5
    ):
        """
        Initialize the LSTM model

        Args:
            vocab_size: Number of unique characters
            embedding_size: Size of character embeddings
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability for regularization
        """
        super(CharLSTM, self).__init__()

        # Model architecture
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through the network

        Args:
            x: Input sequence [batch_size, seq_length]
            hidden: Initial hidden state (optional)

        Returns:
            output: Predictions for each character [batch_size, seq_length, vocab_size]
            hidden: Final hidden state
        """
        # Get batch size and sequence length
        batch_size, seq_length = x.size()

        # Embed the input
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_size]

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # Forward through LSTM
        output, hidden = self.lstm(embedded, hidden)

        # Apply dropout and project to vocabulary size
        output = self.dropout(output)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state with zeros"""
        # Return a tuple of (hidden_state, cell_state)
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
        )

    def generate(self, dataset, seed_text, gen_length=100, temperature=1.0):
        """
        Generate text using the trained model

        Args:
            dataset: The character dataset used for training
            seed_text: Initial text to start generation
            gen_length: Number of characters to generate
            temperature: Controls randomness (higher = more random)

        Returns:
            generated_text: The generated text
        """
        self.eval()  # Set to evaluation mode

        # Convert seed text to indices
        if len(seed_text) == 0:
            # If no seed text, randomly select a character
            seed_text = random.choice(list(dataset.char_to_idx.keys()))

        chars = [ch for ch in seed_text]
        char_indices = [dataset.char_to_idx[ch] for ch in chars]

        # If seed text is shorter than sequence length, pad it
        # Alternatively, we could just use the available seed text length
        if len(char_indices) < dataset.seq_length:
            char_indices = [dataset.char_to_idx[" "]] * (
                dataset.seq_length - len(char_indices)
            ) + char_indices

        # If seed text is longer, truncate it
        if len(char_indices) > dataset.seq_length:
            char_indices = char_indices[-dataset.seq_length :]

        generated = chars

        # Move to device
        device = next(self.parameters()).device
        x = torch.tensor([char_indices], dtype=torch.long).to(device)
        hidden = None

        # Generate characters
        with torch.no_grad():
            for i in range(gen_length):
                # Forward pass
                output, hidden = self(x, hidden)

                # Get the prediction for the last character in the sequence
                output = output[:, -1, :].squeeze(0)

                # Apply temperature to the logits
                output = output / temperature

                # Convert to probabilities and sample
                probs = torch.softmax(output, dim=-1)
                next_char_idx = torch.multinomial(probs, 1).item()

                # Add the new character
                generated.append(dataset.idx_to_char[next_char_idx])

                # Update the input sequence
                x = torch.cat(
                    [x[:, 1:], torch.tensor([[next_char_idx]], device=device)], dim=1
                )

        return "".join(generated)


def train_model(model, dataloader, epochs, lr=0.001, clip=5.0):
    """
    Train the LSTM model

    Args:
        model: The LSTM model
        dataloader: DataLoader with training data
        epochs: Number of training epochs
        lr: Learning rate
        clip: Gradient clipping value

    Returns:
        losses: List of average losses per epoch
    """
    device = next(model.parameters()).device

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        hidden = None

        # Track time
        start_time = time.time()

        for batch_idx, (data, targets) in enumerate(dataloader):
            # Move to device
            data, targets = data.to(device), targets.to(device)

            # Initialize hidden state at the start of each batch
            hidden = tuple([h.detach() for h in hidden]) if hidden else None

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output, hidden = model(data, hidden)

            # Reshape output for loss calculation
            output = output.view(-1, model.vocab_size)
            targets = targets.view(-1)

            # Calculate loss
            loss = criterion(output, targets)
            epoch_loss += loss.item()

            # Backward pass and optimize
            loss.backward()

            # Clip gradients to prevent explosion
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Update parameters
            optimizer.step()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}"
                )

        # Calculate average loss for this epoch
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} completed | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s"
        )

        # Generate a sample every epoch
        seed = "the old pond"
        generated = model.generate(
            dataloader.dataset, seed, gen_length=50, temperature=0.7
        )
        print(f"\nSample generation:\n{generated}\n")

    return losses


def plot_loss(losses):
    """Plot the training loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def generate_haiku_samples(model, dataset, n_samples=5, temperature=0.7):
    """Generate multiple haiku samples"""
    print("\n== Generated Haiku Samples ==\n")

    seeds = [
        "an old pond",
        "autumn leaves",
        "summer moon",
        "winter snow",
        "spring breeze",
    ]

    for i in range(n_samples):
        seed = seeds[i] if i < len(seeds) else random.choice(seeds)
        generated = model.generate(
            dataset, seed, gen_length=70, temperature=temperature
        )

        # Try to format as 3-line haiku by finding line breaks
        lines = generated.strip().split("\n")
        if len(lines) >= 3:
            formatted = "\n".join(lines[:3])
        else:
            # If not enough line breaks, add them approximately
            words = generated.split()
            if len(words) >= 5:
                line1 = " ".join(words[:5])
                line2 = (
                    " ".join(words[5:12]) if len(words) >= 12 else " ".join(words[5:])
                )
                line3 = " ".join(words[12:]) if len(words) >= 12 else ""
                formatted = f"{line1}\n{line2}\n{line3}"
            else:
                formatted = generated

        print(f"Seed: '{seed}'\n{formatted}\n")


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    SEQ_LENGTH = 30
    EMBEDDING_SIZE = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    EPOCHS = 50

    # Clean the text a bit
    text = HAIKU_CORPUS.strip()

    # Create dataset
    dataset = CharDataset(text, SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Print some info
    print(f"Corpus size: {len(text)} characters")
    print(f"Vocabulary size: {dataset.get_vocab_size()} characters")
    print(f"Number of sequences: {len(dataset)}")

    # Sample a batch
    data, target = next(iter(dataloader))
    print(f"Sample input: {dataset.decode(data[0])}")
    print(f"Sample target: {dataset.decode(target[0])}")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CharLSTM(
        vocab_size=dataset.get_vocab_size(),
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    print(model)

    # Train the model
    losses = train_model(
        model=model, dataloader=dataloader, epochs=EPOCHS, lr=LEARNING_RATE
    )

    # Plot training loss
    plot_loss(losses)

    # Generate some haiku
    generate_haiku_samples(model, dataset, n_samples=5)

    # Generate with different temperatures
    print("\n== Temperature Experiment ==\n")
    seed = "autumn leaves"

    for temp in [0.3, 0.7, 1.0, 1.5]:
        print(f"\nTemperature: {temp}")
        generated = model.generate(dataset, seed, gen_length=70, temperature=temp)
        print(generated)

    # Haiku moment to finish
    print("\nHaiku:")
    print("Through the ancient gates")
    print("Neural pathways remember")
    print("Forgotten patterns")


if __name__ == "__main__":
    main()
