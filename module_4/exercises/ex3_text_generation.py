"""
Exercise 4.3: Haiku Generation with Character-Level RNNs

In this exercise, you'll implement a character-level LSTM model to generate haikus.
Haikus are short poems with a specific structure (typically 3 lines with 5-7-5 syllables),
making them perfect for experimenting with sequence generation.

Learning Objectives:
1. Implement a character-level language model using LSTMs
2. Understand how to sample from the model's predicted distributions
3. Learn how to control the generation process with temperature
4. Explore creative text generation and model the structure of haikus
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import string
import time
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sample haiku corpus - you can expand this or load from a file
HAIKU_CORPUS = """
ancient pond
a frog leaps in
water's sound

the light of a candle
is transferred to another candle
spring twilight

over-ripe sushi
the master
is full of regret

an old silent pond
a frog jumps into the pond
splash! silence again

autumn moonlight
a worm digs silently
into the chestnut

in the twilight rain
these brilliant-hued hibiscus
a lovely sunset

the first cold shower
even the monkey seems to want
a little coat of straw

won't you come and see
loneliness? just one leaf
from the kiri tree

temple bells die out
the fragrant blossoms remain
a perfect evening

the lamp once out
cool stars enter
the window frame

a summer river being crossed
how pleasing
with sandals in my hands

lightning flash
what I thought were faces
are plumes of pampas grass

no one travels
along this way but I
this autumn evening

as the wind
does with the cloud
you swept me off my feet

in cherry blossom shadows
no one is
a stranger

a giant firefly
that way, this way, that way, this
and it passes by

sun setting
pealing an orange
in cold hands

a bitter morning
sparrows sitting together
without any necks

out of the water
and breathing in the air
the dewdrops glisten

fresh morning dew
flowing down the blade
of a grass

spring rain
a child squeezes their toes
into the mud

summer grasses
all that remains 
of soldiers' dreams

first fall morning
the mirror I stare into
shows my father's face

road-side violets
pickles are cooking
in the house

from time to time
the clouds give rest
to the moon gazers
"""


class CharLSTM(nn.Module):
    """Character-level LSTM for text generation"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # For generation task, we use softmax to get probabilities
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, hidden=None):
        # Initialize hidden state and cell state if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0))

        # Embedding
        embeds = self.embedding(x)

        # Forward propagate through LSTM
        out, hidden = self.lstm(embeds, hidden)

        # Pass through fully connected layer
        out = self.fc(out)

        # Apply log softmax
        out = self.softmax(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)


class HaikuDataset:
    """Dataset for character-level haiku generation"""

    def __init__(self, corpus, seq_length=50):
        self.corpus = corpus
        self.seq_length = seq_length

        # Process corpus to create character-to-integer mapping
        self.chars = sorted(list(set(corpus)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Encode corpus
        self.encoded_corpus = [self.char_to_idx[ch] for ch in corpus]

        # Prepare data
        self.prepare_data()

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Corpus length: {len(corpus)} characters")
        print(f"Number of sequences: {len(self.X)}")

    def prepare_data(self):
        """Prepare input-target pairs for training"""
        self.X = []  # Input sequences
        self.y = []  # Target sequences

        # Create sequences
        for i in range(0, len(self.encoded_corpus) - self.seq_length - 1, 1):
            input_seq = self.encoded_corpus[i : i + self.seq_length]
            target_seq = self.encoded_corpus[i + 1 : i + self.seq_length + 1]

            self.X.append(input_seq)
            self.y.append(target_seq)

        # Convert to PyTorch tensors
        self.X = torch.tensor(self.X, dtype=torch.long, device=device)
        self.y = torch.tensor(self.y, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(
    model,
    dataset,
    epochs=50,
    batch_size=32,
    lr=0.001,
    print_every=10,
    generate_every=0,
    seed_text=None,
):
    """
    Train the character-level model on the haiku dataset.

    Args:
        model: The LSTM model
        dataset: The HaikuDataset
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        print_every: How often to print progress
        generate_every: How often to generate sample text (0 to disable)
        seed_text: Seed text for generation during training

    Returns:
        model: Trained model
        losses: List of training losses
    """
    model = model.to(device)

    # NLLLoss (works with LogSoftmax outputs)
    criterion = nn.NLLLoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For tracking progress
    losses = []

    # Number of batches
    n_batches = len(dataset) // batch_size

    # Time the training
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Generate random indices for this epoch
        indices = torch.randperm(len(dataset))

        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(dataset))
            batch_indices = indices[start_idx:end_idx]

            # Get batch data
            X_batch, y_batch = dataset.X[batch_indices], dataset.y[batch_indices]

            # Initialize hidden state
            hidden = model.init_hidden(len(batch_indices))

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, hidden = model(X_batch, hidden)

            # Reshape for loss calculation
            outputs = outputs.view(-1, dataset.vocab_size)
            y_batch = y_batch.view(-1)

            # Calculate loss
            loss = criterion(outputs, y_batch)

            # Backward pass and optimize
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            # Record loss
            epoch_loss += loss.item()

        # Calculate average loss
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Generate sample text
        if generate_every > 0 and (epoch + 1) % generate_every == 0 and seed_text:
            print("\nGenerating sample text:")
            text = generate_text(
                model, dataset, seed_text, max_length=100, temperature=0.5
            )
            print(text)
            print()
            model.train()

    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    return model, losses


def generate_text(model, dataset, seed_text, max_length=200, temperature=1.0):
    """
    Generate text using the trained model.

    Args:
        model: Trained LSTM model
        dataset: HaikuDataset with character mappings
        seed_text: Initial text to seed the generation
        max_length: Maximum length of generated text
        temperature: Controls randomness (lower = more deterministic)

    Returns:
        generated_text: The generated text
    """
    model.eval()

    # Convert seed text to tensor
    seed_encoded = [
        dataset.char_to_idx.get(ch, dataset.char_to_idx["\n"]) for ch in seed_text
    ]

    # If seed is shorter than sequence length, pad with newlines
    if len(seed_encoded) < dataset.seq_length:
        seed_encoded = [dataset.char_to_idx["\n"]] * (
            dataset.seq_length - len(seed_encoded)
        ) + seed_encoded

    # If seed is longer, take the most recent characters
    if len(seed_encoded) > dataset.seq_length:
        seed_encoded = seed_encoded[-dataset.seq_length :]

    input_tensor = torch.tensor([seed_encoded], dtype=torch.long, device=device)

    # Initialize hidden state
    hidden = model.init_hidden(1)

    # The generated text starts with the seed text
    generated_text = seed_text

    # Generate new characters
    with torch.no_grad():
        for i in range(max_length):
            # Forward pass
            output, hidden = model(input_tensor, hidden)

            # Get the predicted logits for the next character (last time step)
            logits = output[0, -1].cpu().numpy()

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Convert logits to probabilities
            exp_logits = np.exp(logits)
            probabilities = exp_logits / np.sum(exp_logits)

            # Sample from the distribution
            next_char_idx = np.random.choice(len(probabilities), p=probabilities)

            # Convert to character and add to generated text
            next_char = dataset.idx_to_char[next_char_idx]
            generated_text += next_char

            # Update input for next step
            input_tensor = torch.cat(
                [
                    input_tensor[:, 1:],
                    torch.tensor([[next_char_idx]], dtype=torch.long, device=device),
                ],
                dim=1,
            )

    return generated_text


def plot_loss(losses):
    """Plot training loss over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.show()


def plot_character_frequencies(texts, dataset):
    """Compare character frequencies between original corpus and generated text"""
    # Get character frequencies in original corpus
    original_freqs = {}
    for char in dataset.corpus:
        if char in original_freqs:
            original_freqs[char] += 1
        else:
            original_freqs[char] = 1

    # Get character frequencies in generated texts
    generated_freqs = {}
    for text in texts:
        for char in text:
            if char in generated_freqs:
                generated_freqs[char] += 1
            else:
                generated_freqs[char] = 1

    # Normalize frequencies
    total_orig = sum(original_freqs.values())
    total_gen = sum(generated_freqs.values())

    norm_orig = {c: count / total_orig for c, count in original_freqs.items()}
    norm_gen = {c: count / total_gen for c, count in generated_freqs.items()}

    # Get common characters for comparison
    common_chars = sorted(set(norm_orig.keys()) & set(norm_gen.keys()))

    # Create lists for plotting
    orig_vals = [norm_orig.get(c, 0) for c in common_chars]
    gen_vals = [norm_gen.get(c, 0) for c in common_chars]

    # Plot character frequencies
    plt.figure(figsize=(15, 6))

    # Create x positions for bars
    x = np.arange(len(common_chars))
    width = 0.35

    plt.bar(x - width / 2, orig_vals, width, label="Original Corpus")
    plt.bar(x + width / 2, gen_vals, width, label="Generated Text")

    plt.xlabel("Characters")
    plt.ylabel("Normalized Frequency")
    plt.title("Character Frequency Comparison")
    plt.xticks(x, common_chars)
    plt.legend()

    plt.tight_layout()
    plt.show()


def analyze_haiku_structure(haikus):
    """
    Analyze the structure of generated haikus.

    Args:
        haikus: List of generated haikus
    """
    # Count lines per haiku
    lines_per_haiku = []
    for haiku in haikus:
        # Split by newline and count non-empty lines
        lines = [line for line in haiku.split("\n") if line.strip()]
        lines_per_haiku.append(len(lines))

    # Plot distribution of lines per haiku
    plt.figure(figsize=(10, 6))
    plt.hist(lines_per_haiku, bins=range(1, max(lines_per_haiku) + 2), alpha=0.7)
    plt.xlabel("Number of Lines")
    plt.ylabel("Count")
    plt.title("Distribution of Lines per Generated Haiku")
    plt.grid(True)
    plt.xticks(range(1, max(lines_per_haiku) + 1))
    plt.show()

    # Analyze line lengths
    all_line_lengths = []
    for haiku in haikus:
        lines = [line for line in haiku.split("\n") if line.strip()]
        for line in lines:
            all_line_lengths.append(len(line))

    # Plot distribution of line lengths
    plt.figure(figsize=(10, 6))
    plt.hist(all_line_lengths, bins=20, alpha=0.7)
    plt.xlabel("Line Length (characters)")
    plt.ylabel("Count")
    plt.title("Distribution of Line Lengths in Generated Haikus")
    plt.grid(True)
    plt.show()


def generate_haikus(model, dataset, seeds, temperature=0.5, count=5):
    """
    Generate multiple haikus with different seeds and temperatures.

    Args:
        model: Trained LSTM model
        dataset: HaikuDataset with character mappings
        seeds: List of seed texts
        temperature: Temperature for generation or list of temperatures
        count: Number of haikus to generate per seed/temperature combination

    Returns:
        generated_haikus: List of generated haikus
    """
    generated_haikus = []

    # Convert temperature to list if it's a single value
    if not isinstance(temperature, list):
        temperature = [temperature]

    for seed in seeds:
        for temp in temperature:
            print(f"\nGenerating haikus with seed: '{seed}' and temperature: {temp}")

            for i in range(count):
                # Generate text
                haiku = generate_text(
                    model, dataset, seed, max_length=100, temperature=temp
                )

                # Clean generated text
                haiku = haiku.strip()

                # Print the generated haiku
                print(f"\nHaiku {i+1}:")
                print(haiku)
                print("-" * 30)

                generated_haikus.append(haiku)

    return generated_haikus


def save_model(model, dataset, path="haiku_lstm_model.pth"):
    """Save the trained model and character mappings"""
    model_state = {
        "model_state_dict": model.state_dict(),
        "char_to_idx": dataset.char_to_idx,
        "idx_to_char": dataset.idx_to_char,
        "vocab_size": dataset.vocab_size,
        "seq_length": dataset.seq_length,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
    }

    torch.save(model_state, path)
    print(f"Model saved to {path}")


def load_model(path="haiku_lstm_model.pth"):
    """Load a trained model and character mappings"""
    if not os.path.exists(path):
        print(f"Model file {path} does not exist.")
        return None, None

    model_state = torch.load(path, map_location=device)

    # Create model with saved parameters
    model = CharLSTM(
        input_size=model_state["vocab_size"],
        hidden_size=model_state["hidden_size"],
        output_size=model_state["vocab_size"],
        num_layers=model_state["num_layers"],
    )

    # Load model weights
    model.load_state_dict(model_state["model_state_dict"])
    model.to(device)

    # Create a simple dataset-like object with character mappings
    class SimpleDataset:
        def __init__(self, model_state):
            self.char_to_idx = model_state["char_to_idx"]
            self.idx_to_char = model_state["idx_to_char"]
            self.vocab_size = model_state["vocab_size"]
            self.seq_length = model_state["seq_length"]

    dataset = SimpleDataset(model_state)

    print(f"Model loaded from {path}")
    return model, dataset


def main():
    print("Exercise 4.3: Haiku Generation with Character-Level RNNs\n")

    # Parameters
    hidden_size = 128
    num_layers = 2
    seq_length = 50
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    dropout = 0.3

    # Create dataset
    dataset = HaikuDataset(HAIKU_CORPUS, seq_length=seq_length)

    # Check if saved model exists
    model_path = "haiku_lstm_model.pth"
    if os.path.exists(model_path):
        print("Found saved model, loading...")
        model, dataset = load_model(model_path)
    else:
        # Create model
        print("Creating new model...")
        model = CharLSTM(
            input_size=dataset.vocab_size,
            hidden_size=hidden_size,
            output_size=dataset.vocab_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Print model architecture
        print(model)

        # Train model
        print("\nTraining model...")
        model, losses = train_model(
            model=model,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            print_every=10,
            generate_every=25,
            seed_text="autumn",
        )

        # Plot training loss
        plot_loss(losses)

        # Save model
        save_model(model, dataset, model_path)

    # Generate haikus with different seeds and temperatures
    print("\nGenerating haikus...")
    seeds = ["spring", "summer", "autumn", "winter", "moon"]
    temperatures = [0.2, 0.5, 1.0]

    generated_haikus = generate_haikus(model, dataset, seeds, temperatures, count=2)

    # Analyze haiku structure
    analyze_haiku_structure(generated_haikus)

    # Plot character frequencies
    plot_character_frequencies(generated_haikus, dataset)

    print("\nExercise Tasks:")
    print(
        "1. Try different hyperparameters (hidden size, layers, dropout) and observe the effect on generation."
    )
    print(
        "2. Experiment with different temperatures and analyze how they affect text diversity."
    )
    print(
        "3. Implement a syllable counter to analyze if the model captures the 5-7-5 syllable structure."
    )
    print(
        "4. Collect more haikus to expand the training corpus. Does this improve generation quality?"
    )
    print(
        "5. Modify the model to enforce the 3-line structure of haikus during generation."
    )


if __name__ == "__main__":
    main()
