"""
Text Generation with Transformers

This module demonstrates how to use a Transformer model for text generation,
particularly focusing on haiku generation to align with the curriculum theme.

The model will be trained on a collection of haikus and then used to generate
new haikus based on prompts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import re
import random
from torch.utils.data import Dataset, DataLoader

# Import Transformer model
from transformer_model import Transformer, create_masks, LabelSmoothing


# Sample haiku collection
HAIKU_COLLECTION = """
ancient pond
a frog leaps in
water's sound

the light of a candle
is transferred to another candle
spring twilight

over the wintry
forest, winds howl in rage
with no leaves to blow

an old silent pond
a frog jumps into the pond
splash! silence again

autumn moonlight
a worm digs silently
into the chestnut

lightning flash
what I thought were faces
are plumes of pampas grass

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

no one travels
along this way but I
this autumn evening

over the wintry
forest, winds howl in rage
with no leaves to blow

first autumn morning
the mirror I stare into
shows my father's face

the lamp once out
cool stars enter
the window frame

with no leaves to blow

in kyoto
hearing the cuckoo's cry
I long for kyoto

the year's first day
thoughts and loneliness
the journey continues

don't weep, insects
lovers, stars themselves
must part

the waterfall
stream seems suspended
clear air

from time to time
the clouds give rest
to the moon-beholders

a summer river being crossed
how pleasing
with sandals in my hands

a mountain village
under the piled-up snow
the sound of water

winter solitude
in a world of one color
the sound of wind

a caterpillar
this deep in fall
still not a butterfly

in the fisherman's hut
mingled with dried shrimp
crickets are chirping

harvest moon
walking around the pond
all night long

the west wind whispered
and touched the eyelids of spring
her eyes, dreaming
"""


class HaikuDataset(Dataset):
    """Dataset for training a model on haikus."""

    def __init__(self, haiku_text, max_length=50):
        """
        Initialize the dataset.

        Args:
            haiku_text: Raw text containing haikus
            max_length: Maximum sequence length
        """
        self.haikus = self._preprocess_haikus(haiku_text)
        self.max_length = max_length

        # Create vocabulary
        self._create_vocab()

        # Tokenize haikus
        self.tokenized_haikus = [self._tokenize_haiku(haiku) for haiku in self.haikus]

    def _preprocess_haikus(self, haiku_text):
        """
        Preprocess the raw haiku text.

        Args:
            haiku_text: Raw text containing haikus

        Returns:
            processed_haikus: List of processed haikus
        """
        # Split by empty lines to get individual haikus
        haikus = [h.strip() for h in haiku_text.split("\n\n") if h.strip()]

        # Clean haikus (remove extra whitespace, etc.)
        processed_haikus = []
        for haiku in haikus:
            # Join lines with special separator
            lines = [line.strip() for line in haiku.split("\n") if line.strip()]
            processed_haiku = " / ".join(lines)
            processed_haikus.append(processed_haiku)

        return processed_haikus

    def _create_vocab(self):
        """Create vocabulary from the haikus."""
        # Get all unique words
        word_set = set()
        for haiku in self.haikus:
            words = haiku.split()
            word_set.update(words)

        # Add special tokens
        special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.word_to_idx = {
            word: idx + len(special_tokens) for idx, word in enumerate(sorted(word_set))
        }

        # Add special token indices
        for idx, token in enumerate(special_tokens):
            self.word_to_idx[token] = idx

        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        # Set special token indices
        self.pad_idx = self.word_to_idx["<pad>"]
        self.sos_idx = self.word_to_idx["<sos>"]
        self.eos_idx = self.word_to_idx["<eos>"]
        self.unk_idx = self.word_to_idx["<unk>"]

        self.vocab_size = len(self.word_to_idx)

    def _tokenize_haiku(self, haiku):
        """
        Convert haiku to token indices.

        Args:
            haiku: Haiku text

        Returns:
            indices: List of token indices
        """
        words = haiku.split()
        indices = [self.word_to_idx.get(word, self.unk_idx) for word in words]

        # Add SOS and EOS tokens
        indices = [self.sos_idx] + indices + [self.eos_idx]

        return indices

    def __len__(self):
        return len(self.tokenized_haikus)

    def __getitem__(self, idx):
        """
        Get a tokenized haiku.

        Args:
            idx: Haiku index

        Returns:
            src: Input sequence (same as target, shifted)
            tgt: Target sequence
        """
        indices = self.tokenized_haikus[idx]

        # For sequence to sequence, input is all but the last token
        src = indices[:-1]

        # Target is all but the first token (SOS)
        tgt = indices[1:]

        # Pad sequences if needed
        if len(src) < self.max_length:
            src = src + [self.pad_idx] * (self.max_length - len(src))
            tgt = tgt + [self.pad_idx] * (self.max_length - len(tgt))
        else:
            src = src[: self.max_length]
            tgt = tgt[: self.max_length]

        return torch.tensor(src), torch.tensor(tgt)

    def decode(self, indices):
        """
        Convert token indices back to text.

        Args:
            indices: List or tensor of token indices

        Returns:
            text: Decoded text
        """
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()

        # Remove special tokens
        words = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx not in [self.pad_idx, self.sos_idx]:
                words.append(self.idx_to_word[idx])

        return " ".join(words)


def create_haiku_transformer(
    dataset, d_model=256, num_heads=8, d_ff=1024, num_layers=4, dropout=0.1
):
    """
    Create a Transformer model for haiku generation.

    Args:
        dataset: Haiku dataset (for vocabulary size)
        d_model: Model's embedding dimension
        num_heads: Number of attention heads
        d_ff: Hidden layer dimension in feed-forward network
        num_layers: Number of encoder/decoder layers
        dropout: Dropout probability

    Returns:
        model: Initialized Transformer model
    """
    vocab_size = dataset.vocab_size

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
    )

    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model: Transformer model
        dataloader: DataLoader with training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        epoch_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        # Create masks
        src_mask, tgt_mask = create_masks(src, tgt)

        # Forward pass
        optimizer.zero_grad()
        output, _, _, _ = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])

        # Calculate loss
        loss = criterion(output.transpose(1, 2), tgt)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate the model.

    Args:
        model: Transformer model
        dataloader: DataLoader with validation data
        criterion: Loss function
        device: Device to validate on

    Returns:
        val_loss: Validation loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt)

            # Forward pass
            output, _, _, _ = model(
                src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1]
            )

            # Calculate loss
            loss = criterion(output.transpose(1, 2), tgt)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=10
):
    """
    Train the model.

    Args:
        model: Transformer model
        train_dataloader: DataLoader with training data
        val_dataloader: DataLoader with validation data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epochs: Number of epochs to train for

    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = validate(model, val_dataloader, criterion, device)

        end_time = time.time()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Time: {end_time - start_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


def plot_losses(train_losses, val_losses):
    """
    Plot training and validation losses.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def generate_haiku(model, dataset, device, prompt=None, max_length=50, temperature=1.0):
    """
    Generate a haiku using the trained model.

    Args:
        model: Trained Transformer model
        dataset: Haiku dataset (for vocabulary)
        device: Device to generate on
        prompt: Optional prompt to start generation (if None, starts with SOS token)
        max_length: Maximum generation length
        temperature: Sampling temperature

    Returns:
        haiku: Generated haiku
        attention_weights: Attention weights used during generation
    """
    model.eval()

    # Start with SOS token if no prompt
    if prompt is None:
        src = torch.tensor([[dataset.sos_idx]], device=device)
    else:
        # Tokenize prompt
        prompt_words = prompt.split()
        prompt_indices = [
            dataset.word_to_idx.get(word, dataset.unk_idx) for word in prompt_words
        ]
        prompt_indices = [dataset.sos_idx] + prompt_indices
        src = torch.tensor([prompt_indices], device=device)

    # Generate sequence
    generated, attention_weights = model.generate(
        src,
        max_length=max_length,
        sos_idx=dataset.sos_idx,
        eos_idx=dataset.eos_idx,
        temperature=temperature,
    )

    # Decode generated sequence
    haiku = dataset.decode(generated[0])

    return haiku, attention_weights


def visualize_generation_attention(attention_weights, haiku, dataset, head_idx=0):
    """
    Visualize attention during generation.

    Args:
        attention_weights: List of attention weight tensors
        haiku: Generated haiku
        dataset: Haiku dataset (for vocabulary)
        head_idx: Head index to visualize
    """
    # Extract tokens from haiku
    tokens = ["<sos>"] + haiku.split()

    # Stack attention weights
    num_steps = len(attention_weights)
    attn_matrices = []

    for i in range(num_steps):
        attn = attention_weights[i][0, head_idx].cpu().numpy()
        attn_matrices.append(attn)

    # Create figure
    fig, axs = plt.subplots(1, num_steps, figsize=(num_steps * 3, 5))
    if num_steps == 1:
        axs = [axs]

    # Plot each step
    for i, attn in enumerate(attn_matrices):
        im = axs[i].imshow(attn, cmap="viridis")

        # Set labels
        axs[i].set_xticks(range(len(tokens[: i + 2])))
        axs[i].set_xticklabels(tokens[: i + 2], rotation=90)
        axs[i].set_yticks([0])
        axs[i].set_yticklabels([tokens[i + 1]])

        axs[i].set_title(f"Step {i+1}")

    fig.suptitle(f"Attention (Head {head_idx+1}) During Haiku Generation", fontsize=16)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


def demo_haiku_generation():
    """
    Demonstrate haiku generation with a Transformer model.
    """
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    print("Creating dataset...")
    dataset = HaikuDataset(HAIKU_COLLECTION)

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    print("Creating model...")
    model = create_haiku_transformer(dataset)
    model.to(device)

    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)

    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(
        model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=20
    )

    # Plot losses
    plot_losses(train_losses, val_losses)

    # Generate haikus
    print("\nGenerating haikus...")

    for temperature in [0.7, 1.0, 1.3]:
        print(f"\nTemperature: {temperature}")
        for _ in range(3):
            haiku, _ = generate_haiku(model, dataset, device, temperature=temperature)
            print(f"{haiku}\n")

    # Generate with prompts
    print("\nGenerating haikus with prompts:")
    prompts = ["autumn moonlight", "ancient pond", "summer river"]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        haiku, attention_weights = generate_haiku(model, dataset, device, prompt=prompt)
        print(f"{haiku}\n")

        # Visualize attention
        visualize_generation_attention(attention_weights, haiku, dataset)

    print("\nHaiku generation demo completed successfully!")


def haiku_example():
    """
    Generate a haiku about transformers... without using a Transformer!
    """
    print("Transformer Haiku:")
    print("Text from text grows,")
    print("Models weave words together,")
    print("New poems emerge.")


if __name__ == "__main__":
    print("Text Generation with Transformers")
    print("=" * 40)

    # Run demo
    demo_haiku_generation()

    # Share a haiku
    haiku_example()
