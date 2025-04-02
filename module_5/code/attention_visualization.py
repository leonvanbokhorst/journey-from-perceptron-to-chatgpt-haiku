"""
Attention Visualization Tool

This module provides tools for visualizing attention patterns in neural networks,
particularly those involving attention mechanisms like:
1. Self-attention in transformers
2. Encoder-decoder attention in seq2seq models
3. Multi-head attention patterns

The visualizations help to interpret how models process information and make decisions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
import colorsys
import io
from PIL import Image
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Set style
plt.style.use("ggplot")
sns.set(font_scale=1.2)


def create_attention_map(
    attention_weights,
    input_tokens,
    output_tokens=None,
    cmap="viridis",
    title="Attention Weights",
):
    """
    Create a heatmap visualization of attention weights.

    Args:
        attention_weights: numpy array of shape (output_seq_len, input_seq_len)
        input_tokens: list of input tokens
        output_tokens: list of output tokens (for encoder-decoder attention)
        cmap: colormap to use
        title: title for the plot

    Returns:
        fig: matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    if output_tokens is not None:
        # For encoder-decoder attention
        sns.heatmap(
            attention_weights,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            cmap=cmap,
            ax=ax,
        )
        ax.set_xlabel("Input Sequence")
        ax.set_ylabel("Output Sequence")
    else:
        # For self-attention
        sns.heatmap(
            attention_weights,
            xticklabels=input_tokens,
            yticklabels=input_tokens,
            cmap=cmap,
            ax=ax,
        )
        ax.set_xlabel("Sequence")
        ax.set_ylabel("Sequence")

    ax.set_title(title)
    plt.tight_layout()

    return fig


def visualize_multihead_attention(
    attention_weights, tokens, title="Multi-Head Attention", num_heads=None
):
    """
    Visualize multi-head attention patterns from a transformer model.

    Args:
        attention_weights: numpy array of shape (num_heads, seq_len, seq_len)
                          or torch tensor of same shape
        tokens: list of tokens corresponding to the sequence
        title: title for the plot
        num_heads: number of heads to visualize (default: all)
    """
    # Convert to numpy if tensor
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()

    # Determine number of heads to visualize
    if num_heads is None:
        num_heads = attention_weights.shape[0]
    else:
        num_heads = min(num_heads, attention_weights.shape[0])

    # Calculate grid dimensions
    n_cols = min(4, num_heads)
    n_rows = (num_heads + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    # Create a common color scale across all heads
    vmin = attention_weights[:num_heads].min()
    vmax = attention_weights[:num_heads].max()

    # Plot each head
    for i in range(num_heads):
        head_weights = attention_weights[i]
        sns.heatmap(
            head_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=axs[i],
        )
        axs[i].set_title(f"Head {i+1}")

        # Only add x labels to bottom row
        if i < len(axs) - n_cols:
            axs[i].set_xlabel("")
            axs[i].set_xticklabels([])

        # Only add y labels to left column
        if i % n_cols != 0:
            axs[i].set_ylabel("")
            axs[i].set_yticklabels([])

    # Hide empty subplots
    for i in range(num_heads, len(axs)):
        axs[i].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    return fig


def create_interactive_attention_viewer(attention_weights, tokens, num_heads):
    """
    Create an interactive widget for exploring attention patterns in multi-head attention.

    Args:
        attention_weights: numpy array of shape (num_layers, num_heads, seq_len, seq_len)
                          or torch tensor of same shape
        tokens: list of tokens corresponding to the sequence
        num_heads: number of heads per layer

    Note: This function is designed to be used in Jupyter notebooks
    """
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()

    num_layers = attention_weights.shape[0]

    # Create widgets
    layer_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=num_layers - 1,
        step=1,
        description="Layer:",
        continuous_update=False,
    )

    head_dropdown = widgets.Dropdown(
        options=[("All Heads", -1)] + [(f"Head {i}", i) for i in range(num_heads)],
        value=-1,
        description="Head:",
    )

    colormap_dropdown = widgets.Dropdown(
        options=[
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "coolwarm",
            "RdBu_r",
        ],
        value="viridis",
        description="Colormap:",
    )

    # Define update function
    def update_plot(layer, head, cmap):
        plt.figure(figsize=(10, 8))

        if head == -1:  # All heads
            # Use the multi-head visualization
            fig = visualize_multihead_attention(
                attention_weights[layer],
                tokens,
                title=f"Layer {layer+1} - All Heads",
                num_heads=num_heads,
            )
        else:  # Single head
            # Use the single-head visualization
            fig = create_attention_map(
                attention_weights[layer, head],
                tokens,
                cmap=cmap,
                title=f"Layer {layer+1} - Head {head+1}",
            )

        plt.close(fig)  # Close the figure to avoid displaying it twice
        display(fig)

    # Create interactive widget
    interactive_plot = widgets.interactive(
        update_plot, layer=layer_slider, head=head_dropdown, cmap=colormap_dropdown
    )

    return interactive_plot


def visualize_attention_on_text(text, attention_weights, tokenizer=None):
    """
    Visualize token-level attention by highlighting text based on attention scores.

    Args:
        text: original text string
        attention_weights: numpy array of attention weights for each token
        tokenizer: optional tokenizer function to split text into tokens
                  (if None, simple space-based tokenization is used)

    Returns:
        HTML display object with highlighted text
    """
    from IPython.display import HTML

    # Tokenize text if no tokenizer provided
    if tokenizer is None:
        tokens = text.split()
    else:
        tokens = tokenizer(text)

    # Ensure attention weights match token count
    assert len(tokens) == len(
        attention_weights
    ), "Number of tokens must match attention weights"

    # Normalize attention weights to range [0, 1]
    normalized_weights = (attention_weights - attention_weights.min()) / (
        attention_weights.max() - attention_weights.min()
    )

    # Generate HTML with heatmap coloring
    html = []
    for token, weight in zip(tokens, normalized_weights):
        # Convert to RGB for better visualization
        # Using a perceptually uniform colormap (viridis-like)
        r, g, b = colorsys.hsv_to_rgb(0.7 * (1 - weight), 0.8, 0.9)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)

        # Convert to hex color code
        color = f"#{r:02x}{g:02x}{b:02x}"

        # Create span element with background color
        html.append(
            f'<span style="background-color: {color}; padding: 0.2em; border-radius: 0.2em;">{token}</span>'
        )

    # Join spans with spaces
    html_string = " ".join(html)

    return HTML(f"<div style='line-height: 2em; font-size: 1.2em;'>{html_string}</div>")


def animate_attention_heads(
    attention_weights, tokens, interval=500, fps=5, title="Attention Heads Animation"
):
    """
    Create an animation showing all attention heads one by one.

    Args:
        attention_weights: numpy array of shape (num_heads, seq_len, seq_len)
                           or torch tensor of same shape
        tokens: list of tokens corresponding to the sequence
        interval: time between frames in milliseconds
        fps: frames per second when saving as video
        title: title for the animation

    Returns:
        matplotlib animation object
    """
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()

    num_heads = attention_weights.shape[0]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tight_layout()

    # Find global min and max for consistent color scale
    vmin = attention_weights.min()
    vmax = attention_weights.max()

    # Function to update each frame
    def update(frame):
        ax.clear()
        i = frame % num_heads

        # Create heatmap
        sns.heatmap(
            attention_weights[i],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )

        ax.set_title(f"{title} - Head {i+1}/{num_heads}")

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_heads, interval=interval, blit=False
    )

    plt.close()  # Close the figure to prevent double display

    return ani


def compare_attention_patterns(attention_patterns, labels, tokens, suptitle=None):
    """
    Compare different attention patterns side by side.

    Args:
        attention_patterns: list of attention weight matrices (numpy arrays or tensors)
        labels: list of labels for each pattern
        tokens: list of tokens corresponding to the sequence
        suptitle: overall title for the figure

    Returns:
        matplotlib figure
    """
    n_patterns = len(attention_patterns)

    # Create figure and axes
    fig, axes = plt.subplots(1, n_patterns, figsize=(n_patterns * 5, 5))
    if n_patterns == 1:
        axes = [axes]

    # Find global min and max for consistent color scale
    all_patterns = [
        p.detach().cpu().numpy() if torch.is_tensor(p) else p
        for p in attention_patterns
    ]
    vmin = min([p.min() for p in all_patterns])
    vmax = max([p.max() for p in all_patterns])

    # Plot each pattern
    for i, (pattern, label) in enumerate(zip(attention_patterns, labels)):
        if torch.is_tensor(pattern):
            pattern = pattern.detach().cpu().numpy()

        sns.heatmap(
            pattern,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=axes[i],
        )

        axes[i].set_title(label)

        # Adjust labels for cleaner display
        if i > 0:
            axes[i].set_ylabel("")

    if suptitle:
        plt.suptitle(suptitle, fontsize=16)

    plt.tight_layout()

    return fig


def haiku_attention_demo():
    """
    Demonstrate attention visualization on a haiku example.
    """
    # Define sample haiku
    haiku = "old pond frog jumps in water sound"
    tokens = haiku.split()
    num_tokens = len(tokens)

    # Create synthetic attention patterns

    # 1. Self-attention synthetic pattern (focused on related words)
    self_attn = np.zeros((num_tokens, num_tokens))
    # pond-water relationship
    pond_idx = tokens.index("pond")
    water_idx = tokens.index("water")
    self_attn[pond_idx, water_idx] = 0.8
    self_attn[water_idx, pond_idx] = 0.7

    # frog-jumps relationship
    frog_idx = tokens.index("frog")
    jumps_idx = tokens.index("jumps")
    self_attn[frog_idx, jumps_idx] = 0.9
    self_attn[jumps_idx, frog_idx] = 0.6

    # Add diagonal (self-attention)
    np.fill_diagonal(self_attn, 0.3)

    # Normalize to create a proper attention pattern
    row_sums = self_attn.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    self_attn = self_attn / row_sums

    # 2. Create several attention heads with different patterns
    heads = []

    # Head 1: Focus on subject-verb relationships
    head1 = np.zeros((num_tokens, num_tokens))
    head1[frog_idx, jumps_idx] = 1.0  # frog -> jumps
    np.fill_diagonal(head1, 0.2)
    row_sums = head1.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    head1 = head1 / row_sums
    heads.append(head1)

    # Head 2: Focus on object relationships
    head2 = np.zeros((num_tokens, num_tokens))
    head2[jumps_idx, tokens.index("in")] = 0.7  # jumps -> in
    head2[tokens.index("in"), water_idx] = 0.8  # in -> water
    np.fill_diagonal(head2, 0.2)
    row_sums = head2.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    head2 = head2 / row_sums
    heads.append(head2)

    # Head 3: Focus on scene elements
    head3 = np.zeros((num_tokens, num_tokens))
    head3[pond_idx, water_idx] = 0.6  # pond -> water
    head3[water_idx, tokens.index("sound")] = 0.7  # water -> sound
    np.fill_diagonal(head3, 0.2)
    row_sums = head3.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    head3 = head3 / row_sums
    heads.append(head3)

    # Head 4: Attending to next token (sequential attention)
    head4 = np.zeros((num_tokens, num_tokens))
    for i in range(num_tokens - 1):
        head4[i, i + 1] = 0.8
    np.fill_diagonal(head4, 0.2)
    row_sums = head4.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    head4 = head4 / row_sums
    heads.append(head4)

    # Stack heads into a multi-head attention tensor
    multi_head_attn = np.stack(heads)

    # Visualize the attention patterns
    print("Haiku:", haiku)
    print("\nSingle Attention Pattern:")
    fig1 = create_attention_map(
        self_attn, tokens, title="Synthetic Self-Attention Pattern"
    )

    print("\nMulti-Head Attention Patterns:")
    fig2 = visualize_multihead_attention(
        multi_head_attn, tokens, title="Synthetic Multi-Head Attention"
    )

    # Compare different heads
    print("\nComparison of Different Attention Heads:")
    head_labels = [
        "Head 1: Subject-Verb Attention",
        "Head 2: Object Relationships",
        "Head 3: Scene Elements",
        "Head 4: Sequential Attention",
    ]
    fig3 = compare_attention_patterns(
        heads, head_labels, tokens, suptitle="Different Attention Patterns in Haiku"
    )

    # Display visualizations
    plt.show()

    # Create a haiku about attention itself
    print("\nAttention Haiku:")
    print("Focus shifts to words")
    print("Each token speaks to others")
    print("Patterns emerge clear")

    return fig1, fig2, fig3


if __name__ == "__main__":
    print("Attention Visualization Demo")
    print("=" * 40)

    # Run the haiku demo
    haiku_attention_demo()

    print(
        "\nNote: For interactive visualizations, use this module in a Jupyter notebook."
    )
