# Module 6: Transformers – Architecture of Modern AI

> _Layer upon layer,_  
> _Self-attention weaves meaning,_  
> _Words transform to thought._

## Overview

The Transformer architecture represents a revolutionary advancement in sequence modeling, displacing RNNs as the dominant paradigm in natural language processing and beyond. By relying entirely on attention mechanisms, Transformers process sequences in parallel rather than sequentially, enabling efficient training on massive datasets and capturing complex dependencies between elements regardless of their distance.

This module builds directly on the attention mechanisms explored in Module 5, showing how they can be combined into a powerful architecture that underlies most modern large language models (LLMs) including GPT, BERT, and their descendants.

## Learning Objectives

- Understand the complete Transformer architecture as introduced in "Attention Is All You Need"
- Implement encoder, decoder, and full Transformer models in PyTorch
- Master the core components: multi-head attention, feed-forward networks, positional encoding, and normalization
- Apply Transformers to tasks like machine translation and text generation
- Appreciate how modern language models evolved from the original Transformer design

## Theory

### Transformer Architecture

The Transformer consists of two main components:

1. **Encoder**: Processes the input sequence (e.g., source sentence in translation)

   - Made up of N identical layers, each with:
     - Multi-head self-attention mechanism
     - Position-wise feed-forward network
     - Residual connections and layer normalization

2. **Decoder**: Generates the output sequence (e.g., target translation)
   - Made up of N identical layers, each with:
     - Masked multi-head self-attention mechanism (prevents attending to future positions)
     - Multi-head attention over encoder outputs
     - Position-wise feed-forward network
     - Residual connections and layer normalization

### Key Components

#### Positional Encoding

Since the Transformer contains no recurrence or convolution, we need to inject information about the position of tokens in the sequence:

```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

This creates a unique pattern for each position that the model can learn to interpret.

#### Multi-Head Attention

As covered in Module 5, multi-head attention allows the model to jointly attend to information from different representation subspaces:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
  where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### Feed-Forward Networks

Each encoder and decoder layer contains a fully connected feed-forward network with a ReLU activation:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

#### Layer Normalization

Applied after each sub-layer along with residual connections:

```
LayerNorm(x + Sublayer(x))
```

### Training and Inference

- **Training**: Use teacher forcing with masked attention in the decoder
- **Inference**: Generate tokens autoregressively, using previously generated tokens as input

## Coding Examples

Refer to the `/code` directory for complete examples:

1. `transformer_components.py` - Implementation of key Transformer building blocks
2. `transformer_model.py` - Complete Transformer model implementation
3. `translation_example.py` - Machine translation with Transformers
4. `text_generation.py` - Text generation with a Transformer-based model

## Exercises

In the `/exercises` directory, you'll find:

1. Implement a Transformer encoder from scratch
2. Build a text classification model using only the Transformer encoder
3. Create a simple chatbot using a full Transformer model
4. Experiment with different positional encoding schemes

## Scaling to Modern Language Models

Modern language models like GPT, BERT, and T5 are essentially scaled-up variants of the Transformer architecture with domain-specific modifications:

- **GPT (Decoder-only)**: Uses only the decoder part of the Transformer for autoregressive language modeling
- **BERT (Encoder-only)**: Uses only the encoder for bidirectional representation learning
- **T5 (Encoder-Decoder)**: Uses the full Transformer architecture for sequence-to-sequence tasks

The scaling of these models (to billions of parameters) has led to emergent capabilities not present in smaller models, birthing the field of large language models that power today's AI revolution.

## Haiku Connection

The Transformer architecture parallels the essence of haiku in surprising ways. Just as a haiku transforms simple observations into profound meaning through its structured form, the Transformer converts raw sequences into rich representations through its layered architecture.

Multi-head attention in Transformers resembles the multiple readings a haiku invites—each head attends to different aspects of the input, just as readers might focus on different elements of a haiku in successive readings. The encoders and decoders work together like the complementary phrases in haiku, building toward meaning that emerges from their interaction.

The positional encoding in Transformers provides a sense of order and structure, similar to how the traditional 5-7-5 syllable pattern gives haiku its distinctive rhythm and flow. And just as a skilled haiku poet learns to work within these constraints to achieve maximum expressiveness, the Transformer architecture's defined structure enables powerful language capabilities.

## Resources

- "Attention Is All You Need" by Vaswoski et al. (2017) - The original Transformer paper
- "The Illustrated Transformer" by Jay Alammar - Visual explanation of the architecture
- "The Annotated Transformer" by Harvard NLP - Annotated implementation of the paper
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018)
- "Language Models are Few-Shot Learners" by Brown et al. (2020) - The GPT-3 paper
