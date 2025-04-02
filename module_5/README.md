# Module 5: Attention Mechanisms – Focus and Context

> _Mind's eye wanders, stops,_  
> _Important details stand out,_  
> _Seeing what matters._

## Overview

As we progress from RNNs, we encounter a pivotal advancement: attention mechanisms. These techniques address a fundamental limitation of sequence models by allowing them to focus on specific parts of the input sequence rather than compressing all information into a fixed-size vector. Attention has revolutionized sequence modeling, particularly in machine translation, question answering, and eventually led to the Transformer architecture.

## Learning Objectives

- Understand the intuition and mathematics behind attention mechanisms
- Implement different forms of attention in PyTorch
- Apply attention to sequence-to-sequence tasks
- Visualize attention weights to interpret model decisions
- Appreciate how attention mitigates the long-sequence limitations of RNNs

## Theory

### The Attention Concept

In essence, attention allows a model to "focus" on different parts of the input when producing each element of the output:

1. **Query, Key, Value Formulation**: The model computes compatibility between a query (what we're looking for) and keys (what we have), then uses these compatibility scores to weight the values.

2. **Attention Weights**: For each output position, the model computes a distribution over input positions, signifying where to focus.

3. **Context Vector**: The weighted sum of input features based on attention weights, capturing relevant information for the current prediction.

### Types of Attention

#### Content-Based Attention

- **Additive/Bahdanau Attention**: Uses a feedforward network to compute alignment scores
- **Multiplicative/Dot-Product Attention**: Uses dot product between query and key
- **Scaled Dot-Product Attention**: Normalizes dot product by √d_k to stabilize gradients

#### Location-Based Attention

- Computes attention weights based on the position in the sequence rather than content

### Self-Attention

- Relates different positions in a single sequence
- Each position attends to all positions, including itself
- Enables the model to capture long-range dependencies without sequential processing

## Coding Examples

Refer to the `/code` directory for complete examples:

1. `basic_attention.py` - Implementing basic attention mechanisms
2. `seq2seq_attention.py` - Sequence-to-sequence model with attention for translation
3. `self_attention.py` - Self-attention layer implementation
4. `attention_visualization.py` - Visualizing attention weights

## Exercises

In the `/exercises` directory, you'll find:

1. Implement a neural machine translation model with attention
2. Compare different attention mechanisms on sequence tasks
3. Add attention to an LSTM-based text classification model
4. Visualize and interpret attention patterns in your models

## Haiku Connection

Attention mechanisms parallel the way a haiku poet focuses on specific elements of a scene. Just as a haiku draws our attention to a particular moment or detail that carries significance beyond its immediate appearance, neural attention highlights parts of the input that are most relevant for the current prediction.

The selective nature of attention reflects the economy of expression in haiku. Both discard the unnecessary and amplify what matters. In haiku, we might focus on the sound of a frog jumping into a pond; in neural attention, the model might focus on key words that determine the meaning of an ambiguous phrase.

The ability of attention to create direct connections between distant elements mirrors how a haiku often contains a "cutting" (kireji) that creates an unexpected relationship between images. This leap across space in the poem, like attention spanning positions in a sequence, creates meaning that transcends linear processing.

## Resources

- "Neural Machine Translation by Jointly Learning to Align and Translate" by Bahdanau et al. (2014)
- "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al. (2015)
- "Attention Is All You Need" by Vaswoski et al. (2017)
- "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter on Sequence Modeling
