# Module 4: Recurrent Neural Networks – Memory and Sequence

> _Memories unfold,_  
> _Each moment shaped by the past,_  
> _Time flows through neurons._

## Overview

Moving beyond the spatial patterns that CNNs excel at, we now explore Recurrent Neural Networks (RNNs), which are designed to handle sequential data. Unlike feedforward networks, RNNs maintain a hidden state that captures information about the sequence's history, making them ideal for tasks like language modeling, time series analysis, and speech recognition.

## Learning Objectives

- Understand the architecture and mathematics of recurrent neural networks
- Master the backpropagation through time algorithm
- Implement RNNs, LSTMs, and GRUs in PyTorch
- Build models for sequence processing and generation
- Appreciate how memory in networks enables contextual understanding

## Theory

### The Recurrent Architecture

Unlike feedforward networks, RNNs include feedback connections that create "memory":

1. **Hidden State**: A vector that maintains information across time steps
2. **Recurrent Connection**: The hidden state from the previous time step affects the current output
3. **Input Handling**: Processes one element of the sequence at a time

Mathematically, the basic RNN update rule is:

- h*t = tanh(W_hh · h*{t-1} + W_xh · x_t + b_h)
- y_t = W_hy · h_t + b_y

Where:

- h_t is the hidden state at time t
- x_t is the input at time t
- y_t is the output at time t
- W_hh, W_xh, W_hy are weight matrices
- b_h, b_y are bias vectors

### The Vanishing/Exploding Gradient Problem

Standard RNNs struggle with long-term dependencies due to:

- Vanishing gradients: Information from far in the past gets lost
- Exploding gradients: Gradients grow exponentially during training

### LSTM and GRU Architectures

To address the limitations of vanilla RNNs, more sophisticated architectures were developed:

#### Long Short-Term Memory (LSTM)

- **Forget Gate**: Controls what information to discard
- **Input Gate**: Controls what new information to store
- **Output Gate**: Controls what information to output
- **Cell State**: A separate memory track that can maintain information across many time steps

#### Gated Recurrent Unit (GRU)

- A simplified version of LSTM with fewer parameters
- **Update Gate**: Combines forget and input gates
- **Reset Gate**: Controls how much past information to forget

## Coding Examples

Refer to the `/code` directory for complete examples:

1. `basic_rnn.py` - Implementing a vanilla RNN from scratch
2. `lstm_pytorch.py` - Using PyTorch's LSTM implementation
3. `text_generation.py` - Generating text with character-level RNNs
4. `time_series_prediction.py` - Predicting sequence data

## Exercises

In the `/exercises` directory, you'll find:

1. Implement a character-level language model
2. Compare vanilla RNN, LSTM, and GRU on sequence tasks
3. Create a sentiment analysis model for text classification
4. Experiment with different sequence lengths and observe the impact

## Haiku Connection

Like a haiku that builds meaning across its three lines, with each new line informed by what came before, RNNs build understanding across a sequence, with each prediction informed by previous elements.

The LSTM's memory mechanisms mirror how a haiku poet carefully chooses what to remember and what to let go – the selective attention that creates powerful imagery with minimal words. The "cell state" of an LSTM is like the underlying emotion or season (kigo) in a haiku, carrying context through the entire poem.

Just as the final line of a haiku often provides resolution or a turning point that casts the previous lines in a new light, the final outputs of an RNN sequence are informed by the full context of what came before.

## Resources

- "Learning long-term dependencies with gradient descent is difficult" by Bengio et al. (1994)
- "Long Short-Term Memory" by Hochreiter & Schmidhuber (1997)
- "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" by Cho et al. (2014)
- "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter on RNNs
