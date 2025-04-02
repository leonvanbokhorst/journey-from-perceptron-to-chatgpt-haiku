# Module 2: Multilayer Perceptrons – Building Depth

> _Hidden layers weave,_  
> _Patterns emerge from the depths,_  
> _Mind's eye awakens._

## Overview

In Module 1, we discovered the limitations of the perceptron - specifically, its inability to solve non-linearly separable problems like XOR. The multilayer perceptron (MLP) overcomes these limitations by stacking multiple layers of neurons, enabling the network to learn more complex decision boundaries.

## Learning Objectives

- Understand the architecture and mathematics of multilayer perceptrons
- Master the backpropagation algorithm and its intuition
- Implement MLPs in PyTorch for basic classification tasks
- Explore activation functions and their impact on network behavior
- Appreciate how depth in networks enables representation learning

## Theory

### The Multilayer Perceptron Architecture

The MLP consists of:

1. An input layer
2. One or more hidden layers
3. An output layer

Each layer contains neurons (nodes) that connect to all neurons in the next layer, creating a fully connected network. Unlike the simple perceptron, MLPs use continuous, differentiable activation functions that enable gradient-based learning.

### Activation Functions

Key activation functions include:

- Sigmoid: σ(x) = 1/(1 + e^(-x))
- Hyperbolic Tangent (tanh): tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
- Rectified Linear Unit (ReLU): f(x) = max(0, x)
- Softmax (for output layer in classification): σ(z)\_j = e^(z_j) / Σ(e^(z_k))

### The Backpropagation Algorithm

The backpropagation algorithm allows MLPs to learn by:

1. Forward pass: Computing outputs from inputs
2. Calculating error/loss
3. Backward pass: Computing gradients of the error with respect to weights
4. Updating weights using gradient descent

The key insight is the chain rule of calculus, which enables efficient gradient computation.

## Coding Examples

Refer to the `/code` directory for complete examples:

1. `mlp_from_scratch.py` - Building an MLP with backpropagation
2. `mlp_pytorch.py` - Using PyTorch to implement an MLP
3. `mnist_classification.py` - Classifying handwritten digits
4. `activation_functions.py` - Visualization of different activation functions

## Exercises

In the `/exercises` directory, you'll find:

1. Implement an MLP to solve the XOR problem
2. Experiment with different activation functions
3. Build a classifier for a small dataset
4. Visualize decision boundaries of MLPs with different architectures

## Haiku Connection

As haiku poets find depth of meaning in limited syllables, hidden layers in MLPs find deep patterns in data. The "hidden" aspect of these layers mirrors the implicit meaning in haiku - what is not explicitly stated, but emerges from the composition.

Like a haiku's juxtaposition of images to create meaning, MLPs juxtapose simple functions to create complex transformations. The network learns representations that capture the essence of data, just as a haiku captures the essence of a moment.

## Resources

- Original backpropagation paper: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.
- "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter on MLPs
- Additional reading materials in the `/resources` directory
