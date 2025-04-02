# Module 1: The Perceptron – Dawn of Neural Networks

> _A lone neural node –_  
> _learning to perceive the world;_  
> _sparks of thought in code._

## Overview

The perceptron, introduced by Frank Rosenblatt in 1957, represents the first computational model of a neuron. Like a haiku's first line sets the scene, the perceptron set the stage for all neural networks to follow.

## Learning Objectives

- Understand the biological inspiration behind the perceptron
- Master the mathematics of the perceptron model and its learning rule
- Implement a perceptron from scratch in PyTorch
- Visualize decision boundaries and understand linear separability
- Appreciate the elegant simplicity of the perceptron (like a haiku's minimalism)

## Theory

### The Perceptron Model

The perceptron is a binary classifier that makes its predictions using a simple rule:

1. Compute the weighted sum of inputs
2. Apply a step activation function
3. Output 1 if the sum is above a threshold, 0 otherwise

Mathematically:

- Given input features x₁, x₂, ..., xₙ
- With weights w₁, w₂, ..., wₙ
- And bias b
- Output = 1 if ∑(wᵢxᵢ) + b > 0, otherwise 0

### The Perceptron Learning Algorithm

1. Initialize weights and bias to small random values
2. For each training example:
   - Compute the predicted output
   - If prediction is correct, do nothing
   - If prediction is wrong:
     - If target=1 but prediction=0: increase weights
     - If target=0 but prediction=1: decrease weights

## Coding Examples

Refer to the `/code` directory for complete examples:

1. `perceptron_from_scratch.py` - Building a perceptron without libraries
2. `perceptron_pytorch.py` - Using PyTorch to implement a perceptron
3. `perceptron_visualization.py` - Visualizing the decision boundary

## Exercises

In the `/exercises` directory, you'll find:

1. Implement a perceptron for the AND logic gate
2. Implement a perceptron for the OR logic gate
3. Attempt to implement a perceptron for the XOR problem (spoiler: it won't work!)
4. Visualize why XOR fails and contemplate what this means for neural networks

## Haiku Connection

Just as a haiku captures a moment in just three lines with strict syllable counts (5-7-5), the perceptron captures a decision boundary with just one line. Both are elegant in their simplicity, yet limited in what they can express.

When we discover that a perceptron cannot solve the XOR problem, we find its limitation - just as a poet might feel constrained by haiku's rigid structure. Yet within these constraints, we find beauty and the seeds of greater innovation.

## Resources

- Original paper: Rosenblatt, F. (1957). The Perceptron: A Perceiving and Recognizing Automaton.
- "The Nature of Code" by Daniel Shiffman - Chapter on Neural Networks
- Additional reading materials in the `/resources` directory
