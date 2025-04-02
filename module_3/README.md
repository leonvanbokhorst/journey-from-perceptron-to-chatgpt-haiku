# Module 3: Convolutional Neural Networks – Vision and Patterns

> _Visual fields converge,_  
> _Filters seeking hidden shapes,_  
> _Images unveiled._

## Overview

After exploring fully connected networks in Module 2, we now turn to Convolutional Neural Networks (CNNs), which revolutionized computer vision. Inspired by the human visual cortex, CNNs use specialized layers that excel at extracting spatial hierarchies of features from images – from simple edges to complex objects.

## Learning Objectives

- Understand the biological inspiration behind convolutional networks
- Master the key operations: convolution, pooling, and feature mapping
- Implement CNNs in PyTorch for image classification
- Visualize learned filters and feature activations
- Apply transfer learning with pre-trained models

## Theory

### The CNN Architecture

The CNN architecture consists of several distinctive components:

1. **Convolutional Layers**: Apply learned filters across the input image to detect features
2. **Pooling Layers**: Reduce spatial dimensions while preserving important information
3. **Activation Functions**: Typically ReLU, introducing non-linearity
4. **Fully Connected Layers**: Connect to all activations in the previous layer, used for final classification

### Local Receptive Fields and Parameter Sharing

Unlike MLPs, CNNs use:

- **Local Receptive Fields**: Each neuron connects to only a small region of the input
- **Parameter Sharing**: The same filter weights are applied across the entire input
- **Translation Invariance**: Features can be detected regardless of their position

These properties dramatically reduce the number of parameters while improving generalization.

### Feature Hierarchies

CNNs naturally learn hierarchical features:

- Early layers detect edges and simple textures
- Middle layers combine these into shapes and patterns
- Later layers assemble complex object parts
- Final layers represent entire objects or scenes

## Coding Examples

Refer to the `/code` directory for complete examples:

1. `cnn_from_scratch.py` - Building a CNN with PyTorch
2. `mnist_cnn.py` - Classifying handwritten digits
3. `visualize_filters.py` - Visualizing learned CNN filters
4. `transfer_learning.py` - Using pre-trained models for new tasks

## Exercises

In the `/exercises` directory, you'll find:

1. Implement a basic CNN for MNIST classification
2. Experiment with different CNN architectures (LeNet, AlexNet)
3. Visualize feature maps at different network depths
4. Apply transfer learning using a pre-trained model

## Haiku Connection

Just as haiku poets distill visual scenes into concentrated form, CNNs extract the essence of images into feature maps. Both perform a type of dimensional reduction – finding what's important while discarding the superfluous.

The CNN's hierarchical structure mirrors how haiku often works: first capturing concrete details (like the convolutional layers), then assembling them into a coherent scene (like the fully connected layers), and finally evoking deeper meaning (the classification).

The local receptive field of CNNs also reminds us of the haiku poet's focused attention on a small but significant detail from which the whole poem unfolds.

## Resources

- "ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky et al. (2012)
- "Visualizing and Understanding Convolutional Networks" by Zeiler and Fergus (2013)
- "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter on CNNs
- Additional reading materials in the `/resources` directory
