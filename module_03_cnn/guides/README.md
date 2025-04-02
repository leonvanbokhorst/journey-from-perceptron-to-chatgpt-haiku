# Module 3 Guide: Convolutional Neural Networks – Vision and Patterns

This guide provides context for the code example in Module 3, focusing on Convolutional Neural Networks (CNNs). For the full theoretical background, historical context (LeNet, AlexNet), and exercises, please refer to **Module 3** in the main `../../curriculum.md` file.

## Objectives Recap

- Introduce CNNs and their suitability for spatial data like images.
- Understand key operations: convolution (`nn.Conv2d`) and pooling (`nn.MaxPool2d`).
- Grasp the concepts of local receptive fields, parameter sharing, and feature hierarchies.
- Implement a basic CNN in PyTorch for image classification (MNIST).

## Code Example: `simple_cnn_mnist.py`

The script `../simple_cnn_mnist.py` demonstrates building, training, and evaluating a simple CNN for handwritten digit classification using MNIST.

**Key Components:**

1.  **`SimpleCNN` Class (inherits from `nn.Module`):**
    - `__init__`: Defines the layers:
      - `conv1`: First convolutional layer (1 input channel, 8 output channels, 3x3 kernel, padding=1).
      - `pool`: Max pooling layer (2x2 kernel, stride=2) – used after both conv layers.
      - `conv2`: Second convolutional layer (8 input channels, 16 output channels, 3x3 kernel, padding=1).
      - `fc`: Final fully connected layer mapping the flattened features to 10 output classes.
    - `forward`: Defines the data flow:
      - Input `x` -> `conv1` -> `ReLU` -> `pool`.
      - Result -> `conv2` -> `ReLU` -> `pool`.
      - Result is flattened (`x.view(x.size(0), -1)`).
      - Flattened tensor -> `fc` layer to produce output logits.
      - Note the use of `F.relu` for the activation function.
2.  **`prepare_mnist_data` Function:** (Identical structure to Module 2) Downloads, transforms, and creates `DataLoader` instances for MNIST.
3.  **`train_model` Function:** (Identical structure to Module 2) Implements the standard PyTorch training loop using the specified model, data, loss (`nn.CrossEntropyLoss`), and optimizer (`Adam`). Backpropagation calculates gradients through all layers, including convolutional ones.
4.  **`evaluate_model` Function:** (Identical structure to Module 2) Evaluates the trained CNN on the test set and reports accuracy.
5.  **`visualize_filters` Function:**
    - Specifically designed for CNNs.
    - Extracts the learned filters (weights) from the _first convolutional layer_ (`model.conv1.weight`).
    - Normalizes the filter weights for better visualization.
    - Uses `matplotlib` to display each filter as a small image, revealing the low-level patterns the CNN learned to detect (e.g., edges, corners, gradients).
6.  **`if __name__ == \"__main__\":` Block:**
    - Sets hyperparameters (learning rate, batch size, epochs).
    - Orchestrates the process: prepares data, instantiates the `SimpleCNN` model, loss, and optimizer, trains, evaluates, and visualizes the learned filters.

**Running the Code:**

Navigate to the module directory and run the script. It will also download MNIST if needed.

```bash
cd module_03_cnn
python simple_cnn_mnist.py
```

You will observe the training process and evaluation, similar to the MLP. The key difference is the visualization at the end, which shows the 2D filters learned by the first convolutional layer – these often look like meaningful image patterns.

## Exercises & Further Exploration

Refer back to **Module 3** in `../../curriculum.md` for:

- **Exercise 3.1:** Experimenting with the CNN architecture (depth, filter sizes, etc.).
- **Exercise 3.2:** Visualizing intermediate feature maps (activations after conv layers).
- **Project 3.3:** Applying a CNN to the CIFAR-10 dataset.
- **Reflection 3.4:** Exploring creative CNN applications like style transfer or DeepDream.

This guide highlights how the code implements fundamental CNN concepts like convolution and pooling for image analysis.
