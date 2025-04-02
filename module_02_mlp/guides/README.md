# Module 2 Guide: Multi-Layer Perceptrons – Building Depth

This guide provides context for the code example in Module 2, focusing on Multi-Layer Perceptrons (MLPs). For the full theoretical background, comparison with single perceptrons, backpropagation details, and exercises, please refer to **Module 2** in the main `../../curriculum.md` file.

## Objectives Recap

- Understand how MLPs overcome single-layer limitations (like XOR).
- Grasp the concept of backpropagation for training multi-layer networks.
- Implement a simple MLP in PyTorch using `nn.Module`, `nn.Linear`, and activation functions.
- Train and evaluate the MLP on a standard dataset (MNIST).

## Code Example: `simple_mlp_mnist.py`

The script `../simple_mlp_mnist.py` demonstrates building, training, and evaluating an MLP for handwritten digit classification using the MNIST dataset.

**Key Components:**

1.  **`SimpleMLP` Class (inherits from `nn.Module`):**
    - `__init__`: Defines the layers: an input linear layer (`fc1`), a ReLU activation, and an output linear layer (`fc2`).
    - `forward`: Defines the data flow: input `x` is flattened, passed through `fc1`, then ReLU activation, then `fc2` to produce output logits.
2.  **`prepare_mnist_data` Function:**
    - Uses `torchvision.datasets.MNIST` to download/load the dataset.
    - Applies necessary transforms (`ToTensor`, `Normalize`).
    - Creates `DataLoader` instances for efficient batching during training and testing.
3.  **`train_model` Function:**
    - Implements the standard PyTorch training loop:
      - Sets model to training mode (`model.train()`).
      - Iterates through epochs and batches from the `train_loader`.
      - Clears gradients (`optimizer.zero_grad()`).
      - Performs a forward pass (`outputs = model(images)`).
      - Calculates the loss using `nn.CrossEntropyLoss` (which expects raw logits).
      - Performs a backward pass (`loss.backward()`) to compute gradients via **autograd/backpropagation**.
      - Updates model weights (`optimizer.step()`).
4.  **`evaluate_model` Function:**
    - Sets model to evaluation mode (`model.eval()`).
    - Disables gradient calculations (`with torch.no_grad():`).
    - Iterates through the `test_loader`.
    - Calculates predictions based on the highest logit (`torch.max`).
    - Computes and prints the overall accuracy.
5.  **`visualize_weights` Function:**
    - Extracts the learned weights from the first linear layer (`model.fc1.weight`).
    - Reshapes each neuron's weight vector into a 28x28 image.
    - Uses `matplotlib` to display these weights, potentially revealing learned features (like stroke patterns).
6.  **`if __name__ == \"__main__\":` Block:**
    - Sets hyperparameters (hidden size, learning rate, epochs, etc.).
    - Orchestrates the overall process: prepares data, instantiates the model, loss, and optimizer (using `Adam`), trains the model, evaluates it, and visualizes the weights.

**Running the Code:**

Navigate to the module directory and run the script. Note that it will download the MNIST dataset (~115MB) into a `./data` subdirectory if it's not already present.

```bash
cd module_02_mlp
python simple_mlp_mnist.py
```

You will see the dataset preparation messages, the model structure, training progress (loss decreasing), the final test accuracy (should be reasonably high, e.g., >90% even with few epochs), and finally, a plot showing the learned weights of the first layer neurons.

## Exercises & Further Exploration

Refer back to **Module 2** in `../../curriculum.md` for:

- **Exercise 2.1:** Training an MLP on XOR and finding the minimal hidden layer size.
- **Exercise 2.2:** Implementing backpropagation manually for a small network.
- **Project 2.3:** Using an MLP to approximate a sine function.
- **Reflection 2.4:** Writing a haiku about layered learning or backpropagation.

This guide connects the implemented code to the core concepts of MLPs and backpropagation introduced in the curriculum.
