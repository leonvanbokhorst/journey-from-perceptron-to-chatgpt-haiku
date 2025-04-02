### Experiment: Sculpting Sight with Convolution

Here, we build a Convolutional Neural Network (CNN), an architecture specifically designed for processing grid-like data such as images. We'll use it to classify MNIST digits and compare its performance to the MLP from Module 2.

**A Guide to Building a Simple CNN for Image Classification**

1.  **Gather the Vision Tools (Imports):**
    We need the standard `torch` libraries, plus `torch.nn.functional` (often aliased as `F`) for activation functions like ReLU used within the `forward` method.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    ```

2.  **Design the Sculptor (Define the CNN Class):**
    This class defines the layers and the flow of data:

    - `conv1`: First convolutional layer. Takes 1 input channel (grayscale), outputs 8 feature maps using 3x3 filters with padding to maintain size.
    - `pool`: Max pooling layer. Reduces spatial dimensions by half (28x28 -> 14x14, then 14x14 -> 7x7).
    - `conv2`: Second convolutional layer. Takes 8 channels from the previous layer, outputs 16 feature maps using 3x3 filters.
    - `fc`: Final fully connected layer. Takes the flattened output from the convolutional/pooling layers (16 channels \* 7x7 spatial size) and maps it to 10 output scores.
    - `forward` method: Defines the sequence: Input -> Conv1 -> ReLU -> Pool -> Conv2 -> ReLU -> Pool -> Flatten -> FC -> Output.

    ```python
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # ... (conv1, pool, conv2, fc definitions as in script) ...

        def forward(self, x):
            # ... (forward pass logic as in script) ...
            return x
    ```

3.  **Prepare the Chisel and Stone (Hyperparameters, Data, Instantiation):**

    - Set hyperparameters like `LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`. Note that CNNs might benefit from slightly different learning rates than MLPs.
    - Load and prepare the MNIST dataset using `torchvision.datasets.MNIST` and `DataLoader`. This part is identical to Module 2.
    - Instantiate the `SimpleCNN` model.
    - Define `CrossEntropyLoss` as the criterion.
    - Define `Adam` as the optimizer.

    ```python
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    NUM_EPOCHS = 3

    print("Preparing MNIST dataset...")
    # ... (Data loading code identical to Module 2) ...
    print("Dataset loaded.")

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nCNN Model Structure:")
    print(model)
    ```

4.  **The Sculpting Process (Training Loop):**
    The structure of the training loop remains the same as in Module 2. The key difference lies within the `model(images)` call, which now executes the convolutional forward pass defined in the `SimpleCNN` class.

    - Loop through epochs.
    - Loop through batches from `train_loader`.
    - Zero gradients (`optimizer.zero_grad()`).
    - Forward pass (`outputs = model(images)`).
    - Calculate loss (`loss = criterion(outputs, labels)`).
    - Backward pass (`loss.backward()`).
    - Update weights (`optimizer.step()`).
    - Log progress.

    ```python
    print(f"\nStarting CNN Training for {NUM_EPOCHS} epochs...")
    # ... (Training loop code identical to Module 2) ...
    print("Training Finished.")
    ```

5.  **Appraise the Form (Evaluation Loop):**
    Similarly, the evaluation loop structure is identical to Module 2. It measures the accuracy of the trained CNN on the unseen test set.

    - Set model to evaluation mode (`model.eval()`).
    - Use `torch.no_grad()` context.
    - Loop through `test_loader`.
    - Get predictions.
    - Calculate accuracy.

    ```python
    print("\nStarting CNN Evaluation...")
    # ... (Evaluation loop code identical to Module 2) ...
    accuracy = 100 * correct / total
    print(f"Accuracy of the CNN on the {total} test images: {accuracy:.2f} %")
    print("Evaluation Finished.")
    ```

**Running the Code:**
Execute `simple_cnn_mnist.py`. Observe the training process and the final accuracy reported. Compare this accuracy to the one achieved by the MLP in Module 2. After evaluation, the script will also attempt to **visualize the learned filters from the first convolutional layer (`conv1`)** using `matplotlib` (if installed).

**6. Peering Through the Filter (Visualize Filters - Added):**
After training and evaluation, the script attempts to visualize the small kernels (filters) learned by the first convolutional layer (`conv1`).

- It requires `matplotlib` and `numpy`. If not installed, it prints a message and skips visualization.
- It retrieves the weights from `model.conv1`.
- The filters (typically 3x3) are normalized for better visibility.
- Each filter is displayed as a small grayscale image in a grid.

  ```python
  # --- 7. Visualize Conv1 Filters ---
  try:
      import matplotlib.pyplot as plt
      import numpy as np
      print("\nVisualizing filters from the first convolutional layer...")
      # ... (rest of filter visualization code as in script) ...
      plt.show()
      print("Filter visualization complete. Check the plot window.")
  except ImportError:
      print("\nVisualization skipped: matplotlib not found. Install it...")
  ```

**Reflection:**
The CNN leverages spatial locality and parameter sharing. **Observe the visualized filters.** What kinds of basic patterns (edges, corners, gradients, textures) do they seem to detect? These simple learned features are the building blocks the CNN uses to understand more complex visual information in deeper layers. This inherent structure makes CNNs powerful tools for image understanding.
