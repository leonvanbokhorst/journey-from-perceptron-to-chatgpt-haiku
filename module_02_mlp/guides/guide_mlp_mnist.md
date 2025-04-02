### Experiment: Weaving the Neural Net (MNIST Edition)

Now, we shall weave a simple tapestry, a network with a hidden layer, using the classic MNIST dataset of handwritten digits. This script will automatically download the data if needed.

**A Guide to Building and Training a Simple MLP for MNIST Classification**

1.  **Gather the Threads (Imports):**
    We need `torch` and its neural network (`nn`), optimizer (`optim`), and data handling (`utils.data`) tools. We also bring in `torchvision` to easily access and prepare the MNIST dataset.

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    ```

2.  **Design the Loom (Define the MLP Class):**
    This structure defines our network:

    - An input layer (`fc1`) flattens the 28x28 image (784 pixels) and maps it to a `hidden_size` (e.g., 128).
    - A ReLU activation (`activation`) adds non-linearity.
    - An output layer (`fc2`) maps the hidden representation to 10 output scores (one for each digit 0-9).
    - The `forward` method dictates the data flow: flatten -> fc1 -> ReLU -> fc2.

    ```python
    class SimpleMLP(nn.Module):
        # ... (class definition as in the python script) ...
    ```

3.  **Set the Loom's Parameters (Hyperparameters):**
    Define key values like hidden layer size, learning rate, batch size (how many images to process at once), and the number of training epochs (full passes over the dataset).

    ```python
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 10
    LEARNING_RATE = 0.01
    BATCH_SIZE = 64
    NUM_EPOCHS = 3 # Adjust as needed
    INPUT_SIZE = 28 * 28
    ```

4.  **Prepare the Yarn (Load and Prepare MNIST Data):**

    - Define transformations: Convert images to tensors and normalize pixel values (using standard MNIST mean and standard deviation).
    - Use `torchvision.datasets.MNIST` to download (if `download=True` and not found in `./data`) or load the training and test sets.
    - Wrap the datasets in `DataLoader` objects. These efficiently provide shuffled batches of data during training and evaluation.

    ```python
    print("Preparing MNIST dataset...")
    transform = transforms.Compose([...]) # As in script
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Dataset loaded...")
    ```

5.  **Prepare the Weaver (Instantiate Model, Loss, Optimizer):**

    - Create an instance of `SimpleMLP`.
    - Define the loss function: `nn.CrossEntropyLoss` is suitable for multi-class classification.
    - Define the optimizer: `optim.Adam` is often a good starting choice, linking it to the model's parameters and the learning rate.

    ```python
    model = SimpleMLP(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model Structure:\n", model)
    ```

6.  **The Weaving Process (Training Loop):**
    This is the core learning phase:

    - Set the model to training mode: `model.train()`.
    - Loop through the specified number of `NUM_EPOCHS`.
    - Inside the epoch loop, iterate through batches provided by `train_loader`.
    - For each batch (`images`, `labels`):
      - Zero the gradients accumulated from the previous batch: `optimizer.zero_grad()`.
      - Perform the forward pass to get predictions (logits): `outputs = model(images)`.
      - Calculate the loss between predictions and true labels: `loss = criterion(outputs, labels)`.
      - Perform the backward pass (backpropagation) to compute gradients: `loss.backward()`.
      - Update the model's weights using the optimizer: `optimizer.step()`.
    - Print loss periodically to monitor progress.

    ```python
    print(f"\nStarting Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        # ... (rest of training loop as in script) ...
    print("Training Finished.")
    ```

7.  **Admire the Tapestry (Evaluation Loop):**
    Assess the trained model's performance on the unseen test data:

    - Set the model to evaluation mode: `model.eval()`.
    - Disable gradient calculations (`with torch.no_grad():`) as they are not needed for evaluation.
    - Iterate through the `test_loader`.
    - For each batch, get predictions.
    - Find the predicted class index using `torch.max`.
    - Count the number of correct predictions.
    - Calculate and print the final accuracy.

    ```python
    print("\nStarting Evaluation...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # ... (rest of evaluation loop as in script) ...
    accuracy = 100 * correct / total
    print(f"Accuracy on test images: {accuracy:.2f} %")
    print("Evaluation Finished.")
    ```

**Running the Code:**
Execute the Python script (`simple_mlp_mnist.py`). It will first attempt to download MNIST data to a `./data` directory if it doesn't exist. Then, it will train the MLP for a few epochs, report the accuracy on the test set, and finally, **display a plot showing the learned weights of the first hidden layer.**

**8. Gaze into the Loom (Visualize Weights):**
After training and evaluation, the script includes a new section to visualize the patterns learned by the neurons in the first hidden layer.
_ It extracts the weight matrix connecting the 784 input pixels to the hidden neurons.
_ For a selection of hidden neurons (e.g., the first 16), it reshapes their corresponding 784 weights back into a 28x28 image. \* It uses `matplotlib` to display these weight images in a grid.

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    # ... (inside the script after evaluation)
    print("\nVisualizing first layer weights...")
    model.eval()
    weights = model.fc1.weight.data.cpu().numpy()
    # ... (rest of visualization code as in script) ...
    plt.show()
    print("Visualization complete. Check the plot window.")
    ```

**Reflection:**
Witness how the layered network learns to distinguish digits. **Observe the weight visualizations.** Do they resemble parts of digits, edges, or abstract patterns? These visualizations offer a glimpse into _how_ the MLP starts to break down the complex task. The jump in capability from the single Perceptron is substantial, and now we can see faint traces of its internal strategy. Consider experimenting with `HIDDEN_SIZE`, `LEARNING_RATE`, or `NUM_EPOCHS` to see how they affect the final accuracy **and** the appearance of the learned weights.
