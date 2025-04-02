# Module 1 Guide: The Perceptron – Dawn of Neural Networks

This guide provides context and details for the code example in Module 1. For the full theoretical background, history, and exercises, please refer to **Module 1** in the main `../../curriculum.md` file.

## Objectives Recap

- Understand the Perceptron model and its historical significance.
- Learn the Perceptron learning rule for linearly separable data.
- Implement a simple Perceptron classifier in PyTorch.

## Code Example: `perceptron_and.py`

The script `../perceptron_and.py` provides a hands-on implementation of the concepts discussed in the curriculum.

**Key Steps in the Code:**

1.  **Dataset:** Defines the simple logical AND dataset (`X` inputs, `y` targets). This dataset is linearly separable.
2.  **Initialization:** Initializes weights (`w`) and bias (`b`) to zero. Sets a learning rate (`lr`).
3.  **Training Loop:**
    - Iterates through the dataset for a fixed number of epochs.
    - For each data point (`xi`, `target`):
      - Calculates the weighted sum: `z = w.dot(xi) + b`.
      - Applies a step activation function: `output = 1 if z >= 0 else 0`.
      - Calculates the error: `error = target - output`.
      - **Perceptron Learning Rule:** If `error` is non-zero, updates weights and bias:
        - `w += lr * error * xi`
        - `b += lr * error`
    - Tracks errors per epoch and stops early if convergence (zero errors) is achieved.
4.  **Testing:** After training, the script tests the learned weights and bias on the dataset to verify if the AND function has been learned correctly and calculates the accuracy.

**Running the Code:**

Navigate to the module directory and run the script:

```bash
cd module_01_perceptron
python perceptron_and.py
```

You should observe the training process, the number of errors decreasing per epoch (potentially fluctuating initially), and the final learned weights/bias that correctly classify the AND gate inputs.

## Exercises & Further Exploration

Refer back to **Module 1** in `../../curriculum.md` for:

- **Exercise 1.1:** Proving the Perceptron's limitation with XOR.
- **Exercise 1.2:** Implementing a Perceptron with a differentiable activation (Sigmoid) and gradient descent.
- **Exploration 1.3:** Researching the historical Mark I Perceptron machine.
- **Reflection 1.4:** Considering the concept of AI "growing wiser".

This structure provides a focused look at the practical implementation within Module 1.
