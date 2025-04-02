"""
Demonstrates a simple Perceptron classifier from scratch using PyTorch.
Trains the Perceptron on the logical AND dataset and visualizes the results.
This corresponds to the hands-on example in Module 1 of the curriculum.
"""

import torch

# Data: Four scenes for the AND gate
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0, 0, 0, 1])  # AND truth table outputs

# Begin with empty hands: zero weights, zero bias
w = torch.zeros(2)
b = torch.zeros(1)
lr = 0.1  # Learning rate

# Observe the world, ten times around (epochs)
print("Starting Perceptron Training for AND gate...")
for epoch in range(10):
    print(f"--- Epoch {epoch+1} ---")
    num_errors = 0
    # Look at each scene one by one
    for i in range(X.size(0)):
        xi = X[i]
        target = y[i].item()

        # See the scene, make a guess
        # Weighted sum + bias
        z = w.dot(xi) + b
        # Step activation function
        output = 1 if z.item() >= 0 else 0

        # Was the guess wrong? Learn from the error.
        error = target - output
        if error != 0:
            num_errors += 1
            # Adjust weights along input paths using Perceptron Learning Rule
            w_change = lr * error * xi
            w += w_change
            # Adjust the bias nudge
            b_change = lr * error
            b += b_change

    print(f"  Errors in epoch: {num_errors}")
    # Stop early if perfect convergence is reached
    if num_errors == 0:
        print("  Convergence reached!")
        break

print(f"Training Complete.")
print(f"Final Learned Weights: {w.tolist()}")
print(f"Final Learned Bias: {b.item():.2f}")

# Test the learned model to verify
print("--- Testing Learned Model ---")
correct_predictions = 0
for i in range(X.size(0)):
    xi = X[i]
    target = y[i].item()
    z = w.dot(xi) + b
    output = 1 if z.item() >= 0 else 0
    print(f"Input: {xi.tolist()}, Target: {target}, Predicted: {output}")
    if output == target:
        correct_predictions += 1

print(f"Model Accuracy: {correct_predictions / X.size(0) * 100:.2f}%")
