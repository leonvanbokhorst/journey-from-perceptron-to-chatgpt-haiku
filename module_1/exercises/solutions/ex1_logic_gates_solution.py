"""
Exercise 1: Implementing Logic Gates with Perceptrons (SOLUTION)
---------------------------------------------------------------
Complete solutions for implementing logic gates using perceptrons.
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """
    A simple perceptron implementation.
    """

    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights randomly (including bias)
        self.weights = np.random.randn(input_size + 1) * 0.1
        self.learning_rate = learning_rate

    def predict(self, inputs):
        # Add bias input of 1
        inputs_with_bias = np.insert(inputs, 0, 1)
        # Calculate net input
        activation = np.dot(inputs_with_bias, self.weights)
        # Apply step function
        return 1 if activation > 0 else 0

    def train(self, training_inputs, labels, max_epochs=100):
        """Train the perceptron"""
        errors_per_epoch = []

        for epoch in range(max_epochs):
            errors = 0

            for inputs, label in zip(training_inputs, labels):
                # Make prediction
                prediction = self.predict(inputs)

                # Update weights only if prediction is wrong
                if prediction != label:
                    # Calculate the error (difference between label and prediction)
                    error = label - prediction

                    # Increment error count
                    errors += 1

                    # Add bias to inputs
                    inputs_with_bias = np.insert(inputs, 0, 1)

                    # Update weights using perceptron learning rule
                    self.weights += self.learning_rate * error * inputs_with_bias

            # Store error count for this epoch
            errors_per_epoch.append(errors)

            # If no errors, we've converged
            if errors == 0:
                print(f"Converged after {epoch + 1} epochs!")
                break

        return errors_per_epoch


def visualize_decision_boundary(perceptron, inputs, labels, title):
    """
    Helper function to visualize the decision boundary of a trained perceptron
    """
    # Create mesh grid
    x_min, x_max = inputs[:, 0].min() - 0.5, inputs[:, 0].max() + 0.5
    y_min, y_max = inputs[:, 1].min() - 0.5, inputs[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Make predictions on the mesh grid
    Z = np.array([perceptron.predict([x, y]) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, edgecolors="k")
    plt.title(title)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid(True)
    plt.show()


def test_logic_gate(perceptron, inputs, labels, gate_name):
    """
    Test a perceptron on a logic gate
    """
    print(f"\nTesting {gate_name} gate:")
    for i, inputs_i in enumerate(inputs):
        prediction = perceptron.predict(inputs_i)
        print(
            f"{int(inputs_i[0])} {gate_name} {int(inputs_i[1])} = {prediction} (Expected: {labels[i]})"
        )

    # Check accuracy
    correct = sum(
        1
        for i, inputs_i in enumerate(inputs)
        if perceptron.predict(inputs_i) == labels[i]
    )
    print(f"Accuracy: {correct}/{len(inputs)} ({correct/len(inputs)*100:.1f}%)")


def implement_and_gate():
    """
    Implement AND gate using a perceptron
    """
    print("\n--- AND Gate Implementation ---")

    # Prepare training data
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Define correct labels for AND gate
    labels = np.array([0, 0, 0, 1])  # Truth table for AND

    # Create and train perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1)

    # Train the perceptron
    errors = perceptron.train(inputs, labels, max_epochs=20)

    # Test the trained perceptron
    test_logic_gate(perceptron, inputs, labels, "AND")

    # Visualize decision boundary
    visualize_decision_boundary(perceptron, inputs, labels, "AND Gate")

    # AND gate haiku
    print("\nAND Gate Haiku:")
    print("Both must be as one")
    print("Together in agreement")
    print("Then truth emerges.")


def implement_or_gate():
    """
    Implement OR gate using a perceptron
    """
    print("\n--- OR Gate Implementation ---")

    # Prepare training data
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 1])  # Truth table for OR

    # Create and train perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1)
    errors = perceptron.train(inputs, labels, max_epochs=20)

    # Test the trained perceptron
    test_logic_gate(perceptron, inputs, labels, "OR")

    # Visualize decision boundary
    visualize_decision_boundary(perceptron, inputs, labels, "OR Gate")

    # OR gate haiku
    print("\nOR Gate Haiku:")
    print("Either will suffice")
    print("Just one spark ignites the flame")
    print("Solitude or pair.")


def implement_nand_gate():
    """
    Implement NAND gate using a perceptron
    """
    print("\n--- NAND Gate Implementation ---")

    # Prepare training data
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([1, 1, 1, 0])  # Truth table for NAND (opposite of AND)

    # Create and train perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1)
    errors = perceptron.train(inputs, labels, max_epochs=20)

    # Test the trained perceptron
    test_logic_gate(perceptron, inputs, labels, "NAND")

    # Visualize decision boundary
    visualize_decision_boundary(perceptron, inputs, labels, "NAND Gate")

    # NAND gate haiku
    print("\nNAND Gate Haiku:")
    print("Negation of AND")
    print("Universal logic gate")
    print("Truth until both meet.")


def attempt_xor_gate():
    """
    Try to implement XOR gate using a single perceptron
    """
    print("\n--- XOR Gate Implementation (Attempt) ---")

    # Prepare training data
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])  # XOR truth table

    # Create and train perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1)
    errors = perceptron.train(inputs, labels, max_epochs=100)

    # Test the trained perceptron
    test_logic_gate(perceptron, inputs, labels, "XOR")

    # Visualize training progress
    plt.figure(figsize=(8, 5))
    plt.plot(errors)
    plt.xlabel("Epoch")
    plt.ylabel("Number of Errors")
    plt.title("XOR Training Progress")
    plt.grid(True)
    plt.show()

    # Visualize decision boundary
    visualize_decision_boundary(perceptron, inputs, labels, "XOR Gate (Attempt)")

    # Reflect on why XOR fails with a single perceptron
    xor_reflection = """
    The XOR problem is not linearly separable, meaning no single straight line
    can separate the two classes (0 and 1). A perceptron can only learn problems
    that are linearly separable because it implements a single linear decision
    boundary. For XOR, we need multiple lines (or a curve), which requires
    a network with at least one hidden layer.
    
    This limitation led to the first AI winter, as researchers realized simple
    perceptrons couldn't solve many interesting problems. The solution came years
    later with multilayer networks and the backpropagation algorithm.
    """
    print("\nReflection on XOR gate implementation:")
    print(xor_reflection)

    # XOR gate haiku
    print("\nXOR Gate Haiku:")
    print("Different, not same")
    print("Single line cannot divide")
    print("Layers must evolve.")


if __name__ == "__main__":
    print("Exercise 1: Implementing Logic Gates with Perceptrons (SOLUTION)")
    print("==============================================================")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run all implementations
    implement_and_gate()
    implement_or_gate()
    implement_nand_gate()
    attempt_xor_gate()
