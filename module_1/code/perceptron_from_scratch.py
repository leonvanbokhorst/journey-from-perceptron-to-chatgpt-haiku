"""
Perceptron from Scratch - Module 1
----------------------------------
A simple perceptron implementation without using neural network libraries.
Like the first haiku line, simple yet full of potential.
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """
    A single perceptron - the fundamental building block of neural networks.
    """

    def __init__(self, input_size, learning_rate=0.1, max_epochs=100):
        """
        Initialize the perceptron with random weights

        Args:
            input_size: Number of input features
            learning_rate: How quickly the model adjusts (eta)
            max_epochs: Maximum training iterations
        """
        # Initialize weights with small random values (including bias)
        self.weights = np.random.randn(input_size + 1) * 0.1
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def predict(self, inputs):
        """
        Make a prediction using the current weights

        Args:
            inputs: Input features (without bias term)

        Returns:
            1 if activation is positive, 0 otherwise
        """
        # Add bias term (1) to inputs
        inputs_with_bias = np.insert(inputs, 0, 1)
        # Calculate activation
        activation = np.dot(inputs_with_bias, self.weights)
        # Apply step function
        return 1 if activation > 0 else 0

    def train(self, training_inputs, labels):
        """
        Train the perceptron using the perceptron learning rule

        Args:
            training_inputs: Array of training examples
            labels: Target values (0 or 1)

        Returns:
            List of errors per epoch for plotting
        """
        errors_per_epoch = []

        for epoch in range(self.max_epochs):
            errors = 0

            # For each training example
            for inputs, label in zip(training_inputs, labels):
                # Make prediction
                prediction = self.predict(inputs)

                # Update weights if prediction is incorrect
                if prediction != label:
                    # Calculate error
                    error = label - prediction
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

    def plot_decision_boundary(self, inputs, labels):
        """
        Visualize the perceptron's decision boundary (only for 2D inputs)

        Args:
            inputs: Training data points (features)
            labels: Target values
        """
        if inputs.shape[1] != 2:
            print("Can only plot 2D inputs")
            return

        # Set min and max for plot range
        x_min, x_max = inputs[:, 0].min() - 0.5, inputs[:, 0].max() + 0.5
        y_min, y_max = inputs[:, 1].min() - 0.5, inputs[:, 1].max() + 0.5

        # Create a mesh grid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # Make predictions on the mesh grid
        Z = np.array([self.predict([x, y]) for x, y in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, edgecolors="k")

        # Plot the line w0 + w1*x + w2*y = 0, which is our decision boundary
        # Solving for y: y = (-w0 - w1*x) / w2
        slope = -self.weights[1] / self.weights[2]
        intercept = -self.weights[0] / self.weights[2]
        x_vals = np.array([x_min, x_max])
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, "r-", label="Decision Boundary")

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Perceptron Decision Boundary")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Example: Logic AND gate
    print("Training a perceptron to learn the AND logic function...")
    print("Truth table for AND:")
    print("0 AND 0 = 0")
    print("0 AND 1 = 0")
    print("1 AND 0 = 0")
    print("1 AND 1 = 1")

    # Prepare training data
    and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_labels = np.array([0, 0, 0, 1])

    # Initialize and train perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1, max_epochs=10)
    errors = perceptron.train(and_inputs, and_labels)

    # Test the model
    print("\nTesting the trained model:")
    for inputs in and_inputs:
        prediction = perceptron.predict(inputs)
        print(f"{inputs[0]} AND {inputs[1]} → {prediction}")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(errors)
    plt.xlabel("Epoch")
    plt.ylabel("Number of Errors")
    plt.title("Training Progress")
    plt.grid(True)
    plt.show()

    # Plot decision boundary
    perceptron.plot_decision_boundary(and_inputs, and_labels)

    # Haiku moment
    print("\nA perceptron's soul:")
    print("Simple line divides the space,")
    print("Wisdom in one stroke.")
