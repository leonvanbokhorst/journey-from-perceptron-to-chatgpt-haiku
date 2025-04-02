### Experiment: Crafting the Perceptron's Dance

Here lies the path to build your first neuron mind. Follow these steps with calm focus, as one might arrange flowers in ikebana, finding beauty in each placement.

**A Guide to Building the Perceptron for AND Logic**

1.  **Prepare Your Canvas (Import PyTorch):**
    Just as a painter needs pigments, we need our tools. Bring forth the PyTorch library.

    ```python
    import torch
    ```

2.  **Gather Your Elements (The Data):**
    We need examples, like pebbles showing a pattern. Here, we use the AND gate – true only when both inputs are bright (1).

    - Inputs (`X`): Four pairs, showing all possibilities (0,0), (0,1), (1,0), (1,1).
    - Outputs (`y`): The matching truth for AND (0, 0, 0, 1).

    ```python
    # Data: Four scenes for the AND gate
    X = torch.tensor([[0.0, 0.0],  # Both off
                      [0.0, 1.0],  # One on
                      [1.0, 0.0],  # Other on
                      [1.0, 1.0]]) # Both on
    y = torch.tensor([0, 0, 0, 1])  # Only last scene is true 'AND'
    ```

3.  **Shape the Clay (Initialize Weights and Bias):**
    Start with potential, unformed. Weights (`w`) and bias (`b`) begin as zero, humble origins. The learning rate (`lr`) sets the pace of change, like gentle guidance.

    ```python
    # Begin with empty hands: zero weights, zero bias
    w = torch.zeros(2)    # Two paths for input features
    b = torch.zeros(1)    # A small nudge, the bias
    lr = 0.1              # How quickly we learn, step size
    ```

4.  **The Cycle of Seasons (Training Loop):**
    Learning takes time, repeated observation. We loop through epochs (full cycles) and within each, examine every scene.

    ```python
    # Observe the world, ten times around (epochs)
    for epoch in range(10):
        print(f"--- Epoch {epoch+1} ---")
        # Look at each scene one by one
        for i in range(X.size(0)):
            xi = X[i]          # Current input scene
            target = y[i].item() # The true answer we seek
            # ... (rest of the inner loop comes next)
    ```

5.  **Perceive and Decide (Forward Pass):**
    For each scene, the perceptron calculates its weighted sum plus bias. If the result is zero or more, it perceives '1', else '0'.

    ```python
            # Inside the inner loop (for i...)
            # See the scene, make a guess
            z = w.dot(xi) + b
            output = 1 if z.item() >= 0 else 0
            # print(f"  Input: {xi.tolist()}, Target: {target}, Prediction: {output}, Z: {z.item():.2f}") # Optional: Add for detailed view
    ```

6.  **Learn from Difference (Update Weights):**
    Did the guess match the truth? If not, adjust the weights and bias. Move them slightly towards the correct answer, guided by the error (`target - output`) and the input scene itself, following the Perceptron Learning Rule.

    ```python
            # Inside the inner loop (for i...)
            # Was the guess wrong? Learn from the error.
            error = target - output
            if error != 0:
                # Adjust weights along input paths
                w_change = lr * error * xi
                w += w_change
                # Adjust the bias nudge
                b_change = lr * error
                b += b_change
                # print(f"    -> Update! Error: {error}, Weight change: {w_change.tolist()}, New W: {w.tolist()}, Bias change: {b_change.item():.2f}, New B: {b.item():.2f}") # Optional: Add for detailed view
            # else:
                # print("    -> Correct!") # Optional: Add for detailed view

        # After checking all scenes in an epoch:
        # Check if converged (optional optimization)
        # ... (code for checking num_errors and breaking)

    # After all epochs:
    print(f"Final Weights: {w.tolist()}, Final Bias: {b.item():.2f}")
    ```

7.  **Observe the Outcome:**
    Run the complete code (found in `perceptron_and.py`). Watch how the weights and bias evolve. After training, the perceptron should reliably predict the AND logic. The script includes a final test section to verify this.

**Reflection:**
See how simple rules, repeated, lead to learning? Like water shaping stone, the updates carve a boundary. Now, consider the XOR problem (target `[0, 1, 1, 0]`) mentioned in the theory. If you modify the `y` tensor in the script, you will observe that this simple Perceptron cannot find a single line to separate the classes correctly; it will likely fail to converge. This illustrates its limitation and motivates the need for more complex structures, which we shall explore in the next module.
