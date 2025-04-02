# Journey from Perceptron to ChatGPT: A Haiku Adventure

This repository contains the code and supporting materials for a curriculum exploring the evolution of neural network architectures, from the humble perceptron to modern transformers like ChatGPT. The journey is framed through the lens of haiku poetry, emphasizing beauty in code, mathematics, and the relationship between technology and creativity.

**The full, detailed curriculum guide can be found in `curriculum.md`.**

## Repository Structure

Each module's primary code and guides reside within its respective directory:

```
module_XX_topic/
├── script_name.py     # Main code example(s) for the module
└── guides/            # (Optional) Further explanatory guides or resources
```

The main data directory (`./data/`) is used by modules requiring datasets like MNIST.

## Curriculum Modules & Code

This curriculum progresses through the following key stages:

1.  **Module 01: The Perceptron – Dawn of Neural Networks**

    - Focus: Linear separability, Perceptron Learning Rule.
    - Code: `module_01_perceptron/perceptron_and.py`
    - Run: `cd module_01_perceptron && python perceptron_and.py`

2.  **Module 02: Multi-Layer Perceptrons – Building Depth**

    - Focus: Universal approximation, backpropagation, `nn.Module`.
    - Code: `module_02_mlp/simple_mlp_mnist.py`
    - Run: `cd module_02_mlp && python simple_mlp_mnist.py` (Downloads MNIST data)

3.  **Module 03: Convolutional Neural Networks – Vision and Patterns**

    - Focus: Convolution, pooling, parameter sharing, local receptive fields.
    - Code: `module_03_cnn/simple_cnn_mnist.py`
    - Run: `cd module_03_cnn && python simple_cnn_mnist.py` (Downloads MNIST data)

4.  **Module 04: Recurrent Neural Networks – Memory and Sequence**

    - Focus: Hidden states, sequence processing, vanishing gradients.
    - Code: `module_04_rnn/rnn_hello.py` (Basic) & `rnn_long_dependency.py` (Limitation demo)
    - Run: `cd module_04_rnn && python rnn_hello.py`
    - Run: `cd module_04_rnn && python rnn_long_dependency.py`

5.  **Module 05: Long Short-Term Memory (LSTM) Networks – Overcoming Forgetfulness**

    - Focus: Gating mechanisms (forget, input, output), handling long dependencies.
    - Code: `module_05_lstm/lstm_long_dependency.py`
    - Run: `cd module_05_lstm && python lstm_long_dependency.py`

6.  **Module 06: Attention and Transformers – Sequence Learning Revolutionized**

    - Focus: Attention mechanism, self-attention, Transformer architecture basics.
    - Code: `module_06_transformer/observe_attention.py`
    - Run: `cd module_06_transformer && python observe_attention.py`

7.  **Module 07: Final Project – Building a Haiku Chatbot**
    - Focus: Applying transformers (GPT-2), fine-tuning for style, conversational AI basics.
    - Code: `module_07_final_project/haiku_bot_finetune.py`
    - Run: (Requires `haiku_dataset.txt`, see code comments)
      1.  Fine-tune: `cd module_07_final_project && python haiku_bot_finetune.py` (uncomment training lines)
      2.  Chat: `cd module_07_final_project && python haiku_bot_finetune.py` (uncomment inference lines)

## Getting Started

1.  **Clone:** `git clone <repository-url>`
2.  **Navigate:** `cd journey-from-perceptron-to-chatgpt-haiku`
3.  **Environment:** Create a Python virtual environment (e.g., using `venv` or `conda`).
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\\Scripts\\activate`
    ```
4.  **Install Deps:** Install required libraries.
    ```bash
    pip install -r requirements.txt
    ```
5.  **Explore:** Read `curriculum.md` for detailed theory and exercises.
6.  **Run Code:** Execute the Python scripts for each module as listed above.

## Requirements

- Python (Recommended: 3.8 or higher)
- PyTorch (See `requirements.txt` for compatible version)
- Other libraries specified in `requirements.txt`.
- An internet connection (for downloading datasets like MNIST).
- An environment capable of running PyTorch models (CPU is fine for early modules, GPU recommended for later modules/fine-tuning).
- Appreciation for both code and poetry!
