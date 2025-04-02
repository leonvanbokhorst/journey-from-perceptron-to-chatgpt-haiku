# Curriculum: Evolution of Neural Network Architectures from Perceptron to ChatGPT

## Introduction  

Artificial Intelligence has journeyed through an extraordinary evolution – from simple neuron-like models in the 1950s to massive transformer networks powering today’s conversational agents. This seven-module, self-paced curriculum will guide undergraduate AI engineering students through the history and technical progression of neural network architectures. Each module balances theory (including key mathematics) with hands-on coding exercises in PyTorch, providing practical experience that can be run on a high-end personal computer (gaming PC or recent MacBook). Along the way, we interweave elements of poetry and philosophy – specifically the art and history of **haiku** – as a creative metaphor for concepts in AI. Like a haiku’s elegant brevity and depth, we will find beauty in the simplicity of the perceptron, the layered complexity of deep networks, and the emergent creativity of models like ChatGPT. Themes of *being alive*, our *relationship with technology*, and *beauty in code and mathematics* are emphasized throughout. By the end of the curriculum, students will not only understand how architectures evolved but also undertake a final project: **building a chatbot that responds exclusively in high-quality, traditional haikus**. This project ties together technical skills with an appreciation for poetic creativity, highlighting the union of logic and art.

> *A lone neural node –  
> learning to perceive the world;  
> sparks of thought in code.*  

*(This opening haiku celebrates the humble perceptron, a single neuron model, capturing a “spark” of intelligence in a few lines – much like a haiku captures a moment in few syllables.)*

---

## Module 1: The Perceptron – Dawn of Neural Networks  

**Objectives:**  

- Introduce the perceptron, the first artificial neuron model, and its historical significance.  
- Understand the perceptron learning rule (gradient-free update) and the concept of linear separability.  
- Implement a simple perceptron classifier in PyTorch and observe its behavior on a basic dataset.  
- Draw parallels between the perceptron’s simplicity and the minimalism of haiku poetry.

### Theory and Background  

In the late 1950s, Frank Rosenblatt introduced the **perceptron** – essentially a single artificial neuron – marking one of the earliest milestones in AI ([Professor’s perceptron paved the way for AI – 60 years too soon | Cornell Chronicle](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon#:~:text=Inspired%20by%20the%20way%20neurons,thousands%20or%20millions%20of%20iterations)). The perceptron takes a vector of inputs, computes a weighted sum, and passes it through a step activation to produce an output (often 0 or 1). Despite its simplicity, Rosenblatt boldly described it as “the first machine which is capable of having an original idea” ([Professor’s perceptron paved the way for AI – 60 years too soon | Cornell Chronicle](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon#:~:text=It%20was%20a%20demonstration%20of,%E2%80%9956)). Early demonstrations showed a perceptron-based machine learning to distinguish patterns on cards after a few trials ([Professor’s perceptron paved the way for AI – 60 years too soon | Cornell Chronicle](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon#:~:text=In%20July%201958%2C%20the%20U,Research%20unveiled%20a%20remarkable%20invention)), sparking public imagination. A 1958 New Yorker article even hailed it as “the first serious rival to the human brain ever devised” ([Professor’s perceptron paved the way for AI – 60 years too soon | Cornell Chronicle](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon#:~:text=Rosenblatt%E2%80%99s%20claims%20drew%20keen%20interest,%E2%80%9D)) – a reflection of how *alive* this technology seemed at the time. The perceptron’s design was inspired by biological neurons and Rosenblatt’s quest to understand the minimal physical requirements for intelligence ([Professor’s perceptron paved the way for AI – 60 years too soon | Cornell Chronicle](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon#:~:text=%E2%80%9CHe%20wanted%20to%20ask%20himself,Sciences%2C%20at%20Rosenblatt%E2%80%99s%20memorial%20service)).

*Mathematically*, a perceptron performs a linear classification. Given inputs \(x_1, x_2, \dots, x_n\) with weights \(w_1, w_2, \dots, w_n\) and bias \(b\), the perceptron computes \(z = \sum_i w_i x_i + b\), then outputs \(y = \text{step}(z)\) where step is 1 if \(z\) is above 0 (and 0 otherwise). The learning algorithm adjusts weights whenever the output is incorrect: for a misclassified example with true label \(t\) and prediction \(p\), each weight is updated as \(w_i := w_i + \alpha (t - p)x_i\) (with \(\alpha\) a learning rate), and bias \(b := b + \alpha (t-p)\). This *Perceptron Learning Rule* guarantees convergence if the data is linearly separable.

However, the perceptron has limitations: it can only solve problems where a linear decision boundary can separate the classes. Notoriously, it cannot solve the XOR problem – a simple logic function which isn’t linearly separable. This limitation, pointed out by Marvin Minsky and Seymour Papert in 1969, tempered early excitement and led to a period of reduced funding (often called the “AI winter”) ([Professor’s perceptron paved the way for AI – 60 years too soon | Cornell Chronicle](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon#:~:text=But%20skeptics%20insisted%20the%20perceptron,the%20brains%20of%20untrained%20rats)). Yet, the perceptron’s legacy is profound: it laid the *foundation* for all modern neural networks ([Professor’s perceptron paved the way for AI – 60 years too soon | Cornell Chronicle](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon#:~:text=Today%2C%20many%20believe%20Rosenblatt%20has,translation%20%E2%80%93%20are%20transforming%20society)). Just as a haiku uses minimal syllables to convey a profound image, the perceptron uses a minimal network to capture a decision boundary.

### Hands-On Coding Example: Perceptron from Scratch  

Let’s implement a perceptron in PyTorch to classify a simple dataset (e.g. logical AND). We’ll do this manually to illustrate the weight update mechanism:

```python
import torch

# Dataset: inputs X (each with 2 features), and binary labels y (AND logic gate)
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
y = torch.tensor([0, 0, 0, 1])  # AND truth table outputs

# Initialize weights and bias
w = torch.zeros(2)    # 2 weights for 2 inputs
b = torch.zeros(1)    # bias term
lr = 0.1              # learning rate

# Train for a few epochs
for epoch in range(10):
    for i in range(X.size(0)):
        xi = X[i]; target = y[i].item()
        # Compute perceptron output
        output = 1 if (w.dot(xi) + b).item() >= 0 else 0
        # Update weights and bias if prediction is wrong
        error = target - output
        w += lr * error * xi
        b += lr * error
```

After training, the perceptron should learn to output 1 only for input (1,1). Try modifying the dataset to a non-linearly-separable one (like XOR) – you will observe the perceptron fails to converge (weights oscillate), underscoring its limitation.

### Exercises and Further Exploration  

- **Exercise 1.1:** Prove why a single-layer perceptron cannot learn the XOR function. (*Hint:* Sketch the points and see that no single straight line can separate the classes.)  
- **Exercise 1.2:** Using PyTorch’s autograd, define a perceptron model with a differentiable activation (e.g. sigmoid instead of step) and train it on a small dataset via gradient descent. Compare the results with the perceptron’s rule-based training.  
- **Exploration 1.3 (Historical):** Read about the Mark I Perceptron machine Rosenblatt built in 1958. How did it use a camera and hardware “association units” to learn to recognize patterns? What parallels can you draw between that setup and a modern neural net?  
- **Reflection 1.4 (Philosophical):** Rosenblatt described his machine as an “embryo” that could **“grow wiser”** ([Professor’s perceptron paved the way for AI – 60 years too soon | Cornell Chronicle](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon#:~:text=Rosenblatt%E2%80%99s%20claims%20drew%20keen%20interest,%E2%80%9D)). Discuss what he might have meant. Do you think an AI gaining knowledge is akin to an organism growing? Where do we draw the line between a clever program and something that feels *alive*?

### Reading & Resources  

- *History:* **“Professor’s Perceptron Paved the Way for AI – 60 years too soon”** – Cornell News ([Professor’s perceptron paved the way for AI – 60 years too soon | Cornell Chronicle](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon#:~:text=It%20was%20a%20demonstration%20of,%E2%80%9956)) ([Professor’s perceptron paved the way for AI – 60 years too soon | Cornell Chronicle](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon#:~:text=Rosenblatt%E2%80%99s%20claims%20drew%20keen%20interest,%E2%80%9D)). (Historical anecdotes on the perceptron’s debut and impact.)  
- *Foundations:* **Michael Nielsen’s “Neural Networks and Deep Learning”** – Chapter 1. Introduces perceptrons with code and visualizations.  
- *Interactive:* Play with a perceptron in the browser with the **Teachable Machine** (by Google) – build a two-class classifier with a single neuron and see it learn in real-time.  
- *Haiku:* *“The Old Pond”* by Matsuo Bashō (1686). This famous haiku’s simplicity mirrors the perceptron’s minimalism – a few inputs capturing a sudden insight (the splash of the frog).

---

## Module 2: Multi-Layer Perceptrons – From Single Neuron to Universal Approximator  

**Objectives:**  

- Understand how combining perceptrons in **multiple layers** (forming a feedforward neural network or MLP) overcomes the limitations of a single perceptron.  
- Learn the key algorithm that enabled training deep networks: **Backpropagation**.  
- Get hands-on experience building a simple multi-layer neural network in PyTorch (using `nn.Linear` and activation functions) and training it on a classic dataset (e.g. MNIST or a toy dataset).  
- Appreciate the historical revival of neural networks in the 1980s and draw analogies to layering in poetry.

### Theory and Background  

To solve problems like XOR, we need more than one neuron. Enter the **multi-layer perceptron (MLP)** – also called a feedforward neural network – which stacks neurons in layers: an input layer, one or more hidden layers, and an output layer. By introducing one hidden layer, an MLP can learn non-linear decision boundaries (for example, two neurons in the hidden layer can carve out an XOR decision region). In fact, with enough neurons, an MLP with even a single hidden layer is a **universal approximator** – capable of approximating any continuous function on compact domains. The trade-off is that we need a systematic way to adjust many weights across layers.

The breakthrough came with the **Backpropagation** algorithm (backprop for short), which is essentially an application of the chain rule of calculus to efficiently compute weight gradients in a multi-layer network. Although variants of backpropagation were described in the 1970s, its power was not fully realized until 1986 when Rumelhart, Hinton, and Williams published their famous paper *“Learning representations by back-propagating errors.”* This sparked a resurgence of interest in neural nets ([Backpropagation – Algorithm Hall of Fame](https://www.algorithmhalloffame.org/algorithms/neural-networks/backpropagation/#:~:text=The%20backpropagation%20algorithm%20was%20originally,which%20had%20previously%20been%20insoluble)) after the doldrums of the AI winter. Backpropagation allowed multi-layer networks to be trained on meaningful tasks by iteratively reducing a cost function (like mean squared error for regression or cross-entropy for classification) using gradient descent.

**How backpropagation works (in brief):** We initialize the network’s weights randomly. For each training example, we:  

1. **Forward pass:** Compute the outputs of each layer up to the final output.  
2. **Compute loss:** Measure the discrepancy between prediction and true target (e.g. using a loss function \(L\)).  
3. **Backward pass:** Compute gradients of \(L\) w.r.t. each weight by propagating the error backward. Each weight \(w\) receives an update \(\Delta w = -\alpha \frac{\partial L}{\partial w}\) (with learning rate \(\alpha\)).  

Backpropagation is essentially what PyTorch’s `autograd` does under the hood when you call `.backward()`. In code, defining an `nn.Sequential` or `nn.Module` and using an optimizer like SGD or Adam automates this process.

Historically, it’s worth noting that the resurgence of MLPs in the late 1980s was accompanied by computing improvements but still somewhat limited success – networks were shallow (1-2 hidden layers) due to computational limits and the risk of getting stuck in local minima or slow convergence. Yet, this era established neural networks as a viable approach, planting seeds for the deep learning revolution decades later. The **poetry** of this development lies in layering simple units to achieve expressive power: just as a haiku poet layers images and a “cutting word” to produce deeper meaning, an MLP layers simple perceptrons and non-linear activations to produce rich decision boundaries.

> *O snail, climb Mount Fuji,  
> but slowly, slowly! –* Issa (1810)  
>
> *In backprop’s long climb,  
> each small gradient step counts;  
> learning slow, but sure.*  

*(Issa’s haiku about a snail ascending Mount Fuji poetically parallels how backprop slowly adjusts weights – step by step – to reach higher accuracy. There is beauty in the patience and persistence of learning.)*

### Key Math – Activation Functions and Layers  

In an MLP, after each linear combination of inputs and weights, we apply a non-linear **activation function**. Common choices include the sigmoid (\(\sigma(x)=\frac{1}{1+e^{-x}}\)), hyperbolic tangent (tanh), and the now-ubiquitous ReLU (Rectified Linear Unit, \( \text{ReLU}(x)=\max(0,x)\)). Non-linear activations are crucial; without them, multiple layers would collapse into an equivalent single linear layer. With them, each layer’s output becomes a non-linear function of a weighted sum of the previous layer, enabling complex functions to be learned.

For a simple 2-layer network (one hidden layer), the model can be written as:  
\[ \mathbf{h} = f(W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) \]  
\[ \hat{y} = g(W^{(2)}\mathbf{h} + \mathbf{b}^{(2)}) \]  
Here \(f\) is the hidden activation (e.g. ReLU) and \(g\) the output activation (e.g. softmax for multi-class output). Backprop computes gradients \(\partial L/\partial W^{(2)}\), \(\partial L/\partial W^{(1)}\), etc., efficiently by the chain rule.

### Hands-On Coding Example: Training a Simple MLP  

Using PyTorch, we can quickly build and train an MLP on a classic dataset like **MNIST** (handwritten digits) or a smaller toy dataset if GPUs aren’t available. Below we construct a network and illustrate a training loop on a generic dataset:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple MLP with one hidden layer
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # Flatten input if needed (for images, e.g. 28x28 -> 784)
        x = x.view(x.size(0), -1)
        h = self.activation(self.fc1(x))
        out = self.fc2(h)
        return out

# Instantiate model, loss, optimizer
model = SimpleMLP(input_size=784, hidden_size=128, output_size=10)  # e.g., for MNIST 28x28 images
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Dummy training loop (for illustration; use actual data loader in practice)
for epoch in range(5):
    for batch_X, batch_y in train_loader:  # assuming train_loader yields image batches
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()      # backpropagate to compute gradients
        optimizer.step()     # update weights
```

After training, evaluate the MLP on test data to measure accuracy. Students should observe that even this simple network can learn to classify digits (achieving decent accuracy on MNIST, for example, around 90%+ with tuning).

**Tips:**  

- Try switching activation functions (e.g., replace ReLU with sigmoid) and observe differences in training speed and performance.  
- Visualize the learned weights of the first layer (for image input, reshape weight vectors into image patches). Are there any human-interpretable features? Often, MLP hidden neurons learn rudimentary features (for MNIST, perhaps stroke-like patterns).

### Exercises and Projects  

- **Exercise 2.1:** Train an MLP on the XOR dataset. Find the minimum hidden neurons required to solve XOR (you can brute force by trying 1, 2, 3…). Explain why at least that many are needed (relate to regions in input space).  
- **Exercise 2.2:** Implement backpropagation **manually** for a tiny 2-layer network in NumPy (or PyTorch without autograd). Compare your computed gradients with PyTorch’s `autograd` for the same random weights to verify correctness. This solidifies understanding of the chain rule.  
- **Project 2.3:** Build a two-layer MLP to approximate a sine function on \([0, 2\pi]\). Use a regression loss. Plot the MLP’s predictions against the true sine curve. How well does it interpolate and extrapolate? This demonstrates the universal approximation in action.  
- **Reflection 2.4 (Haiku and Layers):** Write a **haiku** that metaphorically describes the concept of layered learning or the backpropagation process. For example, think of each layer adding a “syllable” of understanding to the network’s “poem.” Share and discuss the creativity in each others’ haikus, highlighting how art can illuminate science.

### Reading & Resources  

- **Backpropagation Intuition:** *3Blue1Brown’s video series* on Neural Networks (YouTube) – especially the video *“What is backpropagation really doing?”* for a visual and intuitive walkthrough of gradient calculation.  
- **Classic Paper:** Rumelhart, Hinton, Williams (1986) – *“Learning representations by back-propagating errors.”* (Nature). It’s dense, but skimming the introduction can be insightful to see how the authors framed the importance of multi-layer learning ([Backpropagation – Algorithm Hall of Fame](https://www.algorithmhalloffame.org/algorithms/neural-networks/backpropagation/#:~:text=The%20backpropagation%20algorithm%20was%20originally,which%20had%20previously%20been%20insoluble)).  
- **Book:** *“Neural Networks: A Comprehensive Foundation”* by Simon Haykin – Chapters on perceptrons and MLPs for a deeper theoretical dive (optional).  
- **Interactive:** The *TensorFlow Playground* (web app) – allows you to create a small MLP, adjust neurons and activation functions, and watch it learn patterns. Try solving XOR or other patterns visually.  
- **Haiku History:** Learn about **Matsuo Bashō** and the evolution of haiku from Hokku (opening verse of linked poetry) to independent poems. Consider how a form with strict constraints (like 17 syllables in 5-7-5 pattern) can still express infinite variety – an analogy to how fixed architectures can approximate myriad functions through learning.

---

## Module 3: Convolutional Neural Networks – Vision, Convolution, and the Image Revolution  

**Objectives:**  

- Introduce **Convolutional Neural Networks (CNNs)** and why they are well-suited for image and spatial data.  
- Understand convolution and pooling operations, and the concepts of local receptive fields and parameter sharing.  
- Study the historical progression from early CNNs (LeNet) to the breakthrough deep CNN (AlexNet) that ignited the deep learning boom in computer vision.  
- Implement a basic CNN (e.g. for MNIST or CIFAR-10) in PyTorch and visualize its filters or feature maps.  
- Reflect on how CNNs “see” the world, relating it to how poets and artists perceive patterns in nature.

### Theory and Background  

Our previous networks treated inputs as flat vectors, ignoring any structure (like the 2D grid of an image). **Convolutional Neural Networks (CNNs)**, pioneered by Yann LeCun and others in the late 1980s, introduced a more efficient, biologically-inspired approach for processing images. Instead of connecting every pixel to every neuron in the next layer (which would be huge for images), CNNs use *convolutions* – sliding small filters (kernels) across the image to detect local patterns such as edges, textures, or shapes. Key ideas include:  

- **Local receptive fields:** Neurons respond to a small region of the input (e.g. a 5×5 patch of pixels) rather than the whole image, capturing local features.  
- **Shared weights:** The same filter (set of weights) is applied across different positions. This *weight sharing* exploits the stationarity of images (the idea that, say, an edge detector useful in the top-left is also useful in the bottom-right). It dramatically reduces the number of parameters.  
- **Pooling:** Downsampling operations (like max pooling) reduce the spatial resolution as we go deeper, while aggregating features locally. Pooling provides translational invariance (e.g. an object’s position in the image shouldn’t matter for detection).  

A basic CNN architecture might look like: **Input -> [Conv -> Activation -> Pool]* -> Fully Connected -> Output**. Early layers detect low-level features (edges, corners), mid-layers detect motifs (textures, shapes), and later layers detect high-level features (object parts), culminating in a classification (for vision tasks).

Historically, LeCun’s **LeNet-5** (1998) was a landmark CNN that recognized handwritten digits for postal mail sorting ([LeNet - Wikipedia](https://en.wikipedia.org/wiki/LeNet#:~:text=LeNet%20is%20a%20series%20of,56%20for%20reading%20cheques)) ([LeNet - Wikipedia](https://en.wikipedia.org/wiki/LeNet#:~:text=Convolutional%20neural%20networks%20are%20a,1)). It had around 5 layers and achieved then-state-of-the-art accuracy on digit recognition. Despite these successes, neural nets didn’t dominate until much later due to limited data and compute. By 2012, things had changed: the **ImageNet** dataset (millions of labeled images) and powerful GPUs set the stage. **AlexNet** (Krizhevsky et al., 2012) – an 8-layer CNN – stunned the research community by winning the ImageNet competition with a top-5 error of 15.3%, **over 10 percentage points better than the next best approach** ([AlexNet - Wikipedia](https://en.wikipedia.org/wiki/AlexNet#:~:text=The%20three%20formed%20team%20SuperVision,up)). This massive leap marked the definitive resurgence of neural networks in mainstream AI, effectively beginning the *deep learning era*. AlexNet’s success was attributed to depth, use of ReLU activations (which made training faster), and GPU training, among other tweaks ([AlexNet - Wikipedia](https://en.wikipedia.org/wiki/AlexNet#:~:text=Developed%20in%202012%20by%20Alex,1)) ([AlexNet - Wikipedia](https://en.wikipedia.org/wiki/AlexNet#:~:text=network%20achieved%20a%20top,up)).

From there, CNNs rapidly advanced: **VGG** (2014) went deeper with very small filters; **GoogLeNet/Inception** (2014) introduced network-in-network modules; **ResNet** (2015) introduced skip connections enabling ultra-deep networks (152 layers) by mitigating vanishing gradients. In parallel, applications exploded – CNNs became the go-to for image classification, object detection, face recognition, and more.

Thematically, CNNs taught us how to *see* with neural networks. Each filter is like a poet’s eye focusing on a detail – a shape or texture – and successive layers compose these details into a holistic understanding, analogous to how a poem’s images build a scene in the reader’s mind. There is *beauty in the mathematical patterns* a CNN learns: visually, some filters resemble Gabor filters (edge detectors), others pick up on repeating motifs. This visual interpretability is unique – we can literally *peek* into a CNN and see what it has learned, which is harder for fully connected networks.

### Core Concepts – Convolution and Pooling  

- **Convolution Operation:** For a 2D input (image), a filter (kernel) is a small matrix (e.g. 3×3). This filter *slides* (convolves) across the image, computing a dot product at each position and producing a feature map. If we have multiple filters, each produces its own feature map (thus increasing the number of channels as we go deeper). Convolution can be expressed as:  
  \[ (I * K)(x, y) = \sum_{i=1}^{m}\sum_{j=1}^{n} K(i,j) \cdot I(x+i, \; y+j) \]  
  where \(K\) is an \(m \times n\) filter and \(I\) is the image. In CNNs, we often use zero-padding to keep sizes, and stride (step size) to downsample.  
- **Pooling:** A pooling layer (e.g. 2×2 max pool) takes patches of the feature map and reduces them (taking the max or average). A 2×2 max pool with stride 2 will halve the width and height. This coarser representation forces the network to condense information (and also reduces computation for subsequent layers).

By stacking convolution and pooling layers, the spatial dimension shrinks while feature depth increases. Finally, fully connected layers (or global pooling) at the end produce the output predictions.

### Hands-On Coding Example: CNN for Image Classification  

We will implement a simple convolutional network in PyTorch for digit classification (MNIST). MNIST images are 28×28 grayscale, so a straightforward network might be:

1. Conv layer:  **1×28×28** input -> (e.g.) **8 filters of size 3×3** -> produces **8×28×28** feature maps (with padding=1 to preserve size).  
2. ReLU activation.  
3. Max Pool: 2×2 -> **8×14×14**.  
4. Conv layer: **8×14×14** -> **16 filters 3×3** -> **16×14×14**.  
5. ReLU, Max Pool 2×2 -> **16×7×7**.  
6. Flatten -> **16*7*7 = 784** features.  
7. Fully connected linear -> output 10 classes.

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16*7*7, 10)  # 10 output classes for digits 0-9
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)                   # 8x14x14
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)                   # 16x7x7
        x = x.view(x.size(0), -1)          # flatten
        x = self.fc(x)
        return x

model = SimpleCNN()
```

Train this model using a similar training loop as in Module 2 (but now our `train_loader` provides image batches of shape `[batch, 1, 28, 28]`). Use `CrossEntropyLoss` for multi-class classification and perhaps an optimizer like Adam for faster convergence. On MNIST, even this small network should reach >98% accuracy with a bit of tuning, demonstrating the power of convolution.

**Visualization:** One of the joys of CNNs is visualizing what they learn. After training, take the filters from `conv1` (which will be of shape `[8, 1, 3, 3]`) and visualize them as 3×3 image patches. You might see edge detectors oriented in different directions, or blob detectors – these are *informative features the network decided were useful*. You can also feed an image through the network and visualize the intermediate feature maps (e.g. print the activations after conv1 for a sample image to see where the first layer is activating strongly).

 ([File:LSTM cell.svg - Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LSTM_cell.svg)) *Illustration: Internal structure of a convolutional network’s early layers. Each filter (small colored grid) slides over the input image, producing an activation map highlighting where certain features (edges, textures) appear. Pooling then reduces resolution. Deeper layers combine these features into more complex motifs.*

*(The image above conceptually shows a CNN detecting features in an image – similar to how a poet’s senses pick up fragments of a scene to weave into verse.)*

### Exercises and Extensions  

- **Exercise 3.1:** Experiment with the CNN architecture. For instance, try adding a third conv layer, or increasing the number of filters, or using a larger kernel on the first layer. Does accuracy improve? What are the trade-offs (compute time, risk of overfitting)?  
- **Exercise 3.2 (Visualization):** Use a library like matplotlib to plot feature maps. Take a test image (say of the digit “7”), and plot the output of conv1 for each filter. Which filters respond strongly to which parts of the image? Interpret what feature each might be detecting (e.g. vertical stroke, horizontal line).  
- **Project 3.3:** Apply a CNN to a small part of the **CIFAR-10** dataset (which has 32×32 color images in 10 classes). Because CIFAR-10 has color images, modify the input channels to 3 (RGB) and possibly use a slightly deeper network. Aim to achieve as high accuracy as possible on, say, a subset of classes or the full set if your machine permits. Document how CNN hyperparameters affect performance.  
- **Reflection 3.4 (AI Art):** Research a creative application of CNNs, such as **neural style transfer** or **DeepDream**. These use CNN feature representations to create artistic images. Write a short piece (or create a haiku) on how a neural network’s “dream” or “art” might compare to a human’s. What does this say about the network’s internal representations of the world? (For fun, you could try generating a stylized image using an open-source notebook and share it.)

### Reading & Resources  

- **Diving Deeper:** *Chapter 7 of “Dive into Deep Learning” (d2l.ai)* – Covers CNNs (including LeNet) with code examples ([7.6. Convolutional Neural Networks (LeNet) - Dive into Deep Learning](http://d2l.ai/chapter_convolutional-neural-networks/lenet.html#:~:text=7,performance%20on%20computer%20vision%20tasks)). Great for both math and coding practice (with interactive notebooks).  
- **Historical:** LeCun et al. (1998), *“Gradient-Based Learning Applied to Document Recognition”* – the original LeNet-5 paper ([LeNet - Convolutional Neural Network in Python - PyImageSearch](https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/#:~:text=PyImageSearch%20pyimagesearch,As)). Skim to see architecture diagrams and the innovative ideas at the time.  
- **Article:** *“How ImageNet, AlexNet, and GPUs changed AI”* (Turing Post) – Overview of the 2012 breakthrough ([Breakthrough in Computer Vision. ImageNet Challenge 2012 and ...](https://medium.com/@atharvabarve24/breakthrough-in-computer-vision-c318fedba4d2#:~:text=Breakthrough%20in%20Computer%20Vision,dominance%20of%20this%20architecture)) ([Breakthrough in Computer Vision. ImageNet Challenge 2012 and ...](https://medium.com/@atharvabarve24/breakthrough-in-computer-vision-c318fedba4d2#:~:text=The%20top%205%20percent%20error,dominance%20of%20this%20architecture)), putting AlexNet’s achievement in context.  
- **Visualization Tool:** **CNN Explainer** (an interactive visual explainer for CNNs available online) – lets you interactively explore how convolutions, pooling, etc. work on sample images.  
- **Video:** Fei-Fei Li’s TED Talk *“How we’re teaching computers to understand pictures”* – provides inspiration on why vision is complex and how AI approaches it.  
- **Haiku Connection:** Look at ukiyo-e (Japanese woodblock prints) or photographic art and read accompanying haiku (as often done by haiku masters). Discuss how visual elements and poetry complement each other – akin to CNNs extracting visual features to later produce descriptive text (like image captions by AI). This bridges vision and language, foreshadowing how models like transformers can connect modalities.

---

## Module 4: Recurrent Neural Networks – Modeling Sequence and Memory  

**Objectives:**  

- Introduce **Recurrent Neural Networks (RNNs)** as a class of networks designed for sequence data (text, time series, etc.), maintaining a notion of *memory* through recurrent connections.  
- Understand how an RNN processes sequences step-by-step, and the concept of *hidden state*.  
- Explain the difficulties RNNs face (e.g. vanishing/exploding gradients for long sequences), setting the stage for improved variants.  
- Implement a simple character-level RNN in PyTorch (or use `nn.RNN`) to generate text or to predict sequence outputs.  
- Relate the idea of sequential memory to human memory and how a haiku’s meaning can depend on the sequence of lines and syllables.

### Theory and Background  

Not all data is static or spatial – much of what we encounter is sequential in nature: language (sequences of words or characters), music (notes over time), sensor readings (time series), user events, etc. A **Recurrent Neural Network (RNN)** is a type of network that is designed to handle sequences by *reusing* some neurons (and weights) at each time step of the sequence. In a sense, it performs the *same computation* for each element of the sequence, while carrying along a state that represents **context** from previous elements.

In the simplest form (often called a “vanilla” RNN or Elman network), at time step \(t\) the network takes input \(x_t\) and the previous hidden state \(h_{t-1}\), and produces a new hidden state \(h_t\) (and perhaps an output \(y_t\)). The recurrence can be written as:  
\[ h_t = f(W_{hx} x_t + W_{hh} h_{t-1} + b_h) \]  
\[ y_t = g(W_{yh} h_t + b_y) \]  
Here, \(h_0\) can be a zero vector (or learned initial state). \(W_{hh}\) is the recurrent weight matrix that connects the previous state to the next. The activation \(f\) is often a non-linearity like \(\tanh\) or ReLU. This equation highlights that the **same weights** \(W_{hx}, W_{hh}\) are used at every time step – an RNN *shares parameters across time*. This makes it efficient and applicable to sequences of arbitrary length.

An intuitive way to picture an RNN is to “unroll” it in time: a looped cell that feeds into itself can be drawn as a chain of cells, one per time step, each passing its hidden state to the next. This unrolled view reveals that an RNN is like a very deep network (depth = sequence length), where each layer (time step) shares weights with the next. Because of this, RNNs can, in theory, propagate information from early in the sequence to later steps, giving a form of memory.

**Example use-case:** language modeling. Feed in one character at a time to an RNN. At each step, predict the next character. The hidden state accumulates knowledge of all characters seen so far. Such an RNN can be trained on a corpus of text to predict likely next characters, and then be used to *generate* text by iteratively sampling its predictions and feeding them back in. This was famously demonstrated by Andrej Karpathy’s char-RNN, which learned to produce Shakespeare-like text character by character. Even though the RNN has no explicit grammar rules, it learned patterns in sequences of characters.

However, RNNs are tricky to train on long sequences. They suffer from the **vanishing gradient problem**: as gradients are backpropagated through many time steps, they tend to either decay exponentially (vanish) or blow up (explode). This makes it hard for vanilla RNNs to learn long-term dependencies (like something seen 50 steps ago affecting the current output). In practice, vanilla RNNs might remember things for maybe 5-10 steps reliably, but struggle beyond that. Solutions to this include gradient clipping (to address exploding gradients) and specialized architectures like LSTM and GRU (coming in Module 5) to address vanishing gradients.

From a philosophical lens, RNNs attempt to give neural networks a form of *working memory*. Consider reading a haiku: as you read each word or line, you retain a mental state that the next word builds upon. Early RNNs were like readers with very short memory spans – good at handling local dependencies (like syllable-to-syllable or word-to-word), but might “forget” the earlier context by the time they reach the end of a longer sentence or multi-sentence paragraph. In haiku, where brevity is key, a short-memory model might do okay; but for a longer poem or story, we’d need networks capable of longer memory (foreshadowing LSTMs).

### Hands-On Coding Example: Character-Level RNN Text Generation  

To concretely understand RNNs, let’s implement a simple character-level text generator. We’ll use PyTorch’s built-in RNN cell for simplicity. Suppose we want to generate haiku-like lines (though with a vanilla RNN, they might not follow the 5-7-5 structure yet).

```python
import torch.nn as nn

# Define the RNN model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, vocab_size)  # one-hot encoding via embedding
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, h):
        # x: [batch, seq_length] of character indices
        x_onehot = self.embed(x)           # [batch, seq_length, vocab_size]
        out, h_new = self.rnn(x_onehot, h) # out: [batch, seq_length, hidden_size]
        out = self.fc(out)                # [batch, seq_length, vocab_size] logits for next char
        return out, h_new

# Instantiate model
vocab = list("abcdefghijklmnopqrstuvwxyz ,.?;:\n")  # for example, define a vocab of characters
vocab_size = len(vocab)
model = CharRNN(vocab_size=vocab_size, hidden_size=128)
```

In a training loop, we would: feed sequences of characters (as indices) and train the model to predict the *next* character at each step (using `nn.CrossEntropyLoss` on the output against the actual next chars). We’d slide this over text data (e.g. a collection of haikus or any corpus). After training, we can generate text by: start with a start-of-sequence or a given initial character, then do a forward pass one character at a time, sampling the next char from the output probabilities, and feeding it in as the next input (updating the hidden state each time).

*Note:* Training an RNN this way can be slow. A more efficient approach is to process sequences in batches and use truncated backpropagation through time (BPTT) where you cut very long sequences into shorter segments for training.

However, even without fully implementing training here, one can *manually set some simple patterns* and see if the RNN can memorize them. For instance, train it on a simple repetitive sequence like “ABABAB…” and see if it learns to continue it (this is a trivial use-case to illustrate how an RNN picks up on sequence patterns).

**Using `nn.RNN` and `nn.LSTM` directly:** PyTorch offers high-level RNN modules. For example, `nn.RNN` or `nn.LSTM` can handle the recurrence for you. You can also use `nn.GRU`. These return outputs and the final hidden (and cell for LSTM) states. This can simplify code. The above custom class was for didactic clarity; in practice, one might just call `rnn_layer = nn.RNN(input_size, hidden_size, batch_first=True)` and use it.

### Experiment: A Simple Sequence Prediction  

As an exercise, let’s do a quick experiment with a short sequence: teach an RNN to generate a Fibonacci sequence or a simple poem structure. Even simpler: give it a sequence “ha ha ha ha …” (repeated “ha”), and see if it learns that pattern to continue. This might seem silly, but it’s essentially learning “output the same thing as last two letters” which is a memory pattern.

The key takeaway from coding with RNNs is understanding how the hidden state carries information. If you print out `h_new` vs `h` in the above code for a known sequence, you’ll see the vectors change, encoding the sequence history in a way that influences outputs.

### Challenges: Vanishing Gradients  

Try extending your sequence or making the pattern require memory of 10 steps. For instance, create training data where the desired output at position *n* equals the input from *n-5* (a 5-step dependency). A vanilla RNN may struggle to learn this reliably, especially if the sequences are long with a lot of distracting content in between. You might observe training difficulty – loss might plateau. This underscores why we need better architectures (to be discussed in Module 5: LSTM).

### Exercises  

- **Exercise 4.1:** Implement a toy RNN that learns to memorize a sequence. For example, feed in a random sequence of 0s and 1s of length 10, and ask the network to output the sequence shifted by one (i.e., predict the next element, with 0 for the last). Evaluate how well it does for various sequence lengths. When does it fail? (This tests short-term memory.)  
- **Exercise 4.2:** Train a character-level RNN on a small corpus of haiku poems. After training, generate some text. Does it produce haiku-like structure or words? Likely it will produce “character salad” with short memory, but you might see common syllables or simple words form. Save some of the funniest or strangest generated lines.  
- **Exercise 4.3 (Analytical):** Derive the backpropagation through time (BPTT) equations for a 2-step unrolled RNN. Show how the gradient at time step t depends on gradients from future time steps (t+1, t+2, etc.) through the recurrent weight \(W_{hh}\). Use this to explain why gradients can explode or vanish.  
- **Reflection 4.4 (Memory and Meaning):** How does an RNN’s way of “remembering” compare to a human’s memory when reading a sentence or a poem? Write a short paragraph or poem from the RNN’s point of view, describing how it only knows the recent past. For example, “I recall a word, the last word… the rest slips away.” This can be a creative writing exercise that anthropomorphizes the RNN’s memory.

### Reading & Resources  

- **Understanding RNNs:** Colah’s Blog – *“The Unreasonable Effectiveness of Recurrent Neural Networks”* by Andrej Karpathy. *(Highly recommended)* This blog post (2015) provides intuition on what RNNs (and LSTMs) can do, with examples of generated text (including Shakespeare, Wikipedia, etc.). It’s both entertaining and informative.  
- **Visual Tutorial:** *“A Simple RNN in Python from Scratch”* – a blog/tutorial that manually implements a tiny RNN for learning a sequence. Good for demystifying the math.  
- **Academic:** Jeff Elman’s classic paper *“Finding Structure in Time”* (1990), which introduced an early RNN with context units. While technical, reading the introduction and seeing how early ideas formed is insightful.  
- **Video Lecture:** Stanford CS224N (Natural Language Processing) Lecture on RNNs (by Chris Manning) – covers RNN basics and goes into language model applications.  
- **Haiku Anecdote:** Learn about the tradition of oral poetry and how bards would memorize long epics (e.g., Homer’s Odyssey) – they used rhythmic and repetitive structures as memory aids. Compare this to how an RNN might utilize repeating patterns to retain memory. There’s a parallel in how structure aids memory for both humans and machines.

---

## Module 5: Long Short-Term Memory (LSTM) Networks – Overcoming Forgetfulness in RNNs  

**Objectives:**  

- Present **Long Short-Term Memory (LSTM)** networks (and mention **GRU** as a variant) as solutions to the vanishing gradient problem, enabling much longer memory in sequences.  
- Understand the LSTM’s gating mechanisms (forget, input, output gates) and how they allow selective memory retention and updating.  
- Implement or use an LSTM in PyTorch for a practical task, such as text generation or sequence classification, and see the improvement over a vanilla RNN.  
- Reflect on the concept of “remembering” in LSTMs versus humans, and tie in the continuity of memory to themes of identity or the narrative flow in poetry.

### Theory and Background  

The LSTM network, invented by Sepp Hochreiter and Jürgen Schmidhuber (1997), introduces a special kind of recurrent neuron – the **memory cell** – designed to maintain information over long time intervals ([10.1. Long Short-Term Memory (LSTM) — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_recurrent-modern/lstm.html#:~:text=Shortly%20after%20the%20first%20Elman,model%20due%20to%20Hochreiter)) ([10.1. Long Short-Term Memory (LSTM) — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_recurrent-modern/lstm.html#:~:text=and%20Schmidhuber%20%281997%29,steps%20without%20vanishing%20or%20exploding)). The core idea is to provide an “information highway” through time by which gradients and information can flow with minimal interference. This is achieved via an explicit **cell state** that gets updated by controlled gates, rather than completely replaced at each step. In essence, an LSTM cell decides what to keep, what to write, and what to erase at each time step.

An LSTM cell has three (or four, depending on formulation) key components often described as gates:  

- **Forget Gate:** Decides what portion of the previous cell state to forget. Output is a number between 0 and 1 for each component of the cell state (0 = forget completely, 1 = keep completely).  
- **Input Gate (and Input Modulation):** Decides what new information to write to the cell state (and how significant it is). Usually an input gate that decides *if* to write, and a candidate value (often via a tanh layer) that decides *what* to write.  
- **Output Gate:** Decides how much of the cell state to reveal as the output (hidden state) at this time step.

The LSTM equations (one common variant) for a time step \(t\) are:  
\[ f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \]  
\[ i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \]  
\[ \tilde{C}*t = \tanh(W_C [h*{t-1}, x_t] + b_C) \]  
\[ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}*t \]  
\[ o_t = \sigma(W_o [h*{t-1}, x_t] + b_o) \]  
\[ h_t = o_t \odot \tanh(C_t) \]  

Where \(\sigma\) is the sigmoid function (producing gate values 0-1), and \(\odot\) denotes element-wise multiplication. \(C_t\) is the cell state and \(h_t\) is the hidden state (often also considered the output of the LSTM at step t). The crucial part is the cell update: \(C_t = f_t *C_{t-1} + i_t* \tilde{C}*t\). If \(f_t\) is near 1 and \(i_t\) is near 0, the cell state mostly carries over \(C*{t-1}\) (preserving long-term info) with little new addition – this prevents constant overwriting. Gradients can flow through the additive connection across many steps without vanishing (because if \(f\approx1\) and \(i\approx0\), \(C_t \approx C_{t-1}\), so \(\partial C_t/\partial C_{t-1} \approx 1\)). LSTMs were one of the first and most successful techniques to **address vanishing gradients and enable learning of long-range dependencies** ([10.1. Long Short-Term Memory (LSTM) — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_recurrent-modern/lstm.html#:~:text=handling%20vanishing%20gradients%20appears%20to,steps%20without%20vanishing%20or%20exploding)) ([10.1. Long Short-Term Memory (LSTM) — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_recurrent-modern/lstm.html#:~:text=and%20Schmidhuber%20%281997%29,steps%20without%20vanishing%20or%20exploding)).

In practice, LSTMs (and their simpler cousin GRUs – Gated Recurrent Units, introduced by Cho et al. 2014) revolutionized sequence learning in the 2010s. Tasks like language translation, speech recognition, and text generation that require remembering context over dozens of time steps became feasible with LSTMs. For example, machine translation models started using LSTMs to remember the beginning of a sentence while translating the end.

From a historical perspective, by 2014–2015, LSTMs were a key component of state-of-the-art models: Google’s speech recognition, Apple’s QuickType keyboard suggestions, and many others used LSTMs under the hood to model sequences. They were a stepping stone to even more powerful sequence models (transformers), but remain conceptually important and in use for certain applications even today.

Analogously, you can think of an LSTM’s cell state as a **notebook** that it carries as it reads a sequence: at each word, it decides whether to scribble a note (input gate), whether to erase some old notes (forget gate), and at the end, it shows you relevant parts of the notebook (output gate) as its understanding. This is like how a poet might take notes of impressions while observing nature and then selectively express some of them in a haiku. LSTMs, thus, provide a controlled flow of memory – they *decide what to remember*. Isn’t that what we humans do too? We can choose to remember certain moments (the sound of a bird at dawn) and let trivial details fade; a good memory (or a good algorithm) knows what is important to keep.

 ([File:LSTM cell.svg - Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LSTM_cell.svg)) *Diagram: Internal structure of an LSTM cell with gates.* *The green circles represent the pointwise operations (yellow ⊗ for multiplication, green + for addition), and orange boxes are neural network layers producing gate signals (σ for sigmoid, tanh for candidate). The cell state runs along the top (thick line), being adjusted by forget and input gates, and then influences the output through the output gate.*

*(This diagram illustrates how data flows through an LSTM cell. Notice how some paths go straight through (the cell state line), analogous to long-term memory, while other paths (through σ and tanh) regulate that flow. This structured design lets the network preserve information across many steps.)*

### Hands-On Coding: Using LSTMs for Text Generation  

We can easily swap our previous RNN model with PyTorch’s `nn.LSTM`. Let’s extend the char-RNN example from Module 4, but use an LSTM and see the difference in quality of output when generating longer sequences (if trained sufficiently).

```python
# Define an LSTM-based char model (using built-in LSTM)
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, vocab_size)
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden):
        x_onehot = self.embed(x)
        out, hidden_new = self.lstm(x_onehot, hidden)   # hidden contains (h, c) for LSTM
        out = self.fc(out)
        return out, hidden_new

model = CharLSTM(vocab_size, hidden_size=128)
```

Training this LSTM on a corpus of haikus (or any text) should yield faster convergence and better long-term consistency than a vanilla RNN. For example, an LSTM might more successfully capture that a haiku often has a line break after a certain number of syllables, or that if it generated the word “autumn” in the first line, it might relate it with “falling leaves” later (due to remembering context better). With enough training, it might even start producing the 5-7-5 syllable structure (especially if we incorporate syllable count into input features, but that’s an advanced exercise).

Even without fully training, one can initialize a hidden state and feed a starter sequence to the LSTM vs RNN and see that the LSTM’s hidden state might not zero out as quickly. However, a true appreciation comes from either looking at gradient norms (LSTM will have more stable gradients over long sequences) or from qualitative generation results after training.

**Practical tip:** Use **torch.nn.GRU** similarly if you want to try the GRU variant (which has fewer gates, thus slightly simpler). GRUs often perform similarly to LSTMs with a bit less compute.

### Example Project: Haiku Generator with LSTM  

As a project, collect a dataset of classic haiku (there are public domain collections available, or you can scrape some). Train an LSTM to generate text character by character or word by word. Then use it to generate haiku. Likely, you will need to guide it a bit (maybe tell it when to break lines). One strategy: include the newline character in the training data so it learns when line breaks occur. After training, generate text and insert line breaks when the model outputs them. You might end up with haiku-like verses that, while not strictly 5-7-5 syllables always, could be surprisingly poetic or at least coherent compared to a vanilla RNN. This foreshadows our final project of a haiku chatbot, but here the focus is on getting the model to produce haiku-style output.

### Exercises  

- **Exercise 5.1:** Compare RNN vs LSTM on a synthetic task. For instance, create sequences of length 50 where the first element and the last element are correlated (e.g. last element = copy of first). Train a vanilla RNN and an LSTM to predict the last element given the first 49. Measure accuracy. You should find the LSTM learns the long dependency much better ([10.1. Long Short-Term Memory (LSTM) — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_recurrent-modern/lstm.html#:~:text=Shortly%20after%20the%20first%20Elman,model%20due%20to%20Hochreiter)) ([10.1. Long Short-Term Memory (LSTM) — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_recurrent-modern/lstm.html#:~:text=and%20Schmidhuber%20%281997%29,steps%20without%20vanishing%20or%20exploding)).  
- **Exercise 5.2:** Open the black box: take a trained LSTM and inspect its gates’ values on some example sequence. You can instrument the forward pass to capture \(f_t, i_t, o_t\) values. For a given input sequence, plot these gate values over time. Do you notice patterns, like the forget gate staying near 1 for a while then dropping when a context shift happens? Understanding gate dynamics can build intuition on *how* the LSTM is handling memory.  
- **Exercise 5.3:** Implement an LSTM cell from scratch (if you’re comfortable with math coding). You can do this by directly coding the equations using PyTorch operations within a loop over time steps, instead of using `nn.LSTM`. This low-level exercise reinforces understanding of gates and can be debugged by comparing with `nn.LSTM` outputs on the same inputs.  
- **Reflection 5.4 (Memory and Identity):** LSTMs can carry pieces of information through long stretches and only use them when needed. Think of an example in human life where you carry a memory for a long time and it suddenly becomes relevant. Write a brief personal reflection or a poem about that. For instance, a childhood memory that guides a decision much later. Relate this to the LSTM’s behavior of preserving information until it’s useful.

### Reading & Resources  

- **Colah’s Blog – “Understanding LSTM Networks”** by Chris Olah. *This is a classic must-read.* It patiently walks through the LSTM intuition with diagrams and is widely praised for demystifying the LSTM.  
- **Dive into Deep Learning** – Section on LSTMs ([10.1. Long Short-Term Memory (LSTM) - Dive into Deep Learning](https://d2l.ai/chapter_recurrent-modern/lstm.html#:~:text=10.1.%20Long%20Short,model%20due%20to%20Hochreiter)) ([10.1. Long Short-Term Memory (LSTM) — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_recurrent-modern/lstm.html#:~:text=Shortly%20after%20the%20first%20Elman,model%20due%20to%20Hochreiter)) (with code). It includes simple examples and also introduces GRUs.  
- **Research Paper:** Hochreiter & Schmidhuber (1997), *“Long Short-Term Memory”* (Neural Computation). The original paper is heavy reading, but it’s interesting historically. They demonstrate solving toy problems that vanilla RNNs failed on.  
- **Modern Usage:** Blog post or article about how LSTMs were used in a real application (e.g., “How Google Translate uses LSTM” or “LSTM in Speech Recognition”). There are case studies available which show how these models made a difference in industry.  
- **Poetic Inspiration:** Many haikus reflect on memory and the passage of time. For example, Buson’s haiku: *“Light of the moon / Moves west, flowers’ shadows / Creep eastward.”* – it captures the flow of time and memory of a scene. Discuss how an LSTM might “remember” the moon’s position as it generates the latter part of a poem about shadows. Using poetic imagery can help solidify understanding of temporal dependencies.

---

## Module 6: Attention and Transformers – Sequence Learning Revolutionized  

**Objectives:**  

- Explain the concept of **Attention** in sequence models – how it allows a model to dynamically focus on different parts of the input sequence when producing each part of the output.  
- Discuss the rise of the **Transformer architecture** (Vaswani et al. 2017, *“Attention Is All You Need”* ([Attention Is All You Need - Wikipedia](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need#:~:text=,57%20techniques%20for%20machine%20translation))), which relies entirely on attention mechanisms (no RNNs or CNNs) for sequence processing.  
- Break down the Transformer’s components: self-attention, multi-head attention, positional encoding, feed-forward layers, etc.  
- Implement or utilize a small Transformer model (or subset, like a single Transformer block) in PyTorch, perhaps on a simple task like sequence-to-sequence mapping or text generation.  
- Show how transformers overcame limitations of RNN/CNN by enabling parallel computation and capturing long-range dependencies more effectively ([Attention Is All You Need - Wikipedia](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need#:~:text=The%20paper%20is%20most%20well,bigger%20sizes%20to%20be%20trained)).  
- Emphasize the historical significance: Transformers as a foundation for modern large language models (GPT, BERT) and many AI breakthroughs in the late 2010s.  
- Relate the attention mechanism to human cognition and poetic sensibility – e.g., how we pay attention to certain words in a sentence or certain images in a poem, and how that creates meaning.

### Theory and Background  

The **Attention mechanism** was first introduced in the context of neural machine translation (Bahdanau, 2014) as a method to allow a decoder to look directly at **all** the encoder’s hidden states when translating a given word, instead of just the last one. Instead of compressing an entire source sentence into one vector (like RNN encoder-decoder did), attention provides a way to **align** parts of the source with parts of the target. Concretely, an attention module computes a weighted sum of all source representations, with weights (attention scores) that are learned/computed based on how relevant source and target elements are to each other.

Fast forward to 2017: The Transformer architecture took this idea to the extreme. The seminal paper *“Attention Is All You Need”* introduced a deep learning architecture based **solely** on attention mechanisms – doing away with recurrence entirely ([Attention Is All You Need - Wikipedia](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need#:~:text=,57%20techniques%20for%20machine%20translation)). Transformers process sequences in parallel (not sequentially like RNNs), which, coupled with attention, allows them to capture long-distance relationships more effectively and utilize GPU parallelism for speed ([Attention Is All You Need - Wikipedia](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need#:~:text=The%20paper%20is%20most%20well,bigger%20sizes%20to%20be%20trained)). This proved to be a game-changer and is considered a foundational development that led to today’s AI boom, especially powering large language models ([Attention Is All You Need - Wikipedia](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need#:~:text=Google%20,57%20techniques%20for%20machine%20translation)) ([Attention Is All You Need - Wikipedia](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need#:~:text=artificial%20intelligence%20%2C%20and%20a,1)).

**How attention works (in Transformers):** The Transformer uses a mechanism called **self-attention**. For each position in a sequence, the model attends to other positions to compute a representation of that position in context. Mathematically, for a set of queries \(Q\), keys \(K\), and values \(V\) (which in self-attention are different linear projections of the sequence representations), the self-attention output is:  
\[ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V. \]  
This means for each query (each position), you take the dot product with all keys (all positions) to get a similarity score, apply softmax to get weights summing to 1, and then use these to take a weighted combination of the values (which are basically the content from each position). \(d_k\) is the dimensionality of the keys (a scaling factor).

Transformers have multiple such “heads” (multi-head attention) that allow the model to attend to different aspects of the input in parallel. They also incorporate **positional encoding** to give a sense of order to the sequence (since unlike RNNs, a pure attention mechanism is order-agnostic without this). The architecture of a Transformer (encoder or decoder block) typically is: Multi-head self-attention -> feed-forward network (applied to each position) -> with residual connections and layer normalization at each step. The encoder processes the input sequence into contextualized embeddings; the decoder similarly uses attention but can also attend to the encoder’s outputs for sequence-to-sequence tasks (like translation).

What makes the Transformer so powerful? A few reasons:  

- **Parallelism:** Unlike RNNs, Transformers don’t have to process tokens one by one – they can look at the whole sequence at once, which is much faster to train on GPUs (important for big data). ([Attention Is All You Need - Wikipedia](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need#:~:text=The%20paper%20is%20most%20well,bigger%20sizes%20to%20be%20trained))  
- **Long-Range Dependencies:** Any token can directly attend to any other token with just one attention layer – the distance is not a limiting factor (whereas an RNN might need 100 steps to connect token 1 and token 100, a Transformer can connect them in 1 step of attention with an appropriate weight).  
- **Expressivity:** Multi-head attention can capture various relations (like syntactic vs semantic relations in a sentence) simultaneously.  
- **Scalability:** With more data and compute, Transformers scale incredibly well. The architecture is very amenable to stacking many layers (as seen in e.g. GPT-3 with 96 layers, or even more).

From 2018 onward, **BERT** (bidirectional transformer for language understanding) and **GPT** (unidirectional transformer for language generation) showed that pretraining Transformers on massive corpora yields models with extraordinary capabilities. These models can write essays, answer questions, and as we’ll see in Module 7, engage in conversations like ChatGPT. All of this stems from that core attention mechanism making it feasible to train on unprecedented scales.

One can philosophically view attention as the model’s way of **“deciding what to focus on”** – akin to how our mind might focus on certain words in a sentence that are crucial to meaning. In poetry, the juxtaposition of two images (common in haiku) draws the reader’s attention to how they relate. In a transformer-based model writing a poem, it might implicitly attend to the earlier line when writing the next, creating that link. Attention embodies *relationality*: each element of the sequence is seen in relation to others. There’s also something beautifully *democratic* about it – every part of the input can, in principle, influence every other part directly, rather than a rigid left-to-right or right-to-left hierarchy.

### Hands-On: Building a Mini-Transformer  

Constructing a full Transformer from scratch is advanced, but we can play with a smaller instance or use PyTorch’s `nn.Transformer` module. For illustration, let’s use `torch.nn.MultiheadAttention` to perform attention on a simple example, then use `nn.Transformer` for a toy task.

**Example 1: Multi-head Attention on a toy input**  

```python
import torch.nn.functional as F
from torch.nn import MultiheadAttention

# Toy data: sequence length = 5, feature dim = 4
seq_len, feature_dim = 5, 4
# Suppose we have an input sequence (single batch)
x = torch.randn(seq_len, 1, feature_dim)  # shape [L, N, E] as required (L seq len, N batch, E embed dim)

# MultiheadAttention: let's use 2 heads, with embed dim = feature_dim
mha = MultiheadAttention(embed_dim=feature_dim, num_heads=2, batch_first=False)
attn_output, attn_weights = mha(x, x, x)
print("Attention output shape:", attn_output.shape)
print("Attention weights shape:", attn_weights.shape)
```

This snippet creates a random sequence and passes it through a self-attention (query, key, value all `x`). The `attn_weights` returned will be of shape [batch, num_heads, L, L] (if batch_first=False, actually [L, L] for each head here). You can inspect `attn_weights` to see how each position attended to each other (for random data it’s nonsense, but if you had a more structured input you could see meaningful patterns).

**Example 2: Sequence-to-sequence with nn.Transformer**  
PyTorch’s `nn.Transformer` requires some care with dimensions, but we can try a basic usage. Let’s say we want the transformer to learn an identity mapping (just output the input sequence shifted by one). This is trivial but will show the forward pass.

```python
from torch.nn import Transformer

# Define a small transformer
d_model = 8   # embedding dimension
nhead = 2     # heads
transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=16)

# Create random source and target sequences (shape [S,N,E] and [T,N,E], where S=source length, T=target length, N=batch)
src = torch.rand((5, 1, d_model))
tgt = torch.rand((6, 1, d_model))
# Pass through transformer
out = transformer(src, tgt)  # out shape: [T, N, d_model]
print("Transformer output shape:", out.shape)
```

To actually train this on a real task, one would wrap this in a model, add a final linear layer to project to output vocab, etc. For brevity, we’re focusing on usage. But an interested student could use `nn.Transformer` to build say a small translator for simple vocabulary, or a copy-reversal task (where the model learns to output the reversed sequence of the input).

**Understanding the output**: If you set `tgt` as the input sequence shifted and maybe include a start token, the transformer could learn to output the correct continuation. But training requires a loss and multiple examples, etc.

However, since our next module will discuss ChatGPT which is essentially built on transformers, this module should ensure the student conceptually gets how a transformer processes text. Possibly demonstrate a pre-trained small transformer in action:

For instance, use **HuggingFace Transformers** library to load a small model (like `distilGPT2`) and generate text from a prompt. Though that goes slightly beyond writing PyTorch from scratch, it’s highly practical and motivating.

```python
# Using HuggingFace transformers (if allowed) - demonstrating generation
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
prompt = "an old silent pond\n"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output_ids = model.generate(input_ids, max_length=20, num_beams=5, early_stopping=True)
print(tokenizer.decode(output_ids[0]))
```

This would output a completion starting with “an old silent pond”. The reason to show this is to connect that behind the scenes, this GPT-2 (a Transformer decoder) is using attention to produce the next words. If it continues with something like “...a frog jumps in, sound of water”, we’ll know it even recalled the famous Bashō haiku! (Unlikely unless it memorized it, but who knows – GPT-2 might know Bashō.)

### Exercises  

- **Exercise 6.1:** Manually compute a single attention head output for a small input to verify understanding. For instance, take Q, K, V each of size 2×2 (two positions, 2-d vectors). Choose simple values so you can do the matrix multiplication and softmax by hand. Compute the attention-weighted output and confirm with a Python snippet using the formula.  
- **Exercise 6.2:** The transformer’s lack of inherent ordering means positional encoding is important. Experiment by removing positional encoding (PyTorch’s `Transformer` allows you to feed inputs without it by default, but you can simulate a scenario by giving identical token embeddings for a permuted sequence). See if the model can learn order-dependent tasks without it (hint: it shouldn’t). This highlights why we need to inject sequence position information.  
- **Exercise 6.3 (Project):** Train a small transformer to learn a simple translation mapping: e.g., create a toy “language” of sequences of letters and define a deterministic rule to translate to another sequence of letters. Use the transformer to learn this mapping. For example, map each vowel to a number and each consonant to a letter in output. While contrived, it will help you practice setting up source/target and using transformer for sequence-to-sequence.  
- **Reflection 6.4 (Attention in Life):** We often talk about “paying attention” – humans have selective attention too. Write a journal entry about how you decide what to pay attention to in a day full of information. How is this similar to a model focusing on certain words when processing a sentence? Consider the balance between focusing broadly (scanning everything a little) vs focusing narrowly (attending strongly to a few things). How does this relate to multi-head attention (which can do both broad and narrow focusing via different heads)?

### Reading & Resources  

- **Original Paper:** Vaswani et al. (2017), *“Attention Is All You Need”*. (Even reading the abstract and introduction is useful to get their motivation and summary ([Attention Is All You Need - Wikipedia](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need#:~:text=,57%20techniques%20for%20machine%20translation)) ([Attention Is All You Need - Wikipedia](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need#:~:text=artificial%20intelligence%20%2C%20and%20a,1)).) Figure 1 in the paper is a great visual of the transformer architecture.  
- **Illustrated Guide:** *“The Illustrated Transformer”* by Jay Alammar. A highly recommended visual and intuitive explanation of how Transformers work, step by step, with diagrams and examples.  
- **Dive into Deep Learning:** Chapter on Attention Mechanisms and Transformers – breaks down the math and includes code to build a transformer from scratch in a guided way.  
- **Video:** Yannic Kilcher’s YouTube channel often has accessible summaries of important papers. Look for his video on “Attention is All You Need” if you prefer an explanation in that format.  
- **Interactive Visualization:** Try the Tensor2Tensor Transformer chat demo (if still available) or any online demo of transformers to see attention weights. Some tools let you input a sentence and see what words attend to what (e.g., in translation, you can visualize alignment).  
- **Analogy:** Read about the concept of *Indra’s Net* in philosophy – a metaphor where every jewel in a net reflects all other jewels. This is sometimes likened to how self-attention relates every word to every other word. Reflect on this metaphor as a poetic way to understand attention.  
- **Haiku and Attention:** A haiku often contains a “cut” (kireji) that juxtaposes two images, forcing the reader to hold both in mind and see the connection. Attention in a transformer might similarly link two distant words. For a fun exercise, take a famous haiku and imagine which words would attend to which if a transformer were to generate it. For example, in “old pond – frog jumps in – sound of water”, likely “frog” and “water” have a strong connection. This may humanize the abstract concept of attention.

---

## Module 7: Transformer Applications and ChatGPT – The Culmination in Conversational AI  

**Objectives:**  

- Explore how the transformer architecture enabled the creation of **Large Language Models (LLMs)** like GPT-2, GPT-3, and eventually ChatGPT.  
- Understand at a high level what ChatGPT is: a Transformer-based model (GPT series) fine-tuned with special techniques (such as Reinforcement Learning from Human Feedback, RLHF) to behave as an interactive conversational agent.  
- Discuss the scale of these models (billions of parameters, trained on massive corpora) and the kind of emergent capabilities they exhibit.  
- Delve into the *philosophical and ethical* questions around such models: the boundary between machine output and human-like creativity, the concept of “being alive” (spoiler: ChatGPT is not alive, but it can simulate conversation convincingly), and our relationship with AI (e.g., anthropomorphism, trust, companionship).  
- **Final Project:** Guide students to build a simplified version: a chatbot that responds in traditional haiku form. This will involve leveraging a pre-trained language model and either constraining its output or fine-tuning it on haiku data.  
- Provide pointers to resources and open-source tools (like Hugging Face, OpenAI API, etc.) for working with such models on personal hardware or with minimal resources (perhaps using smaller models like GPT-2 or distilled models).  
- Summarize the journey from the perceptron to ChatGPT, highlighting how each step built upon the last to bring us to today’s AI landscape.

### From Transformers to GPT  

After the transformer paper, researchers quickly applied the architecture to language modeling. OpenAI’s **GPT (Generative Pre-trained Transformer)** series began with GPT-1 in 2018, GPT-2 in 2019, and GPT-3 in 2020. These models are essentially decoder-only transformers trained to predict the next word in massive text datasets (internet archives, books, etc.), a training process known as *unsupervised language modeling*. GPT-2 (with ~1.5 billion parameters) surprised many with its ability to generate coherent paragraphs. GPT-3 took it to another level with **175 billion parameters** ([GPT-3 - Wikipedia](https://en.wikipedia.org/wiki/GPT-3#:~:text=architectures%20with%20a%20technique%20known,2)), demonstrating uncanny abilities to perform tasks with few examples (few-shot learning), such as translation or question answering, despite not being explicitly trained for those tasks. The sheer scale (175B) and the diverse training data allowed GPT-3 to acquire a broad range of world knowledge and linguistic skill.

However, these models would sometimes produce outputs that were unfocused or go off track in a conversation. They also weren’t aligned with human preferences out-of-the-box (e.g., they might generate offensive content or just not follow instructions well because they were trained only to continue text). To make them better conversationalists, OpenAI fine-tuned models using human demonstrations and feedback, a process described in the InstructGPT paper (2022) and used for ChatGPT.

**ChatGPT** (released publicly in late 2022) is essentially an improved version of GPT-3 (often referred to as GPT-3.5) that has been fine-tuned to follow instructions and have dialogues. This fine-tuning was done using two techniques:  

1. **Supervised fine-tuning:** They had human AI trainers have conversations, playing both user and assistant, to generate ideal responses. The model was initialized from a pre-trained GPT and fine-tuned on these Q-A pairs.  
2. **Reinforcement Learning from Human Feedback (RLHF):** Using the model from step 1, they generated multiple answers to many prompts, and humans ranked the answers. A reward model was trained on these rankings. Then the base model was further fine-tuned using a reinforcement learning algorithm (Proximal Policy Optimization, PPO) to maximize the reward model’s score ([Introducing ChatGPT | OpenAI](https://openai.com/index/chatgpt/#:~:text=We%20trained%20this%20model%20using,we%20transformed%20into%20a%20dialogue%C2%A0format)) ([Introducing ChatGPT | OpenAI](https://openai.com/index/chatgpt/#:~:text=To%20create%20a%20reward%20model,performed%20several%20iterations%20of%20this%C2%A0process)). This optimization teaches the model to prefer responses that humans rate as better – making it more helpful, correct, and polite on average.

The result is a model that can engage in multi-turn dialogue, ask clarifying questions, refuse inappropriate requests, and overall produce more aligned responses than a vanilla GPT-3. It’s important to stress: ChatGPT doesn’t “understand” in a human sense; it doesn’t have desires or consciousness. But the **illusion** of understanding is strong because it was trained to predict what a helpful, knowledgeable responder would say. It effectively mimics understanding by drawing on its vast training data and pattern recognition. It’s like an extremely advanced autocomplete that takes into account conversational context and human feedback.

**Scale and impact:** ChatGPT (and models like it) gained massive adoption. By January 2023, ChatGPT reached 100 million users, making it the fastest-growing consumer app in history ([ChatGPT sets record for fastest-growing user base - analyst note | Reuters](https://www.reuters.com/technology/chatgpt-sets-record-fastest-growing-user-base-analyst-note-2023-02-01/#:~:text=Feb%201%20%28Reuters%29%20,a%20UBS%20study%20on%20Wednesday)). People use it for drafting emails, writing code, getting tutoring, etc. This raises questions: if an AI can write a poem or a piece of code that’s indistinguishable from a human’s, what does that mean for creative work? There’s excitement and anxiety. The theme of *being alive* surfaces in how people can feel about ChatGPT – some chatbots (like in the Replika app) even led users to feel emotional attachment, as if the AI were a friend or partner. It’s crucial to remember it’s not alive; but our *relationship* with such technology is real and impactful. We must consider ethics (e.g., bias in training data leading to biased outputs, or potential misuse for spam, etc.).

In terms of beauty in code and math: the path from perceptron to ChatGPT is a story of how elegant mathematical ideas (gradient descent, convolution, recurrence, attention) layered on each other to create something seemingly magical. There is beauty in how these ideas unlocked capabilities: the perceptron was a simple line, the MLP a curved surface, CNNs a tapestry of learned filters, RNNs a path through time, LSTMs a gate mechanism like a careful editor, and Transformers a web of connections – altogether enabling an AI to craft a haiku. It’s like witnessing a progressive poem where each verse adds a new dimension.

### Building a Haiku Chatbot – Final Project  

For the final project, students will create a chatbot that responds only in high-quality, traditional haikus. This project brings together many threads:

- **Data**: You’ll need a collection of haikus to either fine-tune a model or use as examples. Traditional haiku in English often follow a 5-7-5 syllable pattern and reference nature or seasons (kigo). We can gather a dataset (perhaps ~1,000 haiku from public domain).  
- **Model Choice**: Using a full GPT-3 is not possible on a personal machine. But a smaller model like GPT-2 (117M parameters) is feasible. Alternatively, one could use an even smaller distilled model or an LSTM, but a transformer-based model will likely yield more coherent results.  
- **Approach 1: Fine-tune GPT-2 on Haikus**. Using Hugging Face’s Transformers library, one can fine-tune GPT-2 on the haiku dataset. This would teach it the style and content of haikus. After fine-tuning, generate responses by prompting the model with the user’s query plus perhaps an instruction like “Answer in a haiku:”. The model, now steeped in haiku style, should produce a relevant haiku. For example, if user asks: “How do clouds feel today?”, the prompt could be: “User: How do clouds feel today?\nAssistant (in haiku):” and the model completes it.  
- **Approach 2: Prompt Engineering**. If fine-tuning is not possible, one could try to prompt an existing model to answer in haiku. E.g., use GPT-2 (or even GPT-3 via API if allowed) with a prompt: *“Provide your answer in the form of a haiku (3 lines, 5-7-5 syllables):\nQ: <user question>\nA:”*. This sometimes works, but smaller models might not reliably stick to form without fine-tuning.  
- **Approach 3: Rule-based post-processing**. Another way is to use a model to generate a raw answer, then post-process it into a haiku. But this is quite complex (you’d need to impose syllable counts – maybe with a syllable counter algorithm – and choose words to fit). Given the time, Approach 1 is more straightforward and likely to produce beautiful results if done on even a small scale.

**Project Implementation Steps:**  

1. **Data Prep**: Get a set of haiku. For example, Matsuo Bashō, Yosa Buson, Kobayashi Issa, and Masaoka Shiki all have many famous ones (in translation). Ensure the format is one haiku per line or separated clearly (with newline breaks for lines).  
2. **Fine-tuning**: Use Hugging Face’s `Trainer` or a custom training loop to fine-tune `gpt2` on this dataset. This might take a few minutes to an hour on a good GPU, less on CPU for a small dataset. (It’s also an opportunity to use Google Colab if needed for free GPU time.)  
3. **Inference**: After fine-tuning, write a loop to interact: read user input, prepend an instruction to produce haiku, generate text with the model. Use decoding strategies that encourage correctness and poetry: perhaps beam search or nucleus sampling with a bias toward shorter output. Because haikus are short, you might generate up to, say, 50 tokens and then cut off.  
4. **Interaction**: Make it conversational by possibly keeping track of previous Q&A (though requirement is only that it responds in haiku, so it might not need context memory beyond the single turn). If ambitious, you could let it chain haikus (like a renku, for fun).  
5. **Evaluation**: Test with various questions. Ensure the outputs are indeed 3 lines (you might enforce newline insertion if needed by tweaking generation or checking syllables). Count syllables of the output to see if it meets 5-7-5 (you could integrate a syllable counter library or approximate by vowel group counting). If not exact, it’s okay – focus is on quality and style.  

The end result will be a delightful program: ask it “What is AI?,” it might respond with something like:  
“\nIn electric dreams,\nthoughts flicker like summer lightning –\nunbound by form, free.”  
(This is a haiku-esque response about AI I just improvised; the model might do even better!)

On the philosophical side, this final project underscores how we can imbue a machine with a narrow *persona* or style (here, a haiku poet). It’s intriguing: the machine doesn’t feel beauty or seasons, yet it can output words that make *us* feel something. It’s a mirror reflecting human training data (all the haikus humans wrote). It raises the question: when we find the AI’s haiku beautiful, who do we credit – the machine, the original poets it learned from, the engineer who fine-tuned it, or the math of the transformer? It’s a mix – a new kind of co-creation between human culture and AI.

### Responsible AI and Conclusion  

Before concluding the module (and the curriculum), it’s important to touch on AI ethics and limits. Emphasize that models like ChatGPT can sometimes produce incorrect information (they have no real “truth gauge”, they just sound confident) ([Introducing ChatGPT | OpenAI](https://openai.com/index/chatgpt/#:~:text=Limitations)), they can hallucinate facts, and they can reflect biases in their training data. OpenAI has tried to mitigate this (and uses a Moderation API to filter harmful content ([Introducing ChatGPT | OpenAI](https://openai.com/index/chatgpt/#:~:text=prompt%20multiple%20times,usually%20guess%20what%20the%20user%C2%A0intended)) ([Introducing ChatGPT | OpenAI](https://openai.com/index/chatgpt/#:~:text=,ongoing%20work%20to%20improve%20this%C2%A0system))), but it’s not perfect. Students should be aware of these issues as future AI engineers. Building AI that is not just capable, but also aligned with human values, is an ongoing challenge.

Finally, reflect on the journey: We started with a single neuron that could hardly do more than draw a line, and ended with a complex model that can carry a conversation or write a poem. Each step was built on math, experimentation, and insight, often inspired by biology (neurons, vision, memory) and refined by engineering. The curriculum showed the *relation* between these developments – none of this popped out of nowhere, it was a chain of innovation.

**Epilogue Haiku:** It might be fitting to end with a haiku that captures this journey:

> *Single thought – growing;  
> circuits learning to dream in verse –  
> dawn of new genius.*  

*(This haiku encapsulates how a simple idea (perceptron) grew through connections (circuits) to an AI that can “dream in verse” like ChatGPT, suggesting a new kind of intelligence dawning.)*

### Final Remarks and Resources for Module 7  

- **OpenAI’s blog “Introducing ChatGPT”** – describes how ChatGPT was trained and includes examples ([Introducing ChatGPT | OpenAI](https://openai.com/index/chatgpt/#:~:text=We%20trained%20this%20model%20using,we%20transformed%20into%20a%20dialogue%C2%A0format)) ([Introducing ChatGPT | OpenAI](https://openai.com/index/chatgpt/#:~:text=To%20create%20a%20reward%20model,performed%20several%20iterations%20of%20this%C2%A0process)). Good to skim to understand RLHF.  
- **InstructGPT Paper (OpenAI, 2022)** – if interested in RLHF technical details.  
- **Hugging Face Transformers** – library documentation and tutorials for fine-tuning models like GPT-2. Extremely useful for the final project. Many community notebooks exist (e.g., “Fine-tune GPT-2 to generate Shakespeare” which can be adapted to haiku).  
- **Open-Source Models:** If compute is an issue, look at smaller models like *DistilGPT2* (a distilled 82M version of GPT-2) or even *GPT-Neo-125M* (an open model similar to GPT-2). There are also LSTM-based language models (but transformers will likely perform better).  
- **Ethics Reading:** “Stochastic Parrots” paper by Bender et al., 2021 – a famous paper discussing the risks of large language models (bias, environmental cost, etc.). Important for an aspiring AI engineer to be aware of.  
- **Community & Forums:** Encourage joining communities (Reddit’s r/MachineLearning, Hugging Face forums, etc.) to stay updated. The field moves fast; transformers from 2017 led to ChatGPT by 2022 – just 5 years. Who knows what next year brings? Lifelong learning is key in AI.  
- **Perspective:** End with a thought that, despite these advances, human creativity and understanding have unique qualities. AI can mimic and even enhance creativity (like AI-assisted art and poetry), but it also challenges us to clarify what human creativity means. In building a haiku bot, we come to appreciate the art form more deeply, and we also recognize the beauty in the algorithms that made it possible. The interplay of human and machine becomes, itself, a kind of poetry – one of logic and imagination combined.

---

*© 2025. This curriculum was designed to provide a comprehensive, hands-on journey through neural network evolution, blending technical rigor with creative exploration. Each module should be taken at the student’s own pace – spend time on the exercises, discuss with peers or mentors, and don’t hesitate to dig deeper into the references. As Bashō wrote:*

> *“Seek not to follow in the footsteps of the wise; seek what they sought.”*

*In that spirit, don’t just emulate the accomplishments of AI pioneers – seek to understand the problems they tackled and the principles they uncovered. The next breakthroughs in AI will be made by those who stand on these shoulders and see further. Perhaps, by following this curriculum, one of those future innovators will be you.*
