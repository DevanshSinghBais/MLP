# Character-Level Name Generator

A character-level name generation model built using **PyTorch and a Multi-Layer Perceptron (MLP)**. The project learns patterns and character combinations from a dataset of names and generates entirely new names one character at a time.

## Overview

This project explores how a neural network can learn the structure of names through character-level language modeling.

Instead of memorizing complete names, the model learns to predict the next character based on the preceding characters. By repeatedly sampling predictions, it can generate new names that resemble the patterns found in its training dataset.

## Model Architecture

The model is built using a Multi-Layer Perceptron with the following components:

* **Character Embeddings:** Maps each character to a learnable embedding vector.
* **Context Window:** Uses the previous characters as input to predict the next character.
* **Hidden Layer:** Processes the concatenated embeddings through a fully connected layer with `tanh` activation.
* **Output Layer:** Produces logits representing the probabilities of possible next characters.

### Architecture Flow

```text
Input Characters
       |
       v
Character Embeddings
       |
       v
Context Concatenation
       |
       v
Fully Connected Layer
       |
       v
Tanh Activation
       |
       v
Output Layer
       |
       v
Softmax Probabilities
       |
       v
Next Character Sampling
       |
       v
Generated Name
```

## Training Methodology

The model is trained using mini-batch gradient descent.

The training process involves:

1. Sampling batches of input contexts and their corresponding target characters.
2. Passing the input through the embedding and hidden layers.
3. Computing predictions using the output layer.
4. Calculating Cross-Entropy Loss.
5. Performing backpropagation to calculate gradients.
6. Updating model parameters using gradient descent.

The implementation uses PyTorch for tensor operations, automatic differentiation, and loss computation.

## Name Generation

After training, the model generates names autoregressively.

The generation process:

1. Starts with a special start/end token.
2. Predicts a probability distribution over possible next characters.
3. Samples the next character from the distribution.
4. Updates the context using the newly generated character.
5. Repeats until the end token is generated.

This allows the model to produce multiple distinct names without modifying its learned parameters.

## Project Structure

```text
MLP/
│
├── mlp.ipynb
├── names.txt
└── README.md
```

| File        | Description                                            |
| ----------- | ------------------------------------------------------ |
| `mlp.ipynb` | Complete implementation, training, and name generation |
| `names.txt` | Dataset used to train the model                        |
| `README.md` | Project documentation                                  |

## Technologies Used

* Python
* PyTorch
* NumPy
* Jupyter Notebook
* Machine Learning
* Deep Learning

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/DevanshSinghBais/MLP.git
```

### 2. Navigate to the Project Directory

```bash
cd MLP
```

### 3. Install Dependencies

```bash
pip install torch numpy jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `mlp.ipynb` and execute the cells sequentially to train the model and generate names.

## Learning Outcomes

Through this project, I explored:

* Character-level language modeling.
* Embedding representations.
* Multi-Layer Perceptron architecture.
* Forward propagation.
* Activation functions and logits.
* Cross-entropy loss.
* Backpropagation and gradient descent.
* Autoregressive text generation.
* Probabilistic character sampling.

---

**Built with Python and PyTorch.**
