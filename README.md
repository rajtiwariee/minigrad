# **Minigrad – A Tiny Autograd Engine and MLP from Scratch**

> **Author:** Raj Tiwari

> **Inspired by:** [Andrej Karpathy’s micrograd](https://github.com/karpathy/micrograd)

Minigrad is a minimal deep learning library that implements automatic differentiation and a simple multi-layer perceptron (MLP) from scratch in Python. It serves as a learning resource for understanding the internals of backpropagation, computational graphs, and training a neural network without relying on external ML libraries like PyTorch or TensorFlow.

---

## **🧠 Features**

* A scalar-based **autograd engine** built from first principles
* Fully connected **MLP (Multi-Layer Perceptron)** with ReLU activation
* **Backward propagation** and gradient calculation via topological sort
* **Training loop** with basic gradient descent
* **Visualization** of computational graphs using Graphviz
* Modular, extensible design using object-oriented principles

---

## **📁 Project Structure**

```
minigrad/
│
├── minigrad/
│   ├── __init__.py
│   ├── engine.py          # Core autograd engine (Value class)
│   ├── neural_net.py      # Neuron, Layer, MLP classes
│   └── visualize_nn.py    # Graphviz-based graph visualizer
│
├── test_training/
│   ├── test_model.py      # Training script
│   ├── final_prediction_graph.png
│   └── loss_over_epochs.png
│
├── micrograd_from_scratch_full.ipynb  # Full walkthrough notebook
└── README.md
```

---

## **🛠️ Installation & Setup**

1. **Clone the repo** **:**

```
git clone https://github.com/your-username/minigrad.git
cd minigrad
```

2. **Create virtual environment (optional)** **:**

```
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies** **:**

```
pip install matplotlib graphviz
```

4. **Install Graphviz** (if not already installed):

* **macOS**: brew install graphviz
* **Ubuntu**: sudo apt-get install graphviz
* **Windows**: [Download Graphviz](https://graphviz.gitlab.io/download/)

---

## **How to Run**

### **Training the MLP**

```
cd test_training
python test_model.py
```

This will:

* Train the MLP on a small toy dataset
* Print loss over 50 epochs
* Save a computational graph (**final_prediction_graph.png**)
* Save training loss curve (**loss_over_epochs.png**)

---

## **🧬 Example Output**

### **🔢 Sample Predictions**

```
Epoch 50/50, Loss: 0.002189

Final predictions:
Sample 1: Target=1.0, Predicted=0.9941
Sample 2: Target=0.0, Predicted=0.053
Sample 3: Target=1.0, Predicted=0.005
Sample 4: Target=0.0, Predicted=0.00
```

### **📉 Loss Over Epochs**

### **🕸️ Final Computational Graph**

Generated using **draw_dot()** for the last prediction.

Saved as **final_prediction_graph.png** and viewable in the default image viewer.

---

## **🧩 Code Highlights**

### **Autograd Engine (engine.py)**

```
class Value:
    def __add__, __mul__, __pow__, relu(), backward(), ...
```

* Supports operations like **+**, *****, **/**, ******, **ReLU**, and more
* Maintains gradient and dependency tracking
* Allows backpropagation via **.backward()** and chain rule

### **Neural Network (neural_net.py)**

```
class Neuron(Module)
class Layer(Module)
class MLP(Module)
```

* Modular architecture: **MLP** is made of **Layer**s, which are made of **Neuron**s
* Uses **Value** objects for weights, inputs, outputs
* Supports arbitrary layer depth and ReLU activations

---

## **🎯 Goals & Learning**

Minigrad is not optimized for performance or large-scale use. It’s meant to:

* **Demystify ****backpropagation** and **autograd**
* Provide a minimal reference to study **MLP training**
* **Help you ****build neural networks from scratch**

---

## **📜 License**

MIT License. Feel free to fork and extend!
