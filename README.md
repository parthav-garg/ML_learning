# Autograd Engine & Neural Network Library

A minimal, PyTorch-inspired deep learning framework built from scratch using **Python** and **NumPy**. This project demonstrates a foundational understanding of **automatic differentiation**, **neural network architectures**, and **low-level numerical computation**.

---

## 🚀 Features

- **🔁 Dynamic Computation Graph**  
  Tracks operations on `Tensor` objects and enables flexible construction of computation graphs for automatic differentiation.

- **🧮 Reverse-Mode Autograd**  
  Implements efficient backpropagation using **topological sorting** of the computation graph.

- **🧱 Modular Neural Network API (`nn.Module`)**  
  - PyTorch-like `Module` base class with automatic submodule registration  
  - Core layer implementations:
    - `Linear`: Fully connected layer  
    - `ReLU`: Rectified Linear Unit  
    - `Conv1D`: 1D convolution with configurable kernel size, stride, and padding

- **📉 Loss Functions**  
  - `MSELoss` (Mean Squared Error)  
  - `CrossEntropyLoss`

- **⚙️ Optimizers**  
  - `SGD` (Stochastic Gradient Descent)  
  - `Adam` (Adaptive Moment Estimation)

- **📦 Data Handling**  
  - Custom `DataLoader` for batching and shuffling datasets efficiently

---

## 📁 Project Structure

```
.
├── base/
│   └── tensor.py            # Core Tensor class with autograd & graph operations
│
├── nn.py                    # Layer implementations (Module, Linear, Conv1D, ReLU, Loss, Optimizers)
├── DataLoader.py            # Minimal DataLoader for batching and shuffling
├── main.py                  # Model training & evaluation entry point
│
├── examples/
│   ├── ecg5000_cnn.py       # 1D CNN training example on ECG5000 dataset
│   └── mnist_mlp.py         # MLP training example on MNIST
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## ⚡ Quick Start

### 🔧 Installation

```bash
pip install -r requirements.txt
```

### 🧪 Running Examples

**Train a CNN on ECG5000**
```bash
python -m examples.ecg5000_cnn
```

**Train an MLP on MNIST**
```bash
python -m examples.mnist_mlp
```

---

## 🔭 Future Enhancements

- [ ] Add `Conv2D` layer and related support
- [ ] Implement tensor broadcasting and advanced element-wise ops
- [ ] GPU acceleration via C++/CUDA backend
- [ ] Add support for dropout and batch normalization
- [ ] Save/load model checkpoints

---

## 🧠 Why This Project?

This framework was built to deeply understand:
- How **reverse-mode autodiff** and computation graphs work  
- How layers are modularized and composed in modern DL libraries  
- The internal mechanics of optimizers and backpropagation  
- The relationship between low-level numerical ops and high-level abstraction
