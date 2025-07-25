# Autograd Engine & Neural Network Library

A custom-built deep learning framework from scratch in Python and NumPy, demonstrating a fundamental understanding of automatic differentiation, neural network architectures, and low-level numerical computation. This project serves as a minimalistic, PyTorch-inspired re-implementation of core deep learning concepts.

## Features & Highlights

*   **Dynamic Computation Graph:** Implements a flexible computation graph that tracks operations on `Tensor` objects, enabling automatic gradient calculation.
*   **Reverse-Mode Automatic Differentiation (Autograd):** Supports efficient backpropagation (`.backward()`) across complex computational paths, driven by a **topological sort** of the graph.
*   **Modular Neural Network API (`nn.Module`):**
    *   **PyTorch-inspired Architecture:** Provides a base `Module` class for building neural networks, with automatic registration of sub-modules for seamless parameter collection.
    *   **Fundamental Layers:** Includes custom implementations of essential layers:
        *   `Linear`: Standard fully-connected layer.
        *   `ReLU`: Rectified Linear Unit activation function.
        *   `Conv1D`: **1D Convolutional Layer** with support for custom kernel sizes, strides, and padding.
*   **Loss Functions:** Includes `MSELoss` (Mean Squared Error) and `CrossEntropyLoss` for various learning tasks.
*   **Optimizers:** Features a basic `SGD` (Stochastic Gradient Descent) optimizer for parameter updates.
*   **Data Handling:** Custom `DataLoader` for efficient batching and shuffling of datasets.

## Structure
├── base/
│ └── tensor.py # Core Tensor class with autograd operations and backward definitions
├── nn.py # Neural Network module definitions (Module, Linear, ReLU, Conv1D, Losses, Optimizer)
├── DataLoader.py # Custom DataLoader for batching and shuffling data
└── main.py # Example script for training and evaluating models (MNIST/ECG5000)

## Installation
   ```bash
    pip install -r requirements.txt
    ```

## Future Enhancements

*   Implement `Conv2D` and `MaxPool` layers.
*   Add more optimizers (e.g., Adam, RMSprop).
*   Extend supported tensor operations (e.g., broadcasting for more complex ops).
*   Integrate GPU acceleration (e.g., via a custom C++/CUDA backend).
