import numpy as np
from base.tensor import Tensor

class Module():

    def __init__(self):
        self._modules = {}
        
    def forward(self, *input, **kwargs):
        # This is an abstract method; subclasses must implement it.
        
        raise NotImplementedError
    
    def __call__(self, *input, **kwargs):
        """used to instantiate the forward pass of the module"""
        return self.forward(*input, **kwargs)
    def __setattr__(self, key, value):
        
        if isinstance(value, Module):
            self._modules[key] = value
        super().__setattr__(key, value)
    
    def parameters(self):
        """
        Recursively collects parameters from all registered sub-modules.
        """
        params = []
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

class Linear(Module):
    """Class to create a Linear Layere"""
    def __init__(self, in_features, out_features):
        self._in = in_features
        self._out = out_features
        self._w = Tensor(np.random.normal(0,.1, size=(in_features, out_features)))
        self._b = Tensor(np.zeros(out_features))
    def forward(self, input):
        """Forward pass, on the input data
        Args:
            input (Tensor): The input tensor.
        Returns:
            Tensor: The output tensor of the calculation of inpput @ weights + biases.
        """
        return (input @ self._w) + self._b
    def layer_values(self):
        """Returns the weights, biases and the gradient tensor of the layer."""
        return {"weights" : self._w, "biases": self._b,"grad": self._w._grad}
    def parameters(self):
        """Returns the parameters of the layer."""
        return [self._w, self._b]

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        """Applies the ReLU activation function to the input tensor."""
        return input_tensor.ReLU()

class Conv1D(Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, batch_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernels = Tensor(np.random.normal(0, .1, size=(out_channels, in_channels, kernel_size)))
        self.bias = Tensor(np.zeros((1, out_channels, 1)))
    def forward(self, input_tensor):
        if len(input_tensor.get_shape()) < 3:
            shape = input_tensor.get_shape()
            input_tensor = input_tensor.reshape(1, shape[0], shape[1])
        output = input_tensor.conv_1d(kernels=self.kernels, stride=self.stride, padding=self.padding)
        return output + self.bias
    def parameters(self):
        return [self.kernels, self.bias]
class MSEloss(Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, true):
        """Computes the mean squared error loss.

        Args:
            pred (Tensor): The predicted output.
            true (Tensor): The ground truth output.

        Returns:
            Tensor: The computed loss.
        """
        diff = pred - true
        squared_diff = diff ** 2
        mean = squared_diff.mean()
        return mean

class CrossEntropyloss(Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, true):
        """Computes the cross-entropy loss.

        Args:
            pred (Tensor): The predicted output probabilities.
            true (Tensor): The ground truth output (one-hot encoded).

        Returns:
            Tensor: The computed loss.
        """
        log_pred = pred.log()
        diff = true * log_pred
        total_diff = diff.sum()
        loss = -total_diff
        return loss

class optimizer():
    class SGD():
        def __init__(self, model, lr):
            self._model = model
            self._lr = lr
        def step(self):
            for parameter in self._model.parameters():
                parameter._data += parameter._grad * (-1 * self._lr)

        def zero_grad(self):
            for parameter in self._model.parameters():
                parameter._grad = np.zeros_like(parameter._grad)