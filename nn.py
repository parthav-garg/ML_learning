import numpy as np
from base.tensor import Tensor

class Module():

    def __init__(self):
        self._modules = {}
        
    def forward(self, *input, **kwargs):
        # This is an abstract method; subclasses must implement it.
        raise NotImplementedError
    def __call__(self, *input, **kwargs):
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
            params.append(module)
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
    
class MSEloss(Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, true):
        diff = pred - true
        squared_diff = diff ** 2
        mean = squared_diff.mean()
        return mean

class CrossEntropyloss(Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, true):
        log_pred = pred.log()
        diff = true * log_pred
        total_diff = diff.sum()
        loss = -total_diff
        return loss
        
class optimizer():
    class SGD():
        def __init__(self, layers, lr, batch_size=1):
            self._layers = layers
            self._lr = lr
            self._batch_size = batch_size
        def step(self):
            for layer in self._layers:
                for parameter in layer.parameters():
                    parameter._data += parameter._grad * (-1 * self._lr)
                
        def zero_grad(self):
            for layer in self._layers:
                for parameter in layer.parameters():
                    parameter._grad = np.zeros_like(parameter._grad)