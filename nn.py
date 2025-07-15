import numpy as np
from base.tensor import Tensor

class Module():

    def __init__(self):
        self = self
    def forward(self, *input, **kwargs):
        # This is an abstract method; subclasses must implement it.
        raise NotImplementedError
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)
    
class Linear(Module):
    def __init__(self, in_features, out_features):
        self._in = in_features
        self._out = out_features
        self._w = Tensor(np.random.normal(0,.1, size=(in_features, out_features)))
        self._b = Tensor(np.zeros(out_features))
    def forward(self, input):
        return (input @ self._w) + self._b
    def parameters(self):
        return {"weights" : self._w, "biases": self._b,"grad": self._w._grad}
    
class optimiser():
    def __init__(self, layers, lr):
        self._layers = layers
        self._lr = lr
    def step(self):
        for layer in self._layers:
            layer._w._data += layer._w._grad * (-1 * self._lr)
            layer._b._data += layer._b._grad * (-1 * self._lr)
    def zero_grad(self):
        for layer in self._layers:
            layer._w._grad = np.zeros_like(layer._w._grad)
            layer._b._grad = np.zeros_like(layer._b._grad)