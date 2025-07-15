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
            update_w = layer._w.grad * (-1 * self._lr)
            layer._w = layer._w + update_w
            update_b = layer._b.grad * (-1 * self._lr)
            layer._b = layer._b + update_b
            