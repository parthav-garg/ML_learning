import numpy as np


class Tensor:
    def __init__(self, data, _children):
        self._data = np.array(data)
        self._children = set()
        self.backward = lambda: None
        self.grad = np.zeros_like(self._data)
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        c = Tensor(self._data + other._data, )
        
        