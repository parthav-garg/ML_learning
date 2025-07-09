import numpy as np


class Tensor:
    def __init__(self, data, _prev=set()):
        self._data = np.array(data)
        self._prev = _prev
        self.backward = lambda: None
        self._grad = np.zeros_like(self._data)
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data + other._data, _prev=(self, other))
        def backward_fn():
            print("self=", self._data)
            print("other=", other._data)
            self._grad += 1 * c._grad
            other._grad += 1 * c._grad
        c.backward = backward_fn
        return c
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data * other._data, _prev=(self, other))
        def backward_fn():
            self._grad += other._data * c._grad
            other._grad += self._data * c._grad
        c.backward = backward_fn
    
        
    def __repr__(self):
        return self._data.__str__()
    def __str__(self):
        return self._data.__str__()