import numpy as np
eps = 1e-9
class Tensor:
    def __init__(self, data, _prev=()):
        global _graph
        self._data = np.array(data, dtype=np.float64)
        self._prev = set(_prev)
        self._backward = lambda: None
        self._grad = np.zeros_like(self._data)
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data + other._data, _prev=(self, other))
        def backward_fn():
            self._grad += c._grad

            grad_for_other = c._grad
              
            if other._data.ndim < grad_for_other.ndim:
            
                axes_to_sum = tuple(range(grad_for_other.ndim - other._data.ndim))
                grad_for_other = np.sum(grad_for_other, axis=axes_to_sum)

            other._grad += grad_for_other
        c._backward = backward_fn
        return c
    def __radd__(self, other):

        return self + other
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data - other._data, _prev=(self, other))
        def backward_fn():
            self._grad += c._grad
            grad_for_other = c._grad
            if other._data.ndim < grad_for_other.ndim:
                axes_to_sum = tuple(range(grad_for_other.ndim - other._data.ndim))
                grad_for_other = np.sum(grad_for_other, axis=axes_to_sum)

            other._grad -= grad_for_other
        c._backward = backward_fn
        return c
    
    def __rsub__(self, other):
        return self - other
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data * other._data, _prev=(self, other))

        def backward_fn():
            self._grad += other._data * c._grad
            other._grad += self._data * c._grad
        c._backward = backward_fn
        return c
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data @  other._data, _prev=(self, other))
        
        def backward_fn():
            self._grad += c._grad @ other._data.T
            other._grad += self._data.T @ c._grad
        c._backward = backward_fn
        return c
    
    def __div__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data/ other._data, _prev=(self, other))
        def backward_fn():
            self._grad += (1.0 / other._data) * c._grad
            other._grad += (-self._data / (other._data ** 2)) * c._grad
        c._backward = backward_fn
        return c
    
    def mean(self):
        c = Tensor(self._data.mean(), _prev=(self,))
        def backward_fn():
            self._grad += np.full_like(self._data, c._grad / self._data.size)
        c._backward = backward_fn
        return c
    
    def sum(self):
        c = Tensor(self._data.sum(), _prev=(self,))
        def backward_fn():
            self._grad += c._grad
        c._backward = backward_fn 
        return c   
    def __pow__(self, n):
    
        assert isinstance(n, (int, float)), "Power operation only supports int/float exponents"
        
        c = Tensor(self._data ** n, _prev=(self,))  
        def _backward_fn():
            self._grad += c._grad * (n * self._data**(n - 1))
    
        c._backward = _backward_fn
        return c
    
    def log(self):
        c = Tensor(np.log(self._data), _prev=(self,))
        def backward_fn():
            self._grad += c._grad / (self._data + eps)
        c._backward = backward_fn
        return c
    def __neg__(self):
        c = Tensor(-self._data, _prev=(self,))
        def backward_fn():
            self._grad += -c._grad
        c._backward = backward_fn
        return c
    def ReLU(self):
        c = Tensor(np.maximum(0, self._data), _prev=(self,))
        def backward_fn():
            grad_mask = (self._data > 0).astype(np.float64)
            self._grad += grad_mask * c._grad
        c._backward = backward_fn
        return c
    def softmax(self):
        shifted = self._data - np.max(self._data, axis=1, keepdims=True) #x(i) - max(x)
        exp_shifted = np.exp(shifted) # e^(x(i) - max(x))
        softmax_out = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True) #e^(x(i) - max(x))/sum(e^(x(i) - max(x)))
        
        c = Tensor(softmax_out, _prev=(self,))
        
        def backward_fn():
            for i in range(self._data.shape[0]):
                s = softmax_out[i].reshape(-1, 1)
                jacobian = np.diagflat(s) - s @ s.T
                grad = jacobian @ c._grad[i].reshape(-1, 1)
                self._grad[i] += grad.flatten()
        
        c._backward = backward_fn
        return c
    def backward(self):
        self._grad = np.ones_like(self._data)
        topo_order = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for prev_node in node._prev:
                    build_topo(prev_node)
               
                topo_order.append(node)
        build_topo(self)
        for node in reversed(topo_order):
            node._backward()
    
    def __repr__(self):
        return self._data.__str__()
    
    def __str__(self):
        return self._data.__str__()
    def get_shape(self):
        return self._data.shape