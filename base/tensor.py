import numpy as np

_graph = []

class Tensor:
    def __init__(self, data, _prev=()):
        global _graph
        self._data = np.array(data, dtype=np.float64)
        self._prev = set(_prev)
        self._backward = lambda: None
        self._grad = np.zeros_like(self._data)
        _graph.append(self)
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data + other._data, _prev=(self, other))
        def backward_fn():
            self._grad += c._grad

        # For other (the bias), we must sum the gradient across the batch dimension.
        # This "un-broadcasts" the gradient to match the bias's shape.
            grad_for_other = c._grad
            
            # This handles the case where a matrix is added to a vector (bias)
            if other._data.ndim < grad_for_other.ndim:
                # Sum along the axes that don't exist in the bias tensor
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

        # For other (the bias), we must sum the gradient across the batch dimension.
        # This "un-broadcasts" the gradient to match the bias's shape.
            grad_for_other = c._grad
            
            # This handles the case where a matrix is added to a vector (bias)
            if other._data.ndim < grad_for_other.ndim:
                # Sum along the axes that don't exist in the bias tensor
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
        #print(self._data.shape, other._data.shape)
        c = Tensor(self._data @  other._data, _prev=(self, other))
        
        def backward_fn():
           # print("c shape", c._grad.shape)
            #print("self shape", self._data.shape)
            #print(self._data, self._data.shape)
            self._grad += c._grad @ other._data.T
            other._grad += self._data.T @ c._grad
        c._backward = backward_fn
        return c
    def __div__(self, other):
        c = Tensor(self._data/ other._data, _prev=(self, other))
        return c
    def __pow__(self, n):
    
        assert isinstance(n, (int, float)), "Power operation only supports int/float exponents"
        
        c = Tensor(self._data ** n, _prev=(self,))  
        def _backward_fn():
            self._grad += c._grad * (n * self._data**(n - 1))
    
        c._backward = _backward_fn
        
        return c
    def backward(self):
        self._grad = np.ones_like(self._data)

        # 2. Build the topological order of nodes in the computational graph
        # This ensures we process nodes in the correct order (children before parents)
        topo_order = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                # Recursively visit parents first
                for prev_node in node._prev:
                    build_topo(prev_node)
                # Add current node to the list after all its parents are added
                topo_order.append(node)

        build_topo(self) # Start building from the current tensor (which is your loss)

        # 3. Propagate gradients backwards through the topologically sorted nodes
        # Iterate in reverse order to go from loss back to inputs/parameters
        for node in reversed(topo_order):
            node._backward()
    
    def __repr__(self):
        return self._data.__str__()
    
    def __str__(self):
        return self._data.__str__()
    def get_shape(self):
        return self._data.shape