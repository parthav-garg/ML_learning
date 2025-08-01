import numpy as np
eps = 1e-9
class Tensor:
    @staticmethod
    def _sum_to_match_shape(raw_grad, original_shape):
        """
        Sums a raw gradient so its shape matches the original tensor's shape.
        Handles broadcasting where dimensions were added or stretched.
        """
        # 1. If ndim is different, sum along the new prepended axes
        if raw_grad.ndim != len(original_shape):
            axes_to_sum = tuple(range(raw_grad.ndim - len(original_shape)))
            raw_grad = np.sum(raw_grad, axis=axes_to_sum)

        # 2. For any remaining dimensions that were stretched (from 1 to >1),
        #    sum along those axes.
        axes_to_sum = tuple([
            i for i, (dim_raw, dim_orig) in enumerate(zip(raw_grad.shape, original_shape))
            if dim_raw != dim_orig
        ])
        if axes_to_sum:
            raw_grad = np.sum(raw_grad, axis=axes_to_sum, keepdims=True)

        return raw_grad
    def __init__(self, data, _prev=()):
        self._data = np.array(data, dtype=np.float64)
        self._prev = set(_prev)
        self._backward = lambda: None
        self._grad = np.zeros_like(self._data)
        self.shape = self._data.shape
        self.ndim = self._data.ndim
    def __add__(self, other):
        """Adds two tensors.

        Args:
            other (Tensor or numeric): The tensor or numeric value to add.

        Returns:
            Tensor: A new tensor representing the sum.
        """
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data + other._data, _prev=(self, other))
        def backward_fn():
            raw_grad_self = c._grad
            raw_grad_other = c._grad

            grad_for_self = Tensor._sum_to_match_shape(raw_grad_self, self._data.shape)
            grad_for_other = Tensor._sum_to_match_shape(raw_grad_other, other._data.shape)

            self._grad += grad_for_self
            other._grad += grad_for_other
        c._backward = backward_fn
        return c
    
    def __radd__(self, other):
        """Adds two tensors.

        Args:
            other (Tensor or numeric): The tensor or numeric value to add.

        Returns:
            Tensor: A new tensor representing the sum.
        """
        return self + other
    
    def __sub__(self, other):
        """Subtracts two tensors.

        Args:
            other (Tensor or numeric): The tensor or numeric value to subtract.

        Returns:
            Tensor: A new tensor representing the difference.
        """
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data - other._data, _prev=(self, other))
        def backward_fn():
            raw_grad_self = c._grad
            raw_grad_other = c._grad

            grad_for_self = Tensor._sum_to_match_shape(raw_grad_self, self._data.shape)
            grad_for_other = Tensor._sum_to_match_shape(raw_grad_other, other._data.shape)

            self._grad += grad_for_self
            other._grad -= grad_for_other
        c._backward = backward_fn
        return c
    
    def __rsub__(self, other):
        """Subtracts two tensors.

        Args:
            other (Tensor or numeric): The tensor or numeric value to subtract.

        Returns:
            Tensor: A new tensor representing the difference.
        """
        return self - other
    
    def __mul__(self, other):
        """Multiplies two tensors.

        Args:
            other (Tensor or numeric): The tensor or numeric value to multiply.

        Returns:
            Tensor: A new tensor representing the product.
        """
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data * other._data, _prev=(self, other))

        def backward_fn():
            raw_grad_self = other._data * c._grad
            raw_grad_other = self._data * c._grad

            grad_for_self = Tensor._sum_to_match_shape(raw_grad_self, self._data.shape)
            grad_for_other = Tensor._sum_to_match_shape(raw_grad_other, other._data.shape)

            self._grad += grad_for_self
            other._grad += grad_for_other
        c._backward = backward_fn
        return c
    
    def __matmul__(self, other):
        """Computes the matrix product of two tensors.

        Args:
            other (Tensor or numeric): The tensor or numeric value to multiply.

        Returns:
            Tensor: A new tensor representing the matrix product.
        """
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data @  other._data, _prev=(self, other))
        
        def backward_fn():
            self._grad += c._grad @ other._data.T
            other._grad += self._data.T @ c._grad
        c._backward = backward_fn
        return c
    
    def __truediv__(self, other):
        """Divides two tensors.

        Args:
            other (Tensor or numeric): The tensor or numeric value to divide.

        Returns:
            Tensor: A new tensor representing the quotient.
        """
        other = other if isinstance(other, Tensor) else Tensor(other, ())
        c = Tensor(self._data/ other._data, _prev=(self, other))
        def backward_fn():
            self._grad += (1.0 / other._data) * c._grad
            other._grad += (-self._data / (other._data ** 2)) * c._grad
        c._backward = backward_fn
        return c
    
    def mean(self):
        """Computes the mean of the tensor.

        Returns:
            Tensor: A new tensor representing the mean.
        """
        c = Tensor(self._data.mean(), _prev=(self,))
        def backward_fn():
            self._grad += np.full_like(self._data, c._grad / self._data.size)
        c._backward = backward_fn
        return c
    
    def sum(self, axis=None, keepdims=False):
        """Computes the sum of the tensor over given axes."""
        c = Tensor(self._data.sum(axis=axis, keepdims=keepdims), _prev=(self,))
        
        def backward_fn():
            
            if axis is None:
                self._grad += np.broadcast_to(c._grad, self._data.shape)
            else:
                sum_axes = axis if isinstance(axis, (list, tuple)) else (axis,)
                
                target_shape = [1] * self._data.ndim
                
                grad_shape = list(c._grad.shape)

                j = 0
                for i in range(self._data.ndim):
                    if i not in sum_axes:
                        target_shape[i] = grad_shape[j]
                        j += 1
                reshaped_grad = c._grad.reshape(target_shape)
                self._grad += reshaped_grad

        c._backward = backward_fn
        return c
    def __pow__(self, n):
        """Computes the power of the tensor.

        Args:
            n (int or float): The exponent to raise the tensor to.

        Returns:
            Tensor: A new tensor representing the result.
        """
        assert isinstance(n, (int, float)), "Power operation only supports int/float exponents"
        
        c = Tensor(self._data ** n, _prev=(self,))  
        def _backward_fn():
            self._grad += c._grad * (n * self._data**(n - 1))
        c._backward = _backward_fn
        return c
    
    def log(self):
        """Computes the natural logarithm of the tensor.

        Returns:
            Tensor: A new tensor representing the natural logarithm.
        """
        global eps
        c = Tensor(np.log(self._data), _prev=(self,))
        c += eps  
        def backward_fn():
            self._grad += c._grad / (self._data)
        c._backward = backward_fn
        return c
    
    def __neg__(self):
        """Computes the negation of the tensor.

        Returns:
            Tensor: A new tensor representing the negation.
        """
        c = Tensor(-self._data, _prev=(self,))
        def backward_fn():
            self._grad += -c._grad
        c._backward = backward_fn
        return c
    
    def ReLU(self):
        """Computes the ReLU (Rectified Linear Unit) of the tensor.

        Returns:
            Tensor: A new tensor representing the ReLU activation.
        """
        c = Tensor(np.maximum(0, self._data), _prev=(self,))
        def backward_fn():
            grad_mask = (self._data > 0).astype(np.float64)
            self._grad += grad_mask * c._grad
        c._backward = backward_fn
        return c
    
    def softmax(self):
        """Computes the softmax of the tensor.

        Returns:
            Tensor: A new tensor representing the softmax activation.
        """
        shifted = self._data - np.max(self._data, axis=1, keepdims=True) #x(i) - max(x)
        exp_shifted = np.exp(shifted) # e^(x(i) - max(x))
        softmax_out = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True) #e^(x(i) - max(x))/sum(e^(x(i) - max(x)))
        
        c = Tensor(softmax_out, _prev=(self,))
        
        def backward_fn():
            for i in range(self._data.shape[0]):
                s = softmax_out[i].reshape(-1, 1)
                a = softmax_out
                grad_i = a[i] * (c._grad[i] - (a[i] * c._grad[i]).sum())
                grad_norm = np.linalg.norm(grad_i)
                if grad_norm > 10.0:
                    grad_i = grad_i / grad_norm * 10.0
                    
                self._grad[i] += grad_i
        
        c._backward = backward_fn
        return c
    
    def reshape(self, *new_shape):
        """Reshapes the tensor to a new shape.

        Args:
            *new_shape: The new shape to reshape the tensor to.

        Returns:
            Tensor: A new tensor with the specified shape.
        """
        c = Tensor(self._data.reshape(new_shape), _prev=(self,))
        
        def backward_fn():
            self._grad += c._grad.reshape(self._data.shape)
        
        c._backward = backward_fn
        return c
    
    def create_padding_width(self, padding, *axis):
        """Creates a padding width for the tensor.

        Args:
            padding (int): The number of zeros to pad.
            axis (int): The axis along which to pad.

        Returns:
            tuple: A tuple representing the padding width.
        """
        padding_width = np.array([(0, 0)] * self._data.ndim)
        for ax in axis:
            padding_width[ax] = (padding, padding)
        return tuple(padding_width)
    
    def pad_1d(self, padding, dims=[1,]):
        """Pads the tensor with zeros on both sides.

        Args:
            padding (int): The number of zeros to pad on each side.

        Returns:
            Tensor: A new tensor with the specified padding.
        """
        padding_width = self.create_padding_width(padding, dims)
        padded_data = np.pad(self._data, padding_width, mode='constant', constant_values=0)
        c = Tensor(padded_data, _prev=(self,))
        def backward_fn():
            slicing_list = np.array([slice(None)] * self._data.ndim)
            for dim in dims:
                if dim < self._data.ndim:
                    slicing_list[dim] = slice(padding, -padding if padding != 0 else None)

            slicing_tuple = tuple(slicing_list)
            self._grad += c._grad[slicing_tuple]
        c._backward = backward_fn
        return c
    
    def __getitem__(self, index):
        c = Tensor(self._data[index], _prev=(self,))
        def backward_fn():
            self._grad[index] += c._grad
        c._backward = backward_fn
        return c
    def __setitem__(self, index, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        self._data[index] =  other._data
        
    
    def backward(self):
        """Computes the backward pass for the tensor.
        """
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
        """Returns a string representation of the tensor."""
        return self._data.__str__()
    
    def get_shape(self):
        """Returns the shape of the tensor."""
        return self._data.shape