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
                jacobian = np.diagflat(s) - s @ s.T
                grad = jacobian @ c._grad[i].reshape(-1, 1)
                self._grad[i] += grad.flatten()
        
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
        
    def conv_1d(self, kernels, stride=1, padding=0):
        """Performs a 1D convolution operation on the tensor.

        Args:
            kernel (Tensor): The kernel to convolve with.
            stride (int): The stride of the convolution.
            padding (int): The amount of zero-padding to apply.

        Returns:
            Tensor: A new tensor representing the result of the convolution.
        """
        if padding > 0:
            padded_data = self.pad_1d(padding, dims=[2]) 

        else:
            padded_data = self
        
        batch_size, in_channels, length = padded_data.get_shape()
        out_channels, _, kernel_size = kernels.get_shape()
        output_length = (length - kernel_size) // stride + 1
        np_outputs = np.zeros((batch_size, out_channels, output_length))
        tensor_comp = []
        
        for i in range(output_length):
            start = i * stride
            end = start + kernel_size
            window = padded_data[:, :, start:end]
            ccr = (window.reshape(batch_size, 1, in_channels, kernel_size) * kernels.reshape(1, out_channels, in_channels, kernel_size)).sum(axis=(2, 3))
            #tensor_comp.append(ccr)
            np_outputs[:, :, i] = ccr._data

        parents = {self, kernels}
        c = Tensor(np_outputs, _prev=parents)
        
        def backward_fn():
            flipped_kernel = np.flip(kernels._data, axis=(0,1))
            for i in range(output_length):
                #tensor = tensor_comp[i]
                #grad_slice = c._grad[: , : , i]
                #tensor._grad += grad_slice.reshape(tensor.get_shape())
                #tensor._backward()
                start = i * stride
                end = start + kernel_size
                window = padded_data[:, :, start:end]
                kernels._grad += window * c._grad
                
        
        c._backward = backward_fn
        return c
    
    def conv_2d(self, kernels, stride=(1, 1), padding=(0, 0)):
        
        if padding[0] > 0  or padding[1] > 0:
            padded_data = self.pad_1d(padding[0], dims=[2])
            padded_data = self.pad_1d(padding[1], dims=[3])  

        else:
            padded_data = self
        
        batch_size, in_channels, height, width = padded_data.get_shape()
        out_channels, _, kernel_height, kernel_width = kernels.get_shape()
        h_out = height - kernel_height // stride[0] + 1
        w_out = width - kernel_width // stride[1] + 1
        np_outputs = np.zeros((batch_size, out_channels, h_out, w_out))
        tensor_comp = []
        
        for i in range(h_out):
            tensor_comp_i = []
            h_start = i * stride[0]
            h_end = h_start + kernel_height
            for j in range(w_out):
                w_start = j * stride[1]
                w_end = w_start + kernel_width
                window = padded_data[:, :, h_start: h_end, w_start:w_end]
                ccr = (window.reshape(batch_size, 1, in_channels, kernel_height, kernel_width) * kernels.reshape(1, out_channels, in_channels, kernel_height, kernel_width)).sum(axis=(2, 3, 4))
                #tensor_comp_i.append(ccr)
                np_outputs[:, :, i, j] = ccr._data
            tensor_comp.append(tensor_comp_i)
        parents = {self, kernels}
        c = Tensor(np_outputs, _prev=parents)
        def backward_fn():
            for i in range(h_out):
                for j in range(w_out):
                    tensor = tensor_comp[i][j]
                    grad_slice = c._grad[: , : , i, j]
                    tensor._grad += grad_slice.reshape(tensor.get_shape())
                    tensor._backward()
            
        c._backward = backward_fn
        return c
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