import numpy as np
from base.tensor import Tensor
from numpy.lib.stride_tricks import sliding_window_view
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
    def train(self):
        for module in self._modules.values():
            module.train()
            
    def eval(self):
        for module in self._modules.values():
            module.eval()
        
class Linear(Module):
    """Class to create a Linear Layere"""
    def __init__(self, in_features, out_features):
        self._in = in_features
        self._out = out_features
        fan_in = in_features
        stddev = np.sqrt(2.0 / fan_in)
        self._w = Tensor(np.random.normal(0,stddev, size=(in_features, out_features)))
        self._b = Tensor(np.zeros(out_features))
        self.training = True
    def forward(self, input):
        """Forward pass, on the input data
        Args:
            input (Tensor): The input tensor.
        Returns:
            Tensor: The output tensor of the calculation of inpput @ weights + biases.
        """
        return (input @ self._w) + self._b if self.training else (input @ self._w._data) + self._b._data
    def layer_values(self):
        """Returns the weights, biases and the gradient tensor of the layer."""
        return {"weights" : self._w, "biases": self._b,"grad": self._w._grad}
    def parameters(self):
        """Returns the parameters of the layer."""
        return [self._w, self._b]
    def train(self):
        self.training = True
    def eval(self):
        self.training = False
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.training = False

    def forward(self, input_tensor):
        """Applies the ReLU activation function to the input tensor."""
        input_tensor = input_tensor if isinstance(input_tensor, Tensor) else Tensor(input_tensor)
        return input_tensor.ReLU()
    def train(self):
        self.training = True
    def eval(self):
        self.training = False
class Conv1D(Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size
        stddev = np.sqrt(2.0 / fan_in)
        self.kernels = Tensor(np.random.normal(0, stddev, size=(out_channels, in_channels, kernel_size)))
        self.bias = Tensor(np.zeros((1, out_channels, 1)))
        self.training = True
        
    def forward(self, input_tensor):
        if len(input_tensor.shape) < 3:
            shape = input_tensor.shape
            input_tensor = input_tensor.reshape(1, shape[0], shape[1])
        if self.padding > 0:
            input_tensor = input_tensor if isinstance(input_tensor, Tensor) else Tensor(input_tensor)
            padded_data = Tensor.pad_1d(input_tensor, self.padding, dims=[2])
        else:
            padded_data = input_tensor if isinstance(input_tensor, Tensor) else Tensor(input_tensor)
        S = self.stride
        B, in_C, L = padded_data.shape
        out_C, _, K = self.kernels.shape
        Lout = (L - K) // S + 1
        windows = sliding_window_view(padded_data._data, window_shape=K, axis=(2))
        in_unroll = windows.transpose(0, 2, 1, 3).reshape(B, Lout, in_C * K)
        k_flat = self.kernels._data.reshape(out_C, in_C * K)
        output = (in_unroll @ k_flat.T).reshape(B, out_C, Lout)
        if not self.training:
            return output + self.bias._data
        output = Tensor(output, _prev=(input_tensor, self.kernels))
        def backward_fn():
            self.kernels._grad += (output._grad.transpose(0, 2, 1) 
                       .reshape(B * Lout, out_C).T 
                       @ in_unroll.reshape(B * Lout, in_C * K) 
                      ).reshape(out_C, in_C, K)
            grad_unroll = output._grad.transpose(0, 2, 1).reshape(B * Lout, out_C) @ k_flat  # (B*Lout, in_C * K)
            grad_unroll = grad_unroll.reshape(B, Lout, in_C, K).transpose(0, 2, 3, 1)  # (B, in_C, K, Lout)

            # Fold back into input shape
            dx_padded = np.zeros_like(padded_data._data)
            for k in range(K):
                idx = np.arange(Lout) * S + k
                dx_padded[:, :, idx] += grad_unroll[:, :, k, :]

            # If padded, remove it
            if self.padding > 0:
                padded_data._grad += dx_padded
                padded_data._backward()
            else:
                input_tensor._grad += dx_padded
                    
        output._backward = backward_fn
        return output + self.bias
    
    def parameters(self):
        return [self.kernels, self.bias]
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

class MaxPool1D(Module):
    def __init__(self, kernel_size=1,padding=0, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    def forward(self, input_tensor):
        input_tensor = input_tensor if isinstance(input_tensor, Tensor) else Tensor(input_tensor)
        if self.padding > 0:
            padded_data = Tensor.pad_1d(input_tensor, self.padding, dims=[2])   
        else:
            padded_data = input_tensor
            
        B, C, L = padded_data.shape
        K = self.kernel_size
        S = self.stride
        Lout = (L - K) // S + 1
        windows = sliding_window_view(padded_data._data, window_shape=K, axis=(2))
        windows = windows[:, :, ::S, :]
        out = windows.max(axis=3)
        #print(out.shape)
        indices = windows.argmax(axis=3)
        out = Tensor(out, _prev=(padded_data,))
        def backward_fn():
            grad = out._grad  # (B, C, L_out)
            B_idx, C_idx, L_out_idx = np.meshgrid(
                np.arange(B), np.arange(C), np.arange(Lout), indexing='ij'
            )

            L_idx = L_out_idx * S + indices  # convert window offset to original indices

            if padded_data._grad is None:
                padded_data._grad = np.zeros_like(padded_data._data)

            np.add.at(padded_data._grad, (B_idx, C_idx, L_idx), grad)
            padded_data._backward()
        out._backward = backward_fn
        return out
    
    def parameters(self):
        return []
class Dropout(Module):
    
    def __init__(self, p=.2):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError("Dropout probablity must be between 0.0 and 1.0")
        self.p = p
        self.training = False
        self.mask = 0
    
    def forward(self, input_tensor):
        if not self.training:
            return input_tensor
        input_tensor = input_tensor if  isinstance(input_tensor, Tensor) else Tensor(input_tensor)
        keep_prob = 1 - self.p
        self.mask = np.random.binomial(1, keep_prob, size=input_tensor.get_shape()) / keep_prob
        output_data = input_tensor._data * self.mask
        c = Tensor(output_data, _prev=(input_tensor,))

        def backward_fn():
            input_tensor._grad += c._grad * self.mask
            
        c._backward = backward_fn
        return c  
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

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
                
    class Adam():
        def __init__(self, model, lr=.001, b1=.9, b2=.999, eps=1e-8):
            self._model = model
            self.lr = lr
            self.beta1 = b1
            self.beta2 = b2
            self.eps = eps
            self.m = {}
            self.v = {}
            for p in self._model.parameters():
                self.m[p] = np.zeros_like(p._grad)
                self.v[p] = np.zeros_like(p._grad)
            self.t = 0
        
        def step(self):
            self.t+= 1
            for p in self._model.parameters():
                self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p._grad
                self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (p._grad ** 2)
                m_t = self.m[p]/(1 - self.beta1 ** self.t)
                v_t = self.v[p]/(1 - self.beta2 ** self.t)
                p._data = p._data - self.lr * m_t/(np.sqrt(v_t) + self.eps)
                
        
        def zero_grad(self):
            for parameter in self._model.parameters():
                parameter._grad = np.zeros_like(parameter._grad)
                