from base.tensor import Tensor
import numpy as np
x = Tensor(np.array([1, 2, 3]).reshape(1, 3, 1))
print(x)
print(x.get_shape())
y = x.pad_1d(1, [1, 2])
print(y)
y._grad = np.random.random(y._data.shape)
print(y._grad)
y._backward()
print(x._grad)