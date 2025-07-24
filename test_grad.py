from base.tensor import Tensor
import numpy as np
x = Tensor(np.array([1, 2, 3]).reshape(1, 3, 1))
print(x[0, 0:2, :])