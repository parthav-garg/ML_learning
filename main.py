from base.tensor import Tensor
import numpy as np
if __name__ == "__main__":
    a = Tensor(np.array([1,2,3,4,5]), ())
    b = Tensor(np.array([1,2,3,4,5]), ())
    c = a + b
    d = Tensor(np.array([1,1,1,1,1]), ())
    e = c + d
    e.grad = np.array([1,2,1,2,1])
    print(d.grad)
    print(c.grad)
    print(a.grad)
    print(b.grad)
    