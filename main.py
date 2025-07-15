from base.tensor import Tensor
import numpy as np
from nn import Linear
from nn import Module
from nn import optimiser
if __name__ == "__main__":
    x = np.array([1,2,3,4,5])
    x = Tensor(x.reshape(1, -1))
    #print(x.get_shape())
    l1 = Linear(5, 5)
    l2 = Linear(5, 1)
    print("l1=", l1.parameters()["weights"])
    
    print("l2=", l2.parameters()["weights"])
    
    out1 = l1(x)
    out2 = l2(out1)
    loss = (out2 - Tensor([.9])) * (out2 - Tensor([.9]))
    optim = optimiser([l1, l2], .01)
    print(loss)
    #print(out1.get_shape())
    out1.backward()
    optim.step()
    print("l1=", l1.parameters()["weights"])
    print("l2=", l2.parameters()["weights"])
    