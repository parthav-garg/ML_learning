
import numpy as np



if __name__ == "__main__":
    x = np.array([[[1,2, 3], [1,2,3], [1,2,3]], [[1,2, 3], [1,2,3], [1,2,3]], [[1,2, 3], [1,2,3], [1,2,3]]])
    y = np.array([[[1,2, 3], [2,1,3], [2,2,3]]])
    print(y.shape)
    print(x.shape)
    z = x * y
    z2 = y * z
    print(z2)
    print(z)