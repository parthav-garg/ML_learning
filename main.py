from base.tensor import Tensor
import numpy as np
from nn import Linear
from nn import Module
from nn import optimiser
from time import sleep
import random
from base.tensor import _graph
if __name__ == "__main__":
    # 1. Create the multi-feature dataset
    num_samples = 1000
    num_features = 3
    x = np.random.rand(num_samples, num_features) * 10
    y = 5*x[:, 0] + 7*x[:, 1] + 10*x[:, 2] + 15

    # 2. Calculate statistics for standardization (using axis=0 for x)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    y_mean = np.mean(y)
    y_std = np.std(y)

    # 3. Apply standardization
    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std
    
    # 4. Define the model with the correct input layer shape
    l1 = Linear(3, 5) # Input features = 3
    l2 = Linear(5, 1) # Output features = 1
    
    # 5. Your training loop (no changes needed here!)
    optim = optimiser([l1, l2], .01) # A smaller learning rate is often safer
    for i in range(2000): # More iterations might be needed
        index = random.randint(0, num_samples - 1)
        # x_scaled[index] is now a row with 3 features
        out1 = l1(Tensor(x_scaled[index].reshape(1, -1)))
        out2 = l2(out1)
        loss = (out2 - Tensor(y_scaled[index].reshape(1, -1)))**2
        if i % 200 == 0: # Print loss occasionally
            print(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    # 6. Test the trained model with a 3-feature input
    print("\n--- Final Test ---")
    original_x = np.array([2.0, 3.0, 4.0])

    # Scale the test input using the training statistics
    test_x_scaled = (original_x - x_mean) / x_std
    input_tensor = Tensor(np.array(test_x_scaled).reshape(1, -1))
    
    # Get the prediction and un-scale it
    scaled_prediction = l2(l1(input_tensor))
    final_prediction = (scaled_prediction._data * y_std) + y_mean

    # Compare to the ground truth
    true_y = 5*original_x[0] + 7*original_x[1] + 10*original_x[2] + 15
    print(f"Input: {original_x}")
    print(f"Final prediction is: {final_prediction[0][0]:.4f}")
    print(f"True answer is: {true_y}")