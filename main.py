from sklearn.datasets import fetch_openml
import numpy as np
from base.tensor import Tensor
from nn import Linear, CrossEntropyloss, optimiser
import random

if __name__ == "__main__":
    # 1. Load MNIST as (70000, 784)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype(int)

    # 2. Normalize pixel values [0, 1]
    X = X.astype(np.float64) / 255.0

    # 3. Subsample for faster testing
    num_samples = 60000
    indices = np.random.choice(len(X), num_samples, replace=False)
    X, y = X[indices], y[indices]

    # 4. One-hot encode labels
    num_classes = 10
    y_one_hot = np.eye(num_classes)[y]

    # 5. Model: 784 → 128 → 64 → 10
    l1 = Linear(784, 128)
    l2 = Linear(128, 64)
    l3 = Linear(64, 10)

    criterion = CrossEntropyloss()
    optim = optimiser.SGD([l1, l2, l3], lr=0.01)

    # 6. Training loop (single sample SGD)
    for i in range(60000):
        idx = random.randint(0, num_samples - 1)
        x_tensor = Tensor(X[idx].reshape(1, -1))
        y_tensor = Tensor(y_one_hot[idx].reshape(1, -1))

        out = l1(x_tensor).ReLU()
        out = l2(out).ReLU()
        logits = l3(out)
        probs = logits.softmax()

        loss = criterion(probs, y_tensor)

        if i % 1000 == 0:
            print(f"Step {i} Loss: {loss}")
            optim._lr *= 0.9

        loss.backward()
        optim.step()
        optim.zero_grad()

 
    correct = 0
    for idx in range(5000):
        x_tensor = Tensor(X[idx].reshape(1, -1))
        out = l1(x_tensor).ReLU()
        out = l2(out).ReLU()
        logits = l3(out)
        probs = logits.softmax()
        pred = np.argmax(probs._data)
        if pred == y[idx]:
            correct += 1

    print(f"\nAccuracy on first 5000 samples: {correct/50}%") #94.56%