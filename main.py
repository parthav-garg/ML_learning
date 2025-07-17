from sklearn.datasets import fetch_openml
import numpy as np
from base.tensor import Tensor
from nn import Linear, CrossEntropyloss, optimizer, Module
import random

if __name__ == "__main__":
    # 1. Load and Split Data (Unchanged)
    # ... (same as before) ...
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype(int)
    X = X.astype(np.float64) / 255.0
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    num_train_samples, num_test_samples = X_train.shape[0], X_test.shape[0]
    num_classes = 10
    y_train_one_hot = np.eye(num_classes)[y_train]
    # In your main script, define your model class
    class Linear_Model(Module):
        def __init__(self):
            super().__init__()
            # Because Linear is a Module, these layers are auto-registered
            self.l1 = Linear(784, 128)
            self.l2 = Linear(128, 64)
            self.l3 = Linear(64, 10)

        def forward(self, x):
            """
            Defines the data flow through the layers.
            """
            x = self.l1(x).ReLU()
            x = self.l2(x).ReLU()
            logits = self.l3(x)
            return logits
    # ===================================================================
    # 2. INSTANTIATE YOUR NEW PYTORCH-LIKE MODEL
    # ===================================================================
    model = Linear_Model() # So clean!
    criterion = CrossEntropyloss()
    # The optimizer call is now perfect. It gets all parameters automatically.
    optim = optimizer.SGD(model.parameters(), lr=0.01)

    # 3. Training Loop (Now uses the model object)
    batch_size = 64
    epochs = 10

    for epoch in range(epochs):
        # ... (shuffling code is the same) ...
        permutation = np.random.permutation(num_train_samples)
        X_train_shuffled = X_train[permutation]
        y_train_one_hot_shuffled = y_train_one_hot[permutation]
        
        epoch_loss = 0.0
        for i in range(0, num_train_samples, batch_size):
            x_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_one_hot_shuffled[i:i+batch_size]
            x_tensor = Tensor(x_batch)
            y_tensor = Tensor(y_batch)

            # --- FORWARD PASS ---
            logits = model(x_tensor) # Call the model like a function
            
            # --- LOSS AND BACKWARD ---
            probs = logits.softmax()
            loss = criterion(probs, y_tensor)
            epoch_loss += loss._data
            loss.backward()
            optim.step()
            optim.zero_grad()
            
        optim._lr *= 0.9
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (num_train_samples / batch_size)}")

    # 4. Evaluation Loop (Also uses the model object)
    print("\nStarting evaluation on the unseen test set...")
    correct = 0
    for i in range(0, num_test_samples, batch_size):
        x_batch = X_test[i:i+batch_size]
        y_batch_labels = y_test[i:i+batch_size]
        x_tensor = Tensor(x_batch)

        logits = model(x_tensor)
        probs = logits.softmax()
        preds = np.argmax(probs._data, axis=1)
        correct += np.sum(preds == y_batch_labels)

    accuracy = (correct / num_test_samples) * 100
    print(f"\nAccuracy on {num_test_samples} unseen test samples: {accuracy:.2f}%")