import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Assuming your files are in the same structure as before
from base.tensor import Tensor
from nn import Module, Linear, ReLU, CrossEntropyloss, optimizer, Dropout
from DataLoader import DataLoader

# ===================================================================
# 1. LOAD THE MNIST DATASET
# ===================================================================
print("Loading MNIST dataset...")
# MNIST is a dataset of 28x28 images of handwritten digits (0-9)
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

# The data is 70,000 images, each flattened to 784 pixels (28*28)
X = mnist['data'].astype(np.float64)
# Labels are strings '0' through '9', convert them to integers
y = mnist['target'].astype(int)

# Normalize pixel values from 0-255 to 0-1 for better training stability
X /= 255.0

# Split into training and a smaller test set for faster execution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

num_classes = 10
# One-hot encode the training labels
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]
print("Dataset loaded and prepared.")

# ===================================================================
# 2. DEFINE A SIMPLE MLP MODEL FOR FASTER TRAINING
# ===================================================================
class MLP_Model(Module):
    def __init__(self):
        super().__init__()
        # An MLP is just a sequence of Linear layers and activations
        self.fc1 = Linear(784, 128) # Input: 784 pixels, Hidden layer: 128 neurons
        self.dropout = Dropout(p=.2)
        self.relu1 = ReLU()
        self.fc2 = Linear(128, num_classes) # Output: 10 neurons for 10 digits

    def forward(self, x):
        # No reshaping needed for MLP, input is already flat
        x = self.fc1(x)
        x = self.relu1(self.dropout(x))
        logits = self.fc2(x)
        return logits

# ===================================================================
# 3. HELPER FUNCTION FOR THE TRAINING LOOP
# ===================================================================
def train_model(model, optim, data_loader, epochs=10):
    """A helper function to run the training loop and record losses."""
    criterion = CrossEntropyloss()
    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in data_loader:
            x_tensor = Tensor(x_batch)
            y_tensor = Tensor(y_batch)

            # --- FORWARD PASS ---
            logits = model(x_tensor)
            
            # --- LOSS AND BACKWARD ---
            probs = logits.softmax()
            loss = criterion(probs, y_tensor)
            epoch_loss += loss._data
            
            loss.backward()
            optim.step()
            optim.zero_grad()
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        epoch_losses.append(avg_epoch_loss)
        # Use the optimizer's class name for a clean print statement
        print(f"[{optim.__class__.__name__}] Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
    return epoch_losses

# ===================================================================
# 4. RUN THE COMPARISON EXPERIMENT
# ===================================================================
# Hyperparameters for the experiment
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64

# Create the data loader
data = DataLoader(X_train, y_train_one_hot, batch_size=BATCH_SIZE, shuffle=True)

# --- Train with SGD ---
print("\n--- TRAINING WITH ADAM---")
model = MLP_Model()
optim = optimizer.Adam(model, lr=LEARNING_RATE)
train_model(model,optim, data)

print("\n--- EVALUATING MODEL ---")
model.eval() # Make sure to set model to evaluation mode

# Use the test DataLoader you already created
test_data = DataLoader(X_test, y_test_one_hot, batch_size=BATCH_SIZE, shuffle=False)
correct_predictions = 0
total_samples = 0

# THIS IS THE CORRECT WAY TO ITERATE:
# Use a standard `for` loop. It's cleaner and works perfectly with the DataLoader.
for x_batch, y_batch in test_data:
    
    logits = model(x_batch)
    shifted = logits - np.max(logits, axis=1, keepdims=True) #x(i) - max(x)
    exp_shifted = np.exp(shifted) # e^(x(i) - max(x))
    softmax_out = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    probs = softmax_out
    
    # Get the predicted class index for each item in the batch
    predicted_classes = np.argmax(probs._data, axis=1)
    # Get the true class index for each item in the batch
    true_classes = np.argmax(y_batch, axis=1)
    
    # Add the number of correct predictions in this batch
    correct_predictions += np.sum(predicted_classes == true_classes)
    total_samples += len(x_batch)

# Calculate final accuracy
accuracy = (correct_predictions / total_samples) * 100
print(f"Accuracy on {total_samples} test samples = {accuracy:.2f}%")
