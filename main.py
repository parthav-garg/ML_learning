import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Assuming your files are in the same structure as before
from base.tensor import Tensor
from nn import Module, Linear, ReLU, CrossEntropyloss, optimizer
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

print("Dataset loaded and prepared.")

# ===================================================================
# 2. DEFINE A SIMPLE MLP MODEL FOR FASTER TRAINING
# ===================================================================
class MLP_Model(Module):
    def __init__(self):
        super().__init__()
        # An MLP is just a sequence of Linear layers and activations
        self.fc1 = Linear(784, 128) # Input: 784 pixels, Hidden layer: 128 neurons
        self.relu1 = ReLU()
        self.fc2 = Linear(128, num_classes) # Output: 10 neurons for 10 digits

    def forward(self, x):
        # No reshaping needed for MLP, input is already flat
        x = self.fc1(x)
        x = self.relu1(x)
        logits = self.fc2(x)
        return logits

# ===================================================================
# 3. HELPER FUNCTION FOR THE TRAINING LOOP
# ===================================================================
def train_model(model, optim, data_loader, epochs=10):
    """A helper function to run the training loop and record losses."""
    criterion = CrossEntropyloss()
    epoch_losses = []
    
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
print("\n--- TRAINING WITH SGD ---")
sgd_model = MLP_Model()
sgd_optimizer = optimizer.SGD(sgd_model, lr=LEARNING_RATE)
sgd_losses = train_model(sgd_model, sgd_optimizer, data, epochs=EPOCHS)


# --- Train with Adam ---
print("\n--- TRAINING WITH ADAM ---")
# CRITICAL: We must create a new model to reset the weights from scratch!
adam_model = MLP_Model() 
adam_optimizer = optimizer.Adam(adam_model, lr=LEARNING_RATE)
adam_losses = train_model(adam_model, adam_optimizer, data, epochs=EPOCHS)


# ===================================================================
# 5. PLOT THE RESULTS FOR COMPARISON
# ===================================================================
print("\nPlotting results...")
plt.figure(figsize=(12, 7))
plt.plot(range(1, EPOCHS + 1), sgd_losses, marker='o', linestyle='--', label='SGD Loss')
plt.plot(range(1, EPOCHS + 1), adam_losses, marker='o', linestyle='-', label='Adam Loss')
plt.title('Adam vs. SGD Optimizer Performance', fontsize=16)
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss')
plt.xticks(range(1, EPOCHS + 1))
plt.legend()
plt.grid(True)
plt.show()