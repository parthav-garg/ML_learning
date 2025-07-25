from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from base.tensor import Tensor
# Make sure to import your Conv1D class!
from nn import Linear, CrossEntropyloss, optimizer, Module, ReLU, Conv1D
from DataLoader import DataLoader

# ===================================================================
# 1. LOAD THE 1D-FRIENDLY ECG5000 DATASET
# ===================================================================
# Fetch the data
ecg = fetch_openml('ECG5000', version=1, as_frame=False)

# The data is in 'data', labels in 'target'
X = ecg['data'].astype(np.float64)
y = ecg['target'].astype(int)

# --- Important: The labels are 1-5, we need them to be 0-4 ---
y = y - 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_classes = 5
y_train_one_hot = np.eye(num_classes)[y_train]

# Use your excellent DataLoader
data = DataLoader(X_train, y_train_one_hot, batch_size=32, shuffle=True)
num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]

# ===================================================================
# 2. DEFINE A CONV MODEL SUITABLE FOR THE ECG DATA
# ===================================================================
class ECG_Conv_Model(Module):
    def __init__(self):
        super().__init__()
        # Each ECG signal has 140 time steps and 1 channel.

        # Layer 1: Takes 1 input channel, produces 16 feature maps.
        # Padding 'same' equivalent: padding = (kernel_size - 1) // 2 = (5-1)//2 = 2
        self.conv1 = Conv1D(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = ReLU()

        # Layer 2: Takes the 16 feature maps, produces 32 new ones.
        # Padding 'same' equivalent: padding = (3-1)//2 = 1
        self.conv2 = Conv1D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()

        # After convolutions, the output shape is (batch_size, 32, 140).
        # We flatten this for the final linear layer.
        self.fc1 = Linear(32 * 140, num_classes)

    def forward(self, x):
        batch_size = x.get_shape()[0]

        # 1. Reshape input from (batch, 140) to (batch, 1, 140) for Conv1D
        x = x.reshape(batch_size, 1, 140)

        # 2. Pass through conv layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        # 3. Flatten the output for the linear layer
        x = x.reshape(batch_size, 32 * 140)

        # 4. Final classification layer
        logits = self.fc1(x)
        return logits

# ===================================================================
# 3. INSTANTIATE MODEL, LOSS, AND OPTIMIZER
# ===================================================================
model = ECG_Conv_Model()
criterion = CrossEntropyloss()
# Using a smaller learning rate is often a good starting point for CNNs
optim = optimizer.Adam(model, lr=0.001)

# ===================================================================
# 4. TRAINING LOOP
# ===================================================================
epochs = 20 # Let's train for a few more epochs on this smaller dataset

for epoch in range(epochs):
    epoch_loss = 0.0
    for x_batch, y_batch in data:
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
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data):.4f}")

# ===================================================================
# 5. EVALUATION LOOP
# ===================================================================
print("\nStarting evaluation on the unseen test set...")
correct = 0
# Create a test DataLoader for easier batching
test_data = DataLoader(X_test, np.eye(num_classes)[y_test], batch_size=32)

for x_batch, y_batch_one_hot in test_data:
    x_tensor = Tensor(x_batch)
    y_batch_labels = np.argmax(y_batch_one_hot, axis=1) # Get original labels

    logits = model(x_tensor)
    probs = logits.softmax()
    preds = np.argmax(probs._data, axis=1)
    correct += np.sum(preds == y_batch_labels)

accuracy = (correct / num_test_samples) * 100
print(f"\nAccuracy on {num_test_samples} unseen test samples: {accuracy:.2f}%")