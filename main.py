import numpy as np
import matplotlib.pyplot as plt
from tslearn.datasets import UCR_UEA_datasets
from sklearn.model_selection import train_test_split

from base.tensor import Tensor
from nn import Module, Conv1D, ReLU, Linear, CrossEntropyloss, optimizer, Dropout
from DataLoader import DataLoader

# ================================================================
# 1. Load ECG5000 Dataset (1D time series)
# ================================================================
print("Loading ECG5000 dataset...")
ucr = UCR_UEA_datasets()
X, y, _, _ = ucr.load_dataset("ECG5000")

# Normalize
X = X.astype(np.float64)
X -= X.mean()
X /= X.std()

# Labels: make zero-based (e.g., 0 to 4)
y = y.astype(int)
y -= y.min()
num_classes = len(np.unique(y))

# Reshape to (B, 1, T)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# One-hot encode targets
y_one_hot = np.eye(num_classes)[y]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42, stratify=y
)

print("ECG5000 dataset loaded.")

# ================================================================
# 2. Conv1D-based CNN Model
# ================================================================
class CNN1D_Model(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1D(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(p=0.1)


        self.flatten_dim = 32 * 140  # length remains 140
        self.fc1 = Linear(self.flatten_dim, 256)
        self.relu4 = ReLU()
        self.dropout4 = Dropout(p=0.4)
        self.fc2 = Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        return self.fc2(x)
# ================================================================
# 3. Training Loop
# ================================================================
def train_model(model, optim, data_loader, epochs=10):
    criterion = CrossEntropyloss()
    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in data_loader:
            x_tensor = Tensor(x_batch)
            y_tensor = Tensor(y_batch)

            logits = model(x_tensor)
            probs = logits.softmax()
            loss = criterion(probs, y_tensor)
            epoch_loss += loss._data

            loss.backward()
            optim.step()
            optim.zero_grad()

        avg_loss = epoch_loss / len(data_loader)
        epoch_losses.append(avg_loss)
        print(f"[Adam] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return epoch_losses

# ================================================================
# 4. Run Training with Adam Optimizer
# ================================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20

train_loader = DataLoader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)

print("\n--- TRAINING WITH ADAM ---")
model = CNN1D_Model()
adam = optimizer.Adam(model, lr=LEARNING_RATE)
losses = train_model(model, adam, train_loader, epochs=EPOCHS)

# ================================================================
# 5. Plot Loss Curve
# ================================================================
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), losses, marker='o')
plt.title('1D CNN on ECG5000 (Adam)', fontsize=16)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.grid(True)

# ================================================================
# 6. Evaluate on Test Set
# ================================================================
def evaluate_model(model, X_test, y_test):
    model.eval()
    x_tensor = X_test
    y_tensor = y_test

    logits = model(x_tensor)
    preds = logits.softmax()._data.argmax(axis=1)
    targets = y_test.argmax(axis=1)

    accuracy = (preds == targets).mean()
    print(f"Final Test Accuracy: {accuracy * 100:.5f}%")
    return accuracy

print("\n--- EVALUATING ON TEST SET ---")
evaluate_model(model, X_test, y_test)
plt.show()