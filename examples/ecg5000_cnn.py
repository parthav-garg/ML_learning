import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from base.tensor import Tensor
from nn import Module, Conv1D, ReLU, MaxPool1D, Linear, Dropout, CrossEntropyloss, optimizer
from DataLoader import DataLoader
import matplotlib.pyplot as plt
# ================================================================
# 1. Load ECG5000 Dataset (1D time series) from OpenML
# ================================================================
print("Loading ECG5000 dataset from OpenML...")

ecg = fetch_openml('ECG5000', version=1, as_frame=False, parser='auto')

X = ecg['data'].astype(np.float64)          # shape: (5000, 140)
y = ecg['target']                           # string labels: '1' to '5'

# Encode string labels to integer class IDs (0–4)
le = LabelEncoder()
y = le.fit_transform(y)                     # e.g., '1' → 0, ..., '5' → 4
num_classes = len(np.unique(y))

# Normalize X
X -= X.mean()
X /= X.std()

# Reshape to (B, 1, T) for Conv1D (B=batch, C=1 channel, T=time)
X = X.reshape((X.shape[0], 1, X.shape[1]))  # shape: (5000, 1, 140)

# One-hot encode targets
y_one_hot = np.eye(num_classes)[y]         # shape: (5000, 5)

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
        self.maxpool1 = MaxPool1D(kernel_size=2, stride=2)
        self.conv2 = Conv1D(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu2 = ReLU()
        self.dropout2 = Dropout(p=0.1)
        self.flatten_dim = 32 * 140  # length remains 140
        self.fc1 = Linear(self.flatten_dim, 256)
        self.relu4 = ReLU()
        self.dropout4 = Dropout(p=0.4)
        self.fc2 = Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        #print("conv1 shape= ", x.shape)
        x = self.relu1(x)  
        x = self.dropout1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = x.reshape(x.shape[0], -1)
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
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    for x_batch, y_batch in dataloader:

        logits = model(x_batch)
        preds = logits.softmax()._data.argmax(axis=1)
        targets = y_batch.argmax(axis=1)

        correct += np.sum(preds == targets, dtype=np.float64)
        total += len(x_batch)

    accuracy = (correct / total) * 100
    print(f"Final Test Accuracy: {accuracy:.5f}%")
    return accuracy

print("\n--- EVALUATING ON TEST SET ---")
data = DataLoader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
evaluate_model(model, data)
plt.show(block = False)