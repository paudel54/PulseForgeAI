import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# ── 1. Load preprocessed data ──────────────────────────────────────────
print("Loading preprocessed data...")
X = np.load('data/X.npy')
y = np.load('data/y.npy')

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ── 2. Encode labels to 0,1,2... ───────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Activities: {le.classes_}")

# ── 3. Train/test split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 4. Convert to PyTorch tensors ───────────────────────────────────────
# ResNet expects (batch, channels, time) so we transpose
X_train_t = torch.FloatTensor(X_train).permute(0, 2, 1)
X_test_t  = torch.FloatTensor(X_test).permute(0, 2, 1)
y_train_t = torch.LongTensor(y_train)
y_test_t  = torch.LongTensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=32)

# ── 5. Build ResNet block ───────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)

# ── 6. Build full ResNet model ──────────────────────────────────────────
class ResNet1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            ResBlock(in_channels, 64),
            ResBlock(64, 128),
            ResBlock(128, 128),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.network(x)
        x = x.squeeze(-1)
        return self.classifier(x)

# ── 7. Setup training ───────────────────────────────────────────────────
num_classes = len(le.classes_)
model    = ResNet1D(in_channels=3, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print(f"\nModel ready! Classes: {num_classes}")
print(f"Training on {len(train_ds)} samples, testing on {len(test_ds)} samples")

# ── 8. Training loop ────────────────────────────────────────────────────
EPOCHS = 30
print(f"\nTraining for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        model.eval()
        all_preds = []
        with torch.no_grad():
            for xb, yb in test_dl:
                preds = model(xb).argmax(dim=1)
                all_preds.extend(preds.numpy())
        f1 = f1_score(y_test, all_preds, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_dl):.4f} | F1: {f1:.4f}")

# ── 9. Final evaluation ─────────────────────────────────────────────────
model.eval()
all_preds = []
with torch.no_grad():
    for xb, yb in test_dl:
        preds = model(xb).argmax(dim=1)
        all_preds.extend(preds.numpy())

print("\n── Final Results ──")
print(classification_report(y_test, all_preds, target_names=[str(c) for c in le.classes_]))

torch.save(model.state_dict(), 'data/model.pth')
print("Model saved to data/model.pth")
