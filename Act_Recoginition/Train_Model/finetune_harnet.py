import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

ACTIVITY_NAMES = {
    0: 'sitting',
    1: 'stair_climbing',
    2: 'walking_6mwt',
    3: 'treadmill_walking',
    4: 'timed_up_and_go',
    5: 'cycling'
}

# ── 1. Load PhysioNet data ──────────────────────────────────────────────
print("Loading PhysioNet data...")
X = np.load('data/physionet_X.npy')
y = np.load('data/physionet_y.npy')
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ── 2. Load pre-trained HARNet10 ────────────────────────────────────────
print("\nLoading pre-trained HARNet10...")
num_classes = len(ACTIVITY_NAMES)
model = torch.hub.load(
    'OxWearables/ssl-wearables',
    'harnet10',
    class_num=num_classes,
    pretrained=True
)
print("✅ HARNet10 loaded!")

# ── 3. Freeze feature extractor, only train classifier ─────────────────
for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")

# ── 4. Prepare data ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# HARNet expects (batch, 3, 300)
X_train_t = torch.FloatTensor(X_train).permute(0, 2, 1)
X_test_t  = torch.FloatTensor(X_test).permute(0, 2, 1)
y_train_t = torch.LongTensor(y_train)
y_test_t  = torch.LongTensor(y_test)

train_dl = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_dl  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=64)

# ── 5. Train classifier only (Phase 1) ─────────────────────────────────
print("\n── Phase 1: Training classifier only (10 epochs) ──")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 2 == 0:
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in test_dl:
                preds.extend(model(xb).argmax(1).numpy())
        f1 = f1_score(y_test, preds, average='macro')
        print(f"  Epoch {epoch+1}/10 | Loss: {total_loss/len(train_dl):.4f} | F1: {f1:.4f}")

# ── 6. Unfreeze all layers (Phase 2) ────────────────────────────────────
print("\n── Phase 2: Fine-tuning all layers (20 epochs) ──")
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in test_dl:
                preds.extend(model(xb).argmax(1).numpy())
        f1 = f1_score(y_test, preds, average='macro')
        print(f"  Epoch {epoch+1}/20 | Loss: {total_loss/len(train_dl):.4f} | F1: {f1:.4f}")

# ── 7. Final evaluation ─────────────────────────────────────────────────
model.eval()
all_preds = []
with torch.no_grad():
    for xb, yb in test_dl:
        all_preds.extend(model(xb).argmax(1).numpy())

print("\n── Final Results ──")
target_names = [ACTIVITY_NAMES[i] for i in range(num_classes)]
print(classification_report(y_test, all_preds, target_names=target_names))

torch.save(model.state_dict(), 'data/harnet_physionet.pth')
print("✅ Model saved to data/harnet_physionet.pth")
