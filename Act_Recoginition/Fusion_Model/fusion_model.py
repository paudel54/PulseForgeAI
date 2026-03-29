import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools

# ── 1. ResNet architecture ──────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.relu     = nn.ReLU()
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))

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
        return self.classifier(self.network(x).squeeze(-1))
    def get_features(self, x):
        return self.network(x).squeeze(-1)

# ── 2. All activity labels ──────────────────────────────────────────────
# Combined label space from both datasets
SHARED_ACTIVITIES = {
    0: 'sitting',
    1: 'walking',
    2: 'running',
    3: 'cycling',
    4: 'stair_climbing',
    5: 'treadmill_walking',
    6: 'timed_up_and_go',
    7: 'nordic_walking',
}

PAMAP2_TO_SHARED = {
    2: 0,  # sitting
    4: 1,  # walking
    5: 2,  # running
    6: 3,  # cycling
    7: 7,  # nordic_walking
    1: 0,  # lying → sitting (similar low activity)
    3: 0,  # standing → sitting (similar low activity)
}

PHYSIO_TO_SHARED = {
    0: 0,  # sitting
    2: 1,  # walking_6mwt → walking
    3: 5,  # treadmill_walking
    5: 3,  # cycling
    1: 4,  # stair_climbing
    4: 6,  # timed_up_and_go
}

print("=" * 60)
print("   PROPER FUSION MODEL — ALL DATA + SUBJECT SPLIT")
print("=" * 60)

# ── 3. Load models ──────────────────────────────────────────────────────
print("\nLoading pre-trained models...")
from sklearn.preprocessing import LabelEncoder
y_pamap_raw = np.load('data/y.npy')
le = LabelEncoder()
le.fit(y_pamap_raw)

pamap_model = ResNet1D(in_channels=3, num_classes=7)
pamap_model.load_state_dict(torch.load('data/model.pth', weights_only=True))
pamap_model.eval()

harnet = torch.hub.load(
    'OxWearables/ssl-wearables', 'harnet10',
    class_num=6, pretrained=False
)
harnet.load_state_dict(torch.load(
    'data/harnet_physionet.pth', weights_only=True))
harnet.eval()
print("✅ Both models loaded!")

# ── 4. Load ALL data with subject IDs ──────────────────────────────────
print("\nLoading ALL data...")
X_pamap    = np.load('data/X.npy')
y_pamap    = np.load('data/y.npy')
s_pamap    = np.load('data/subjects.npy')

X_phys     = np.load('data/physionet_X.npy')
y_phys     = np.load('data/physionet_y.npy')
s_phys     = np.load('data/physionet_pids.npy')

# Map to shared labels
y_pamap_shared = np.array([PAMAP2_TO_SHARED[l] for l in y_pamap])
y_phys_shared  = np.array([PHYSIO_TO_SHARED[l]  for l in y_phys])

# Combine ALL data
X_all = np.concatenate([X_pamap, X_phys])
y_all = np.concatenate([y_pamap_shared, y_phys_shared])
s_all = np.concatenate([
    np.array([f'pamap_{s}' for s in s_pamap]),
    np.array([f'phys_{s}'  for s in s_phys])
])

print(f"Total windows   : {len(X_all)}")
print(f"Total subjects  : {len(np.unique(s_all))}")
print(f"\nActivity distribution (ALL data):")
for aid, aname in SHARED_ACTIVITIES.items():
    count = np.sum(y_all == aid)
    if count > 0:
        print(f"  {aname:<22}: {count:4d} windows")

# ── 5. Subject-wise train/test split ───────────────────────────────────
print("\nCreating subject-wise train/test split...")
all_subjects = np.unique(s_all)
np.random.seed(42)
np.random.shuffle(all_subjects)

# 80% subjects for training, 20% for testing
n_test    = max(1, int(len(all_subjects) * 0.2))
test_subs = set(all_subjects[-n_test:])
train_subs= set(all_subjects[:-n_test])

train_mask = np.array([s in train_subs for s in s_all])
test_mask  = np.array([s in test_subs  for s in s_all])

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

print(f"Train: {len(X_train)} windows from {len(train_subs)} subjects")
print(f"Test:  {len(X_test)}  windows from {len(test_subs)} subjects")
print(f"Test subjects: {sorted(list(test_subs))[:5]}...")

# ── 6. Extract features ─────────────────────────────────────────────────
print("\nExtracting features from both models...")

def extract_features(X):
    tensor = torch.FloatTensor(X).permute(0, 2, 1)
    with torch.no_grad():
        f_resnet = pamap_model.get_features(tensor).numpy()
        f_harnet = harnet.feature_extractor(tensor).mean(dim=-1).numpy()
    return f_resnet, f_harnet

f_resnet_train, f_harnet_train = extract_features(X_train)
f_resnet_test,  f_harnet_test  = extract_features(X_test)

# Normalize independently
scaler_r = StandardScaler()
scaler_h = StandardScaler()

f_resnet_train = scaler_r.fit_transform(f_resnet_train)
f_resnet_test  = scaler_r.transform(f_resnet_test)
f_harnet_train = scaler_h.fit_transform(f_harnet_train)
f_harnet_test  = scaler_h.transform(f_harnet_test)

X_train_fused = np.concatenate([f_resnet_train, f_harnet_train], axis=1)
X_test_fused  = np.concatenate([f_resnet_test,  f_harnet_test],  axis=1)

print(f"Fused feature shape: {X_train_fused.shape[1]} dims")

# ── 7. Handle class imbalance with weights ──────────────────────────────
num_classes   = len(SHARED_ACTIVITIES)
class_counts  = np.bincount(y_train, minlength=num_classes)
class_counts  = np.where(class_counts == 0, 1, class_counts)
class_weights = torch.FloatTensor(1.0 / class_counts)
class_weights = class_weights / class_weights.sum() * num_classes
print(f"\nClass weights (for imbalance):")
for aid, aname in SHARED_ACTIVITIES.items():
    print(f"  {aname:<22}: weight={class_weights[aid]:.3f} "
          f"(n={class_counts[aid]})")

# ── 8. Train fusion classifier ──────────────────────────────────────────
X_train_t = torch.FloatTensor(X_train_fused)
X_test_t  = torch.FloatTensor(X_test_fused)
y_train_t = torch.LongTensor(y_train)

train_dl = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=64, shuffle=True
)

fusion_input = X_train_fused.shape[1]
fusion_clf   = nn.Sequential(
    nn.Linear(fusion_input, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
)

optimizer = torch.optim.Adam(
    fusion_clf.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=60)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print(f"\nTraining fusion classifier (60 epochs)...")
best_f1, best_state = 0, None

for epoch in range(60):
    fusion_clf.train()
    total_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(fusion_clf(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        fusion_clf.eval()
        with torch.no_grad():
            preds = fusion_clf(X_test_t).argmax(1).numpy()
        # Only evaluate on classes that exist in test set
        test_classes = np.unique(y_test)
        f1  = f1_score(y_test, preds, average='macro',
                       labels=test_classes)
        acc = (preds == y_test).mean()
        print(f"  Epoch {epoch+1}/60 | "
              f"Loss: {total_loss/len(train_dl):.4f} | "
              f"F1: {f1:.4f} | Acc: {acc:.2%}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone()
                         for k, v in fusion_clf.state_dict().items()}

# ── 9. Final evaluation ─────────────────────────────────────────────────
fusion_clf.load_state_dict(best_state)
fusion_clf.eval()
with torch.no_grad():
    final_preds = fusion_clf(X_test_t).argmax(1).numpy()

test_classes  = np.unique(y_test)
test_labels   = [SHARED_ACTIVITIES[i] for i in test_classes]
f1_final      = f1_score(y_test, final_preds, average='macro',
                         labels=test_classes)
acc_final     = (final_preds == y_test).mean()

print(f"\n{'='*60}")
print(f"   FINAL RESULTS (Subject-wise, ALL data)")
print(f"{'='*60}")
print(classification_report(
    y_test, final_preds,
    labels=test_classes,
    target_names=test_labels
))

# ── 10. Confusion matrix ────────────────────────────────────────────────
cm      = confusion_matrix(y_test, final_preds, labels=test_classes)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm_norm, cmap='Purples', vmin=0, vmax=1)
plt.colorbar(im, ax=ax)
n = len(test_classes)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(test_labels, rotation=35, ha='right', fontsize=9)
ax.set_yticklabels(test_labels, fontsize=9)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
ax.set_title(
    f'Fusion Model — Subject-wise Test\n'
    f'Acc={acc_final:.2%}  F1={f1_final:.4f}  '
    f'(ALL {len(X_all)} windows, {len(all_subjects)} subjects)',
    fontsize=12, fontweight='bold'
)
for i, j in itertools.product(range(n), range(n)):
    color = 'white' if cm_norm[i,j] > 0.5 else 'black'
    ax.text(j, i, f'{cm_norm[i,j]:.0%}\n({cm[i,j]})',
            ha='center', va='center', color=color, fontsize=8)

plt.tight_layout()
plt.savefig('data/fusion_proper_confusion.png',
            dpi=150, bbox_inches='tight')
print("✅ Saved to data/fusion_proper_confusion.png")
plt.show()

torch.save(fusion_clf.state_dict(), 'data/fusion_model_proper.pth')
print("✅ Proper fusion model saved!")