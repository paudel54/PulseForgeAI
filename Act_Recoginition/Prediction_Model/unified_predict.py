import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("   FULL ACTIVITY CONFIDENCE TEST — BOTH MODELS")
print("=" * 60)

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

# ── 2. Labels ───────────────────────────────────────────────────────────
PAMAP2_ACTIVITIES = {
    1: 'lying', 2: 'sitting', 3: 'standing',
    4: 'walking', 5: 'running', 6: 'cycling',
    7: 'nordic_walking'
}

PHYSIONET_ACTIVITIES = {
    0: 'sitting',        1: 'stair_climbing',
    2: 'walking_6mwt',   3: 'treadmill_walking',
    4: 'timed_up_and_go', 5: 'cycling'
}

# ── 3. Load models ──────────────────────────────────────────────────────
print("\nLoading models...")
y_pamap = np.load('data/y.npy')
le = LabelEncoder()
le.fit(y_pamap)

pamap_model = ResNet1D(in_channels=3, num_classes=7)
pamap_model.load_state_dict(torch.load('data/model.pth', weights_only=True))
pamap_model.eval()

harnet = torch.hub.load(
    'OxWearables/ssl-wearables', 'harnet10',
    class_num=6, pretrained=False
)
harnet.load_state_dict(torch.load('data/harnet_physionet.pth', weights_only=True))
harnet.eval()
print("✅ Both models loaded!\n")

# ── 4. Load all data ────────────────────────────────────────────────────
X_pamap  = np.load('data/X.npy')
y_pamap  = np.load('data/y.npy')
X_phys   = np.load('data/physionet_X.npy')
y_phys   = np.load('data/physionet_y.npy')

def get_bar(score, width=20):
    filled = int(score * width)
    empty  = width - filled
    return '█' * filled + '░' * empty

def test_pamap2_activity(activity_id, activity_name):
    idx = np.where(y_pamap == activity_id)[0]
    if len(idx) == 0:
        return
    sample = X_pamap[idx[0]]
    tensor = torch.FloatTensor(sample).unsqueeze(0).permute(0, 2, 1)

    print(f"\n{'─'*60}")
    print(f"📥 True Activity: {activity_name.upper()} (from PAMAP2)")
    print(f"{'─'*60}")

    # Model 1 prediction
    with torch.no_grad():
        out   = pamap_model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = out.argmax(dim=1).item()
    act_id   = le.inverse_transform([pred])[0]
    act_name = PAMAP2_ACTIVITIES[act_id]
    correct  = "✅" if act_id == activity_id else "❌"

    print(f"\n  🏃 ResNet/PAMAP2 → {act_name.upper()} {correct}")
    scores = [(PAMAP2_ACTIVITIES[le.inverse_transform([i])[0]], probs[i].item())
              for i in range(len(probs))]
    scores.sort(key=lambda x: x[1], reverse=True)
    for name, score in scores:
        marker = " ◄" if name == act_name else ""
        print(f"    {name:<18} {score:5.1%} {get_bar(score)}{marker}")

    # Model 2 prediction
    with torch.no_grad():
        out   = harnet(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = out.argmax(dim=1).item()
    act_name2 = PHYSIONET_ACTIVITIES[pred]
    print(f"\n  🏥 HARNet/PhysioNet → {act_name2.upper()}")
    scores2 = [(PHYSIONET_ACTIVITIES[i], probs[i].item())
               for i in range(len(probs))]
    scores2.sort(key=lambda x: x[1], reverse=True)
    for name, score in scores2[:3]:
        marker = " ◄" if name == act_name2 else ""
        print(f"    {name:<22} {score:5.1%} {get_bar(score)}{marker}")

def test_physionet_activity(activity_id, activity_name):
    idx = np.where(y_phys == activity_id)[0]
    if len(idx) == 0:
        return
    sample = X_phys[idx[0]]
    tensor = torch.FloatTensor(sample).unsqueeze(0).permute(0, 2, 1)

    print(f"\n{'─'*60}")
    print(f"📥 True Activity: {activity_name.upper()} (from PhysioNet)")
    print(f"{'─'*60}")

    # Model 1 prediction
    with torch.no_grad():
        out   = pamap_model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = out.argmax(dim=1).item()
    act_id   = le.inverse_transform([pred])[0]
    act_name = PAMAP2_ACTIVITIES[act_id]
    print(f"\n  🏃 ResNet/PAMAP2 → {act_name.upper()}")
    scores = [(PAMAP2_ACTIVITIES[le.inverse_transform([i])[0]], probs[i].item())
              for i in range(len(probs))]
    scores.sort(key=lambda x: x[1], reverse=True)
    for name, score in scores[:3]:
        marker = " ◄" if name == act_name else ""
        print(f"    {name:<18} {score:5.1%} {get_bar(score)}{marker}")

    # Model 2 prediction
    with torch.no_grad():
        out   = harnet(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = out.argmax(dim=1).item()
    act_name2 = PHYSIONET_ACTIVITIES[pred]
    correct   = "✅" if pred == activity_id else "❌"
    print(f"\n  🏥 HARNet/PhysioNet → {act_name2.upper()} {correct}")
    scores2 = [(PHYSIONET_ACTIVITIES[i], probs[i].item())
               for i in range(len(probs))]
    scores2.sort(key=lambda x: x[1], reverse=True)
    for name, score in scores2:
        marker = " ◄" if name == act_name2 else ""
        print(f"    {name:<22} {score:5.1%} {get_bar(score)}{marker}")

# ── 5. Test ALL PAMAP2 activities ───────────────────────────────────────
print("\n" + "=" * 60)
print("  PART 1: PAMAP2 ACTIVITIES (healthy adults)")
print("=" * 60)
for act_id, act_name in PAMAP2_ACTIVITIES.items():
    test_pamap2_activity(act_id, act_name)

# ── 6. Test ALL PhysioNet activities ────────────────────────────────────
print("\n\n" + "=" * 60)
print("  PART 2: PHYSIONET ACTIVITIES (elderly clinical)")
print("=" * 60)
for act_id, act_name in PHYSIONET_ACTIVITIES.items():
    test_physionet_activity(act_id, act_name)

print("\n\n" + "=" * 60)
print("  TEST COMPLETE!")
print("=" * 60)