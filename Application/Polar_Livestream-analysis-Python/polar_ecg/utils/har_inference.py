import os
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample

# Same ResBlock structure used to train the PAMAP2 model
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

# Combined label space defined during fusion
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

class HARInferenceEngine:
    """
    Manages loading the 3-part PyTorch architecture (ResNet + HARNet + Fusion Classifier)
    for high-frequency realtime execution.
    """
    def __init__(self, act_recognition_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. PAMAP2 ResNet
        self.pamap_model = ResNet1D(in_channels=3, num_classes=7)
        pamap_path = os.path.join(act_recognition_dir, 'model.pth')
        self.pamap_model.load_state_dict(torch.load(pamap_path, map_location='cpu'))
        self.pamap_model.to(self.device).eval()
        
        # 2. PhysioNet HARNet10 via Torch Hub
        self.harnet = torch.hub.load(
            'OxWearables/ssl-wearables', 'harnet10', 
            class_num=6, pretrained=False
        )
        harnet_path = os.path.join(act_recognition_dir, 'harnet_physionet.pth')
        self.harnet.load_state_dict(torch.load(harnet_path, map_location='cpu'))
        self.harnet.to(self.device).eval()
        
        # 3. Fusion Classifier
        # ResNet outputs 128-dim, HARNet outputs 1024-dim -> Total 1152
        num_classes = len(SHARED_ACTIVITIES) # 8
        self.fusion_clf = nn.Sequential(
            nn.Linear(1152, 512),
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
        fusion_path = os.path.join(act_recognition_dir, 'fusion_model_proper.pth')
        self.fusion_clf.load_state_dict(torch.load(fusion_path, map_location='cpu'))
        self.fusion_clf.to(self.device).eval()

    @torch.no_grad()
    def predict(self, acc_100hz_window: np.ndarray) -> dict:
        """
        Expects a numpy array of shape (1000, 3) representing exactly 10 seconds 
        of native Polar H10 accelerometer data at 100 Hz.
        """
        if acc_100hz_window.shape[0] < 500: # Need at least 5s to make a partial guess
            return {"label": "unknown", "confidence": {}}
            
        # 1. Downsample to 30 Hz (requires exactly 300 samples for 10s)
        # We calculate exact dimension matching length of provided signal chunk
        target_len = int(acc_100hz_window.shape[0] * (30.0 / 100.0))
        acc_30hz = resample(acc_100hz_window, target_len, axis=0) # shape: (T_30, 3)
        
        # 2. Format tensor (Batch=1, Channels=3, Length)
        tensor = torch.FloatTensor(acc_30hz).unsqueeze(0).permute(0, 2, 1).to(self.device)
        
        # 3. Extract Features
        f_resnet = self.pamap_model.get_features(tensor) # (1, 128)
        f_harnet = self.harnet.feature_extractor(tensor).mean(dim=-1) # (1, 1024)
        
        # 4. Instance Normalization (Fallback for missing StandardScaler fitting)
        # Fuses 1152 dim vector and zeroes mean to help linear layer stability
        f_resnet_np = f_resnet.cpu().numpy()
        f_harnet_np = f_harnet.cpu().numpy()
        
        f_r_norm = (f_resnet_np - f_resnet_np.mean()) / (f_resnet_np.std() + 1e-8)
        f_h_norm = (f_harnet_np - f_harnet_np.mean()) / (f_harnet_np.std() + 1e-8)
        
        fused_features = np.concatenate([f_r_norm, f_h_norm], axis=1) # (1, 1152)
        fused_tensor = torch.FloatTensor(fused_features).to(self.device)
        
        # 5. Classify
        logits = self.fusion_clf(fused_tensor) # (1, 8)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        pred_idx = int(np.argmax(probs))
        pred_label = SHARED_ACTIVITIES.get(pred_idx, "unknown")
        
        # Convert to dictionary mapping
        confidences = {SHARED_ACTIVITIES[i]: float(probs[i]) for i in range(len(probs))}
        
        return {
            "label": pred_label,
            "confidence": confidences
        }
