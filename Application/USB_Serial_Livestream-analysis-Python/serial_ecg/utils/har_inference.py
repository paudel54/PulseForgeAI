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
        
        # State memory for Markov transitions & EMA temporal smoothing
        self._last_state = "sitting"
        self._ema_probs = None

    @torch.no_grad()
    def predict(self, acc_100hz_window: np.ndarray) -> dict:
        """
        Expects a numpy array of shape (1000, 3) representing exactly 10 seconds 
        of accelerometer data at 100 Hz.
        Uses a Hybrid Heuristic + ML approach restricting to 4 classes:
        Sitting, Standing, Walking, Stair Climbing.
        """
        if acc_100hz_window.shape[0] < 500:
            return {"label": "unknown", "confidence": {}}
            
        # 1. Component Analysis (Heuristics)
        x   = acc_100hz_window[:, 0]
        y   = acc_100hz_window[:, 1]
        z   = acc_100hz_window[:, 2]
        mag = np.sqrt(x**2 + y**2 + z**2)
        var_mag = np.var(mag, ddof=1)
        
        if var_mag < 2500:
            self._ema_probs = None 

            y_mean = np.mean(y)
            z_mean = np.mean(z)
            pitch_angle = np.arctan2(abs(z_mean), abs(y_mean)) * (180.0 / np.pi)
            
            if pitch_angle > 15.0:
                self._last_state = "sitting"
                return {"label": "sitting", "confidence": {"sitting": 1.0}}
            else:
                self._last_state = "standing"
                return {"label": "standing", "confidence": {"standing": 1.0}}

        # 2. Dynamic Activity (Walking / Stair Climbing) via PyTorch ML
        if self._last_state == "sitting":
            self._last_state = "timed_up_and_go"
            return {"label": "timed_up_and_go", "confidence": {"timed_up_and_go": 1.0}}

        # Downsample to 30 Hz
        target_len = int(acc_100hz_window.shape[0] * (30.0 / 100.0))
        acc_30hz = resample(acc_100hz_window, target_len, axis=0)
        
        tensor = torch.FloatTensor(acc_30hz).unsqueeze(0).permute(0, 2, 1).to(self.device)
        
        f_resnet = self.pamap_model.get_features(tensor)
        f_harnet = self.harnet.feature_extractor(tensor).mean(dim=-1)
        
        f_resnet_np = f_resnet.cpu().numpy()
        f_harnet_np = f_harnet.cpu().numpy()
        
        f_r_norm = (f_resnet_np - f_resnet_np.mean()) / (f_resnet_np.std() + 1e-8)
        f_h_norm = (f_harnet_np - f_harnet_np.mean()) / (f_harnet_np.std() + 1e-8)
        
        fused_features = np.concatenate([f_r_norm, f_h_norm], axis=1)
        fused_tensor = torch.FloatTensor(fused_features).to(self.device)
        
        logits = self.fusion_clf(fused_tensor).squeeze(0)
        
        target_indices = [1, 4]
        mask = torch.full_like(logits, float('-inf'))
        for idx in target_indices:
            mask[idx] = logits[idx]
        
        current_probs = torch.softmax(mask, dim=0).cpu().numpy()
        
        if self._ema_probs is None:
            self._ema_probs = current_probs
        else:
            alpha = 0.5
            self._ema_probs = (alpha * current_probs) + ((1.0 - alpha) * self._ema_probs)
        
        pred_idx = int(np.argmax(self._ema_probs))
        pred_label = SHARED_ACTIVITIES.get(pred_idx, "unknown")
        
        self._last_state = pred_label
        
        confidences = {
            "walking": float(self._ema_probs[1]),
            "stair_climbing": float(self._ema_probs[4])
        }
        
        return {
            "label": pred_label,
            "confidence": confidences
        }
