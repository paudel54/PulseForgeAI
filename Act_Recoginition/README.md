# Activity Recognition Module

Human Activity Recognition (HAR) system using multi-dataset sensor fusion. Combines accelerometer data from two populations — healthy adults (PAMAP2) and elderly/clinical patients (PhysioNet) — to train a unified 8-class activity classifier.

## Architecture Overview

```
PAMAP2 Dataset                    PhysioNet Dataset
(healthy adults, lab)             (elderly, clinical)
        │                                 │
  preprocess.py               extract_physionet.py
        │                                 │
   data/X.npy                  data/physionet_X.npy
        │                                 │
  train_model.py              finetune_harnet.py
  (ResNet1D)                  (HARNet10, SSL pretrained)
        │                                 │
  data/model.pth            data/harnet_physionet.pth
        └─────────┬───────────────────────┘
                  │
           fusion_model.py
        (1152-dim feature fusion)
                  │
        data/fusion_model_proper.pth
                  │
         ┌────────┴────────┐
   unified_predict.py    report.py
   (inference)           (visualizations)
```

## Directory Structure

```
Act_Recoginition/
├── Data_Preparation/
│   ├── preprocess.py           # PAMAP2 preprocessing (100Hz → 30Hz, 10s windows)
│   └── extract_physionet.py    # PhysioNet WFDB extraction (200Hz → 30Hz, 10s windows)
├── Train_Model/
│   ├── train_model.py          # Train ResNet1D on PAMAP2 (~88.9% accuracy)
│   └── finetune_harnet.py      # Two-phase fine-tune HARNet10 on PhysioNet (~73.3% accuracy)
├── Fusion_Model/
│   └── fusion_model.py         # Feature fusion + 8-class unified classifier
├── Prediction_Model/
│   └── unified_predict.py      # Inference demo across both datasets
└── Report/
    └── report.py               # Generate 5 publication-quality figures
```

## Activity Classes

The fusion model recognizes 8 unified activities mapped from both source datasets:

| ID | Activity | Source |
|----|----------|--------|
| 0 | sitting | PAMAP2, PhysioNet |
| 1 | walking | PAMAP2 |
| 2 | running | PAMAP2 |
| 3 | cycling | PAMAP2, PhysioNet |
| 4 | stair_climbing | PhysioNet |
| 5 | treadmill_walking | PhysioNet |
| 6 | timed_up_and_go | PhysioNet |
| 7 | nordic_walking | PAMAP2 |

## Setup

### Prerequisites
```
torch
torchvision
numpy
scikit-learn
matplotlib
wfdb
```

### Data Sources
- **PAMAP2**: [UCI PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) — place under `data/PAMAP2_Dataset/Protocol/`
- **PhysioNet**: [Wearable Exercise Frailty](https://physionet.org/content/wearable-exercise-frailty/) — place under `~/Downloads/wearable-exercise-frailty/acc/`

## Usage

Run steps in order:

```bash
# 1. Preprocess datasets
python Data_Preparation/preprocess.py
python Data_Preparation/extract_physionet.py

# 2. Train individual models
python Train_Model/train_model.py
python Train_Model/finetune_harnet.py

# 3. Train fusion classifier
python Fusion_Model/fusion_model.py

# 4. Run inference demo
python Prediction_Model/unified_predict.py

# 5. Generate report figures
python Report/report.py
```

All outputs (`.npy` data, `.pth` model weights, `.png` figures) are written to `data/`.

## Model Details

### ResNet1D (PAMAP2)
- Architecture: 3× ResBlock (Conv1D, kernel=7, BatchNorm, ReLU) → AdaptiveAvgPool → Linear
- Channels: 3 → 64 → 128 → 128
- Input: `(batch, 300, 3)` — 10s at 30Hz, 3-axis accelerometer
- Output feature dim: 128

### HARNet10 (PhysioNet)
- Base: `OxWearables/ssl-wearables` pretrained on UK Biobank (100k+ participants)
- Fine-tuned two-phase: frozen backbone (10 epochs) → full fine-tune (20 epochs)
- Input: `(batch, 3, 300)`
- Output feature dim: 1024

### Fusion Classifier
- Input: 1152-dim (128 + 1024 concatenated, normalized independently)
- Architecture: 512 → 256 → 128 → 8 (BatchNorm, ReLU, Dropout 0.4/0.3)
- Training: 60 epochs, Adam, CosineAnnealingLR, class-weighted loss
- Split: subject-wise 80/20 (prevents sample-level leakage across 47 subjects)
