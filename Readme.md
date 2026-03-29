# Talk to Your Heart

**An AI-powered cardiac rehabilitation assistant transforming wearable data into actionable clinical insights and accessible patient education.**

## Overview

"Talk to Your Heart" is an end-to-end AI solution designed for patients undergoing clinical cardiac rehabilitation post-surgery. By synthesizing continuous physiological data from edge wearables (ECG & PPG) with advanced foundation models, this platform bridges the communication gap between complex physiological metrics, clinical care teams, and patients.

The system leverages a multi-agent LLM architecture augmented by clinical knowledge bases to generate automated clinical reports (like SOAP notes and exercise stress test summaries) while providing tailored conversational interfaces for both doctors and patients.

## Repository Structure

```
PulseForgeAI/
├── Application/
│   └── Polar_Livestream-analysis-Python/   # Real-time ECG dashboard + MQTT streaming
├── Act_Recoginition/                       # HAR fusion model (PAMAP2 + PhysioNet)
├── ECG_Embedding/                          # ECG foundation model inference (CLEF, ECG-FM)
├── ECG_Signal_Pipeline/                    # Offline signal processing notebooks
├── MedLLM/                                 # Medical LLM agent configuration
├── backend/                                # FastAPI backend services
├── Documents/                              # Brainstorm notes, text corpus references
├── Figures/                                # Generated visualizations
├── Src/                                    # Shared source utilities
├── master-plan.md                          # Full technical architecture document
└── Overall_Architecture.md                 # High-level system diagram
```

## Key Features

- **Dual-Interface Conversational Agents:**
  - *For Clinicians:* A highly technical assistant capable of deep-diving into physiological metrics, analyzing trends, and referencing treatment guides.
  - *For Patients:* An empathetic, easy-to-understand educational assistant that helps patients comprehend their health status, rehab progress, and treatment options.
- **Automated Clinical Reporting:** Automatically generates formal clinical summary reports, including SOAP notes and treadmill stress test evaluations, reducing clinician administrative burden.
- **Real-Time Wearable Integration:** Ingests live 130 Hz ECG and 100 Hz accelerometer data from Polar H10 chest straps via BLE. Streams unified 5-second analysis windows over MQTT (`broker.emqx.io`) for downstream AI consumption.
- **Clinical RAG Pipeline:** Grounds LLM responses in established medical science, checking patient data against clinical literature, ACC/AHA protocols, and cardiac guidelines.
- **Multi-Dataset Activity Recognition:** Fusion model combining PAMAP2 (healthy adults) and PhysioNet (elderly/clinical) datasets for 8-class activity classification.

## System Architecture

Our solution is built on a scalable, 5-stage pipeline:

### 1. Edge Data Acquisition

Patient data is captured continuously using Polar H10 chest straps and transmitted via Bluetooth Low Energy (BLE) to a local edge device running the Polar ECG Dashboard.

### 2. Signal Processing & Quality Assessment

Real-time 4th-order Butterworth bandpass filtering (0.5–40 Hz), 3-metric SQI computation, R-peak detection, HRV analysis (RMSSD, SDNN, LF/HF), and DWT-based ECG morphological delineation (P, QRS, ST, QT/QTc widths).

### 3. MQTT Telemetry Streaming

Unified 5-second JSON payloads containing raw ECG arrays, computed metrics, and accelerometer features are published to `broker.emqx.io` for real-time consumption by downstream services.

### 4. Cognitive Processing and RAG Integration

The core reasoning engine relies on the **Qwen2.5-72B-AWQ** model operating in a multi-agent setup on **NVIDIA DGX Spark**:

- **Retrieval-Augmented Generation (RAG):** Context is enriched using a **ChromaDB Vector Store** loaded with AHA/AACVPR protocols and cardiac guidelines.
- The LLM synthesizes the physiological feature tokens, patient context payload, and retrieved literature.

### 5. Output Dissemination and Feedback Loops

The processed insights are routed to three primary outputs:

- **Rule-based Alerts:** Immediate deterministic flags for critical physiological anomalies (never LLM-generated).
- **Clinician Interface:** Detailed dashboards with auto-generated SOAP notes.
- **Patient Interface:** A simplified conversational agent for education and encouragement.

## Tech Stack

- **Edge Dashboard:** PyQt5, pyqtgraph, bleak, bleakheart
- **Signal Processing:** NumPy, SciPy, NeuroKit2, vital_sqi, pyHRV
- **Streaming:** paho-mqtt (broker.emqx.io)
- **Foundation Models:** CLEF encoder, ECG-FM, NormWear
- **LLM & Reasoning:** Qwen2.5-72B-AWQ, MedGemma-27B, multi-agent framework
- **Vector Database (RAG):** ChromaDB
- **Activity Recognition:** PyTorch (ResNet1D + HARNet10 fusion)
- **Compute:** NVIDIA DGX Spark (128 GB unified memory, Blackwell GPU)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/paudel54/PulseForgeAI.git
cd PulseForgeAI

# Install dashboard dependencies
cd Application/Polar_Livestream-analysis-Python
pip install -r requirements.txt

# Run the real-time dashboard (requires Polar H10)
python main.py

# Or run in simulation mode (no hardware)
python main.py --mock
```

See individual module READMEs for detailed setup instructions:

- [Application/Polar_Livestream-analysis-Python/README.md](Application/Polar_Livestream-analysis-Python/README.md) — Real-time ECG dashboard & MQTT streaming
- [Act_Recoginition/README.md](Act_Recoginition/README.md) — Activity recognition fusion model