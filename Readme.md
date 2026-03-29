<div align="center">
  <h1>Talk to Your Heart (PulseForgeAI)</h1>
  <p><strong>On-Campus Cardiac Rehab Intelligence — Edge AI on NVIDIA DGX Spark</strong></p>
  <p>
    <a href="#architecture">Architecture</a> •
    <a href="#key-features">Features</a> •
    <a href="#live-demo">Live Demo</a> •
    <a href="#getting-started">Getting Started</a>
  </p>
</div>

Cardiac rehabilitation is a Class 1A recommended therapy that reduces all-cause mortality by 13% and hospitalizations by 31%. Yet **only 24% of eligible Medicare beneficiaries** ever attend a session, and fewer than 27% of those who start complete the program. The CMS Million Hearts initiative estimates closing this gap would save **25,000 lives** and prevent **180,000 hospitalizations** every year.

**Talk to Your Heart** solves the cardiac rehab supervision bottleneck with an AI copilot that runs **entirely on-campus**. The system ingests live Polar H10 ECG/ACC data via BLE, extracts interpretable clinical features through a dual-window signal processing pipeline, and coordinates two specialized AI agents — a patient-facing Nurse and a clinician-facing Clinical Assistant — grounded in deterministic safety checks, reference cohort matching, and RAG-retrieved AHA/AACVPR guidelines.

**Proof of concept is live:** [pulse-forge-ai.vercel.app](https://pulse-forge-ai.vercel.app)
**Production target:** NVIDIA DGX Spark with zero PHI egress.

---

## 🧠 Architecture

```
+=============================================================================+
|                     NVIDIA DGX Spark (Production Target)                    |
|              (Current proof-of-concept: Vercel + Ollama tunneling)          |
|                                                                             |
| +-----------+  BLE   +------------------+  MQTT   +----------------------+  |
| | Polar H10 |------->| Dual-Window      |-------->| MQTT Broker          |  |
| | ECG 130Hz |        | Signal Engine    |         | (EMQX / Mosquitto)   |  |
| | ACC 100Hz |        | 5s: SQI,HR,HAR   |         +----------+-----------+  |
| +-----------+        | 30s: HRV,Morph   |              sub   v              |
|                      +------------------+                    |              |
| +-----------+                                                |              |
| | Google Fit|---> Intake JSON       +------------------------+-----------+  |
| | 7-day     |        |              | Lead Orchestrator                  |  |
| | Baseline  |        v              | (Role-Based Router + Safety)       |  |
| +-----------+ +---------------+     +----------+--------------------+    |  |
|               | ChromaDB      |                |          |              |  |
|               | • RAG Lit.    |                v          v              |  |
|               | • Cohorts     |           +--------+ +-----------+       |  |
|               | • Live MQTT   |           | Nurse  | | Clinical  |       |  |
|               +---------------+           | Agent  | | Assistant |       |  |
|                                           +----+---+ +-----+-----+       |  |
|                                                |           |             |  |
|                                           Patient Chat  Doctor Chat      |  |
|                                           + Education   + SOAP Notes     |  |
+=============================================================================+
```

## ⚡ Key Features

**1. Dual-Window Signal Processing Pipeline**
- 5-second window: SQI (template + SNR + motion), instantaneous HR, 4 HAR features
- 30-second window: RMSSD, SDNN, LF/HF (Lomb-Scargle), DWT morphology (QRS, QT/QTc, ST)
- Pan-Tompkins + Hamilton consensus QRS detection via NeuroKit2
- All metrics map directly to published cardiac rehab literature — fully interpretable

**2. Two-Agent Clinical Intelligence**
- **Nurse Agent** (Qwen3): Patient-facing wellness companion with Spanish-language support. Strict wellness-framing guardrails — never diagnoses, never recommends medication changes.
- **Clinical Assistant** (MedGemma-27B): Clinician-facing interactive Q&A and SOAP note generation. RAG-grounded in AHA/AACVPR guidelines with SQI-conditioned confidence reporting.

**3. 4-Layer Deterministic Safety Architecture**
- Layer 1: `EnergySafeWindow` — deterministic HR/SQI threshold alerts, pre-LLM
- Layer 2: Emergency keyword classifier — hardcoded response, no LLM
- Layer 3: Output validator — blocks diagnostic language in Nurse responses
- Layer 4: System prompt guardrails — role boundaries enforced via instruction

**4. Live MQTT → ChromaDB → LLM Context Pipeline**
- Polar H10 vitals published to MQTT topics per patient
- `mqtt_subscriber.py` upserts live telemetry into ChromaDB `live_patients` collection
- Agent orchestrator retrieves real-time context for every query

**5. Reference Cohort Matching**
- PhysioNet Wearable Exercise Frailty Dataset (47 post-cardiac-surgery patients)
- ECG-FM 768-dim embeddings + clinical metadata in ChromaDB
- Interpretable patient comparisons (surgery type, age, frailty, gait velocity)

**6. Google Fit Longitudinal Baseline**
- 7-day historical HR (15-min buckets), steps, sleep stages, body temperature
- Between-session context for the 4–5 days/week patients aren't in clinic

**7. Activity Recognition (HAR Fusion Model)**
- ResNet1D (PAMAP2, 88.9%) + HARNet10 SSL (PhysioNet, 73.3%)
- 1152-dim feature fusion, 8 unified activity classes
- Subject-wise train/test splits prevent data leakage

## 🌐 Live Demo

The proof of concept is deployed at **[pulse-forge-ai.vercel.app](https://pulse-forge-ai.vercel.app)** — demonstrating the full pipeline:

- **PDF Upload** → AHA/AACVPR guidelines chunked into ChromaDB RAG
- **Role-Based Chat** → toggle Clinician / Patient for different agent personas
- **Live Telemetry** → real-time Polar H10 HR, HRV, activity via MQTT WebSocket
- **SOAP Notes** → automated structured clinical documentation
- **Deterministic Safety** → EnergySafeWindow evaluates before every LLM call

The Vercel deployment uses Ollama tunneling for LLM inference. The **production deployment** targets NVIDIA DGX Spark for hardware-enforced HIPAA compliance — same codebase, only infrastructure changes.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Polar H10 chest strap (or use `--mock` mode)
- Ollama with MedGemma-27B or compatible model

### 1. Signal Processing Application (PyQt5 Dashboard)
```bash
cd Application/Polar_Livestream-analysis-Python
pip install -r requirements.txt
python main.py          # Real Polar H10 via BLE
python main.py --mock   # Mock sensor for testing
```

### 2. Backend API Server
```bash
cd backend
pip install -r requirements.txt
python ingest_literature.py   # Ingest AHA/AACVPR PDFs into ChromaDB
python ingest_cohorts.py      # Ingest PhysioNet reference cohort
python mqtt_subscriber.py &   # Start MQTT → ChromaDB sync (background)
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. MQTT Live Telemetry Bridge
```bash
# From the signal processing app, vitals publish to EMQX broker
# The mqtt_subscriber.py process syncs these into ChromaDB live_patients
# The web dashboard polls /api/live/metrics for real-time display
```

## 🔑 Key Files for Evaluator Validation

To verify the components mentioned in our documentation, please review the following key backend implementation files:
- **`backend/main.py`** — Contains the FastAPI endpoints, including the `/api/session/{patient_id}/soap` generation endpoint and the ChromaDB KNN logic (lines 135-155).
- **`backend/agent_orchestrator.py`** — Contains the `PulseForgeOrchestrator` role-based prompt assembly logic with emergency/safety bounds propagation (Layer 2 safety).
- **`backend/safety_engine.py`** — Contains `EnergySafeWindow` implementing deterministic threshold checks (Layer 1 safety) before any LLM invocations.
- **`backend/ingest_cohorts.py`** — Contains the script for ingesting physiological datasets into ChromaDB for cohort matching.

## 📁 Repository Structure

```
PulseForgeAI/
├── backend/                           ← FastAPI + Vercel web application
│   ├── main.py                        ← FastAPI server (271 lines)
│   ├── agent_orchestrator.py          ← Role-based prompt assembly
│   ├── safety_engine.py               ← EnergySafeWindow deterministic safety
│   ├── mqtt_pipeline.py               ← MQTT pub/sub with QoS differentiation
│   ├── mqtt_subscriber.py             ← EMQX → ChromaDB live state sync
│   ├── ingest_literature.py           ← PubMedBERT RAG ingestion pipeline
│   ├── ingest_cohorts.py              ← PhysioNet cohort → ChromaDB
│   ├── utils.py                       ← Ollama inference with tunnel bypass
│   ├── vercel.json                    ← Vercel serverless deployment
│   └── static/                        ← Web dashboard (HTML/CSS/JS)
│       ├── index.html                 ← Clinical reasoning dashboard
│       ├── app.js                     ← Chat, upload, MQTT WebSocket, Chart.js
│       └── style.css                  ← Glassmorphism design system
│
├── Application/
│   └── Polar_Livestream-analysis-Python/  ← PyQt5 BLE signal processing
│       ├── main.py                    ← Entry point (real or --mock)
│       ├── intake_state.json          ← Clinical + Google Fit schema
│       └── polar_ecg/
│           ├── workers/
│           │   ├── processing_worker.py  ← Dual-window (5s/30s) pipeline
│           │   ├── ble_worker.py      ← Polar H10 BLE connection
│           │   └── mqtt_worker.py     ← QThread MQTT publisher
│           ├── ui/
│           │   ├── dashboard.py       ← Real-time ECG/HRV visualization
│           │   └── intake_form.py     ← Clinical intake form
│           └── utils/
│               ├── google_fit_fetcher.py  ← Google Fit 7-day baseline
│               ├── har_inference.py   ← HAR fusion model inference
│               ├── mock_sensor.py     ← Synthetic ECG/ACC generator
│               └── ring_buffer.py     ← Efficient circular buffer
│
├── Act_Recoginition/                  ← HAR dual-model training pipeline
│   ├── Data_Preparation/              ← PAMAP2 + PhysioNet preprocessing
│   ├── Train_Model/                   ← ResNet1D + HARNet10 SSL fine-tuning
│   ├── Fusion_Model/                  ← 1152-dim feature fusion classifier
│   └── *.pth                          ← Trained model weights
│
├── ECG_Embedding/
│   └── ECG_frailty-db_feature_lookup.py  ← ECG-FM 768-dim + signal metrics
│
├── system_prompts.md                  ← Production agent system prompts (505 lines)
├── master-plan.md                     ← Master plan + execution strategy
├── master-plan_fin.md                 ← Extended master plan with technical detail
├── tests/                             ← Safety engine unit tests
└── archive/                           ← Previous plan versions
```

## 🛡️ Safety & Compliance

- **HIPAA by architecture** — DGX Spark production deployment: zero PHI egress, no BAAs
- **Deterministic safety** — EnergySafeWindow fires before LLM, catches HR threshold violations in <1s
- **4-layer guardrails** — Critical alerts never depend on LLM availability
- **General Wellness framing** — FDA January 2026 guidance compliant; SOAP notes as administrative documentation
- **Emergency keyword bypass** — "chest pain", "can't breathe" trigger hardcoded response without LLM

---

*Talk to Your Heart — On-Campus Cardiac Rehabilitation Intelligence*
*PoC: [pulse-forge-ai.vercel.app](https://pulse-forge-ai.vercel.app) | Production: NVIDIA DGX Spark*
*Disclaimer: PulseForgeAI is a research prototype. Not FDA-approved as SaMD.*