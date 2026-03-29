<div align="center">
  <h1>Talk to Your Heart (PulseForgeAI)</h1>
  <p><strong>On-Campus Cardiac Virtual Rehab — Powered by Edge AI on DGX Spark</strong></p>
  <p>
    <a href="#architecture">Architecture</a> •
    <a href="#key-features">Features</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#clinical-ai">Clinical Agents</a>
  </p>
</div>

Cardiac rehabilitation reduces mortality by 13% and hospitalizations by 31%, but only 24% of eligible patients ever attend a session. Clinics lack the supervision bandwidth to run multi-patient sessions safely. Current telemetry solutions rely on cloud endpoints, which introduce HIPAA compliance risks, unpredictable latency, and vendor lock-in.

**PulseForgeAI** is an intelligent supervision system that lives entirely on-campus. It ingests live physiologic data from Polar H10 chest straps, processes signal patterns using local foundation models, and coordinates three distinct local LLM agents to provide real-time patient coaching and clinical documentation. Patient data never leaves the building.

## 🧠 The Architecture

By deploying to the **NVIDIA DGX Spark (128 GB Unified LPDDR5x)**, we bypass PCIe transfer bottlenecks. Real-time CPU signal processing directly hands off to GPU-bound vLLM instances in a zero-copy loop.

```text
                             NVIDIA DGX Spark
                (Zero Egress • 128GB Unified Memory • Blackwell)
+=============================================================================+
|                                                                             |
| +-----------+  BLE   +----------------+  MQTT   +------------------------+  |
| | Polar H10 |------->| Signal Engine  |-------->| Mosquitto Local Broker |  |
| | (130 Hz)  |        | (NeuroKit2)    |         | patient/{id}/vitals    |  |
| +-----------+        | SQI + Padding  |         +----------+-------------+  |
|                      +-------+--------+              sub   v                |
|                              |                             |                |
| +----------------------------v-----+       +---------------+-------------+  |
| | ChromaDB (Knowledge Base)        |       | Lead Orchestrator Router    |  |
| | • RAG Medical Literature         |       +---+-----------+-----------+-+  |
| | • Historical Vitals              |           |           |           |    |
| | • Intake Telemetry (Google Fit)  |           v           v           v    |
| +----------------------------------+       +-------+   +-------+   +-------+|
|                                            | Nurse |   | Duty  |   | Asst  ||
|                                            | Qwen3 |   | Gemma |   | Gemma ||
|                                            +-------+   +-------+   +-------+|
+=============================================================================+
```

## ⚡ Core Capabilities

1. **Continuous ECG Processing (130 Hz):**
   - Pan-Tompkins + Hamilton consensus QRS detection. 
   - ECG morphology delineation (DWT method): QRS width, QT interval, ST deviation.
   - Beat-to-beat HRV metrics (SDNN, RMSSD, pNN50, LF/HF).
   - High-fidelity **Historical Google Fit Baseline Integration** (15-min bucketing for HR, Body Temp, and segmented sleep stages).

2. **Hardware-Enforced HIPAA Compliance:**
   - No cloud orchestration. No third-party API keys. The entire stack—from MQTT to 72B foundation models—runs inside the physical walls of the clinic.

3. **Multi-Agent Clinical Roles:**
   - **Nurse Agent (Qwen3):** Patient-facing interface limited strictly to wellness phrasing and positive reinforcement.
   - **Duty Doctor Agent (MedGemma-27B):** Generates structured SOAP notes conditioned on continuous Signal Quality Index (SQI) scores.
   - **Clinical Assistant (MedGemma-27B):** Provides the clinician interrogative access to a patient's historical vitals and guidelines via local RAG retrieval.

4. **Deterministic Safety Guardrails:**
   - LLMs *do not* generate alarms. A deterministic, threshold-driven "Energy Safe Window" evaluates age, high-resolution historical intake, and real-time MET estimations before any automated response occurs.

## 🚀 Quickstart

### Prerequisites
- **Hardware**: NVIDIA DGX Spark (or equivalent >80GB VRAM system for full stack testing).
- **Software**: Python 3.10+, Docker (for vLLM), Local Mosquitto Broker.

### 1. Model Deployment (vLLM)
PulseForgeAI requires the Foundation and Clinical LLMs bound to specific local ports.

```bash
# Primary Agent (Qwen2.5)
docker run --gpus all -v /models:/models -p 8000:8000 nvcr.io/nvidia/vllm:latest \
  --model /models/Qwen2.5-72B-Instruct-AWQ --quantization awq --max-model-len 32768

# Clinical Agent (MedGemma)
docker run --gpus all -v /models:/models -p 8001:8001 nvcr.io/nvidia/vllm:latest \
  --model /models/MedGemma-27B-IT --quantization awq --port 8001
```

### 2. Environment Setup
```bash
git clone https://github.com/paudel54/PulseForgeAI.git
cd PulseForgeAI/Application/Polar_Livestream-analysis-Python
pip install -r requirements.txt
```

### 3. Launch the Pipeline
*To execute the full telemetry GUI with Google Fit Intake integration and MQTT live-streaming:*

```bash
python main.py
```

## 🛡️ Clinical Design Philosophy

Clinical charting traditionally forces providers to reconstruct patient states from memory. PulseForgeAI flips this paradigm. By capturing high-density continuous physiological data and gating it behind real-time SQI evaluations, the system transforms a live rehab session into verifiable administrative documentation automatically. 

Our benchmark target: **80-87% reduction in SOAP note authoring time**, expanding concurrent nurse supervision capacity from 3 patients to 8 patients.

---
*Disclaimer: Research and development prototype only. Not currently FDA-cleared as a Software as a Medical Device (SaMD).*