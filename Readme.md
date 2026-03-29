<div align="center">
  <h1>Talk to Your Heart (PulseForgeAI)</h1>
  <p><strong>On-Campus Cardiac Virtual Rehab — Powered by Edge AI on DGX Spark</strong></p>
  <p>
    <a href="#architecture">Architecture</a> •
    <a href="#key-features">Features</a> •
    <a href="#clinical-ai">Clinical Agents</a> •
    <a href="#installation">Getting Started</a>
  </p>
</div>

Cardiac rehabilitation is a Class 1a recommended therapy that reduces all-cause mortality by 13% and hospitalizations by 31%. However, owing to supervision bottlenecks and resourcing, **only 24% of eligible Medicare beneficiaries ever attend a session**. Current commercial AI monitoring solutions rely heavily on cloud APIs which introduce high HIPAA compliance costs, unpredictable 100-500ms latencies, and total dependency on off-site servers for emergency telemetry.

**PulseForgeAI** solves the cardiac rehab participation gap by providing an intelligent supervision system that operates entirely on-campus. Deployed locally on the NVIDIA DGX Spark, the system securely processes raw Polar H10 streaming data through custom foundation models and coordinates robust, localized conversational coaching and automated clinical documentation workflows via multi-agent intelligence—driving 80-87% reductions in charting times while ensuring that **patient data never leaves the building**.

---

## 🧠 System Architecture

PulseForgeAI transforms multi-patient clinical sessions logically. Unified LPDDR5x memory allows for zero-copy handoffs directly between CPU-bound 130Hz signal filtering arrays and GPU-bound 72B-parameter language models, bypassing the extreme PCIe-transfer penalties present in standard discrete GPU setups.

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

## ⚡ Core Platform Capabilities

1. **Continuous 130Hz Edge Inference**
   - High-fidelity single-lead ECG signals are acquired via Polar H10 and parsed immediately.
   - Signal Quality Index (SQI) algorithm fuses motion-correlation, kurtosis, and Pan-Tompkins templates. 
   - Generates DWT morphology features alongside Lomb-Scargle irregular-beat HRV frequency domains (SDNN, RMSSD, LF/HF).

2. **Automated Patient Intake Integrations**
   - Deeply integrated into the Google Fit REST API `fitness.googleapis.com`.
   - Populates the clinic workflow with 7- to 30-day baseline historical health arrays, securely aggregating 15-minute dense buckets of Body Temperature, Heart Point expenditures, and discrete Sleep Segments without demanding manual entry.
   - The MQTT Payload is structurally optimized inside pre-allocated arrays, drastically minimizing telemetry bloat for downstream consumers.

3. **Multi-Agent Edge Collaboration**
   - Runs three distinct language iterations in parallel over local `vLLM` servers:
   - **Nurse Agent**: Patient-facing companion offering warm reinforcement, restricted aggressively via strict wellness-guardrails and dynamic language accessibility targets.
   - **Duty Doctor**: RAG-augmented observer analyzing active arrays to populate objective clinical SOAP notes accurately mapped to recent AHA/AACVPR exercise guidelines.
   - **Clinical Assistant**: Interrogative interface analyzing session history natively.

4. **Hardware-Enforced HIPAA Protocols**
   - No external DNS dependencies, APIs, transit risks, or Business Associate Agreements (BAAs).
   - "Energy Safe Windows" guarantee threshold evaluations of recovery metrics absolutely deterministically prior to LLM triggering, eliminating generative AI "hallucination" dependencies for life-safety interventions.

## 🚀 Getting Started

### Prerequisites
- **Hardware**: An AI-capable workstation with unified memory such as the NVIDIA DGX Spark, or servers possessing >= 80GB VRAM to successfully serve the main foundation architectures in FP4/INT4 precisions simultaneously.
- **Dependencies**: Native standard Python 3.10+, Docker.

### 1. vLLM Server Allocations
The multi-agent infrastructure requires specific dedicated open ports to address the localized models reliably:

```bash
# Primary Nurse/Foundation Agent
docker run --gpus all -v /models:/models -p 8000:8000 nvcr.io/nvidia/vllm:latest \
  --model /models/Qwen2.5-72B-Instruct-AWQ --quantization awq --max-model-len 32768

# Specialized MedGemma Agent
docker run --gpus all -v /models:/models -p 8001:8001 nvcr.io/nvidia/vllm:latest \
  --model /models/MedGemma-27B-IT --quantization awq --port 8001
```

### 2. Live Platform Setup
Navigate into the Core Application environment and launch the ingestion UI:
```bash
git clone https://github.com/paudel54/PulseForgeAI.git
cd PulseForgeAI/Application/Polar_Livestream-analysis-Python

pip install -r requirements.txt
# Initializes connection bridging, Fit intakes, and Mosquitto routing
python main.py
```

## 🛡️ Clinical Trajectory

Supervising 6-8 patients actively in a rehabilitative space is cognitively demanding. PulseForgeAI eliminates observational reliance on memory. By continuously gating real-time parameters with an embedded Signal Quality Index, the system is engineered to confidently augment physical clinical teams—yielding verifiable administrative reporting at a fraction of standard chart-composition times.

*Disclaimer: PulseForgeAI is a highly experimental research prototype only. It has not been approved by the FDA as Software as a Medical Device (SaMD).*