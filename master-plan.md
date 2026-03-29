# Talk to Your Heart — Multi-Agent Edge AI for In-Clinic Cardiac Rehabilitation

> **One-line vision:** A DGX Spark-powered, HIPAA-compliant-by-architecture intelligence platform that turns raw wearable ECG into real-time multi-patient monitoring, automated SOAP documentation, and role-specific AI copilots — all on-premise, all concurrent, zero patient data leaving the building.

## Problem

Every year in the United States, hundreds of thousands of people survive a heart attack or cardiac procedure only to face a second, quieter crisis: the rehabilitation that could keep them alive is failing them. Cardiac rehabilitation is a Class Ia recommended therapy proven to reduce all-cause mortality by 13% and hospitalizations by 31%, yet only 24% of eligible Medicare beneficiaries ever attend a single session, and barely 27% of those who start will finish the full 36-session program. Fewer than 1% of U.S. hospitals meet the CMS Million Hearts target of 70% participation. The gap between what cardiac rehab can do and what it actually delivers represents one of the largest preventable losses of life in modern cardiology — the CMS Million Hearts initiative estimates that closing this gap would save 25,000 lives and prevent 180,000 hospitalizations every single year.

The crisis runs deeper than access alone. Inside clinics that operate cardiac rehabilitation programs, the care model is stretched thin. A single supervising clinician may oversee six to ten patients exercising simultaneously, each wearing a heart rate monitor, each responding differently to exertion, each recovering at a different rate. The clinician is expected to watch every screen, catch every subtle shift in heart rate recovery, notice when a patient's effort drifts outside their prescription, and document it all — often by hand, often after the session, often from memory. Wearable sensors generate rich physiologic data during these sessions — continuous ECG, beat-to-beat heart rate variability, movement intensity, recovery dynamics — but that data flows into fragmented displays with no unified interpretation, no intelligent alerting beyond simple threshold alarms, and no automated documentation.

At the same time, cloud-based AI tools introduce HIPAA compliance exposure, latency risk, and operational dependency on third-party infrastructure. For a system that needs to respond in real time to a patient whose heart rate is not recovering as expected, a cloud round-trip is architecturally wrong.

Among 366,000 eligible Medicare fee-for-service beneficiaries studied, hospital-level variation in enrollment spans 10-fold. Hispanic and non-Hispanic Black patients participate at roughly half the rate of White patients. CR participants show 48 fewer subsequent inpatient hospitalizations per 1,000 beneficiaries per year and $1,005 lower Medicare expenditures per beneficiary per year. Every CR session is associated with a 1.8% lower incidence of 1-year cardiac readmission. The U.S. cardiac rehab market stands at $984 million, projected to $1.39 billion by 2030, with the AI-driven platform segment growing at 21.1% CAGR toward $3.66 billion — yet the core participation problem remains unsolved.

What cardiac rehabilitation needs is an intelligent system that lives inside the clinic, processes patient data without sending it offsite, supports multiple patients and multiple clinical roles simultaneously, and turns raw physiologic streams into actionable care intelligence in real time — a system where better software directly translates into fewer deaths.

## Solution

We are building **Talk to Your Heart**, an on-campus cardiac rehabilitation intelligence platform powered by **NVIDIA DGX Spark**. The system is purpose-built for supervised clinical sessions where multiple patients exercise simultaneously and clinicians need live support, structured monitoring, and efficient documentation.

The platform ingests live physiologic data from Polar H10 chest straps over BLE, performs real-time signal processing on DGX Spark, publishes structured patient state over MQTT, stores and retrieves patient vitals and ECG embeddings through ChromaDB, augments clinical reasoning with RAG-retrieved AHA/AACVPR guidelines, and routes structured context into a multi-agent AI workflow where each agent serves a distinct clinical role. The entire pipeline — from raw ECG waveform to SOAP note — runs on campus. No patient data leaves the building.

The system deploys three role-specific AI agents coordinated by a Lead Agent Orchestrator:

**Nurse Agent (Qwen3)** — the patient-facing communication layer. Translates complex physiologic state into warm, understandable language. Provides patient education and encouragement calibrated to actual effort. Operates under strict wellness-framing guardrails: never diagnoses, never recommends medication changes, routes emergency-keyword input to hardcoded safety responses without LLM involvement.

**Duty Doctor Agent (MedGemma-27B)** — the multi-patient clinical oversight layer. Runs every 15 minutes or on event triggers, reviewing every active patient. Cross-references patient intake data, RAG-retrieved literature, and embedding search results. Prepares structured SOAP notes. Acts as a second set of eyes that never gets distracted across a room of eight exercising patients.

**Clinical Assistant Agent (MedGemma-27B)** — the clinician-facing interactive reasoning layer. Powers the Doctor Chat Interface for targeted questions: What changed in this patient's HRV? Is recovery slower than baseline? Generate a session summary for charting. Transforms the rehab session into a searchable, interpretable clinical record.

The safety architecture enforces a hard boundary: every critical clinical alert is generated by deterministic rule-based logic, never by an LLM. The utilitarian case is direct — if closing the rehab participation gap saves 25,000 lives annually, then every improvement in clinic throughput, monitoring quality, and documentation speed contributes to that number.

## Why DGX Spark

DGX Spark is the architectural foundation that makes this system possible. Every design decision traces to a specific capability that no alternative hardware provides at this form factor.

### Unified Memory Makes Multi-Agent Edge AI Viable

The system must run signal processing, multiple foundation models, multiple LLMs, vector retrieval, and a message broker simultaneously as concurrent real-time services. DGX Spark's 128 GB unified LPDDR5x at 273 GB/s bandwidth is the only commercially available configuration that fits this workload:

```
+=====================================================+
|         DGX SPARK MEMORY ALLOCATION                 |
|         (128 GB Unified LPDDR5x @ 273 GB/s)        |
|                                                     |
|   Qwen2.5-72B-Instruct-AWQ (INT4)  ~37 GB          |
|   MedGemma-27B (INT4 quantized)     ~14 GB          |
|   Qwen3 (smaller variant)           ~ 8 GB          |
|   CLEF ECG foundation model         ~ 0.2 GB        |
|   ECG-FM foundation model           ~ 0.4 GB        |
|   NormWear multimodal model         ~ 0.3 GB        |
|   PubMedBERT embedding model        ~ 0.4 GB        |
|   ChromaDB indices + data           ~ 4 GB          |
|   vLLM KV cache                     ~20 GB          |
|   MQTT broker + services            ~ 1 GB          |
|   Signal processing buffers         ~ 2 GB          |
|   OS + system overhead              ~ 8 GB          |
|   ----------------------------------------          |
|   TOTAL ESTIMATED                   ~95 GB          |
|   REMAINING HEADROOM                ~33 GB          |
+=====================================================+
```

Unified coherent memory means zero-copy handoff between CPU signal processing and GPU model inference — the CPU writes structured patient state into the same physical memory the GPU reads for vLLM context. On discrete-GPU systems, this requires explicit PCIe transfers adding latency to every inference call.

### HIPAA Compliance Through Architecture

Cloud AI introduces three HIPAA risks: data in transit, data at rest on third-party infrastructure, and vendor access to PHI. DGX Spark eliminates all three by design — patient data never leaves the building. The compliance conversation changes from "show us your BAAs and encryption certificates" to "the data stays on this device in this room."

### Why Not Alternatives?

**Cloud (AWS/GCP/Azure):** Fails HIPAA architecture, adds 100-500ms network latency per inference, creates ongoing cost dependency. **Consumer GPU (RTX 4090, 24GB):** Cannot fit even the primary 37GB LLM. **Professional GPU (A6000 48GB):** Fits one LLM but cannot simultaneously host clinical model + foundation models + retrieval + KV cache. **Apple Silicon (M4 Ultra, 192GB):** Has memory but ~20x less GPU compute than Blackwell — multi-agent serving becomes infeasible at clinical latency.

DGX Spark is the only device combining 128 GB unified memory, Blackwell-class compute (1 PFLOP FP4), desktop form factor, and the NVIDIA inference stack in a single package deployable in a clinical environment.

## Innovation

Talk to Your Heart represents the first convergence of five capabilities never combined for cardiac rehabilitation:

**1. Foundation-model ECG analysis on wearable single-lead signal.** CLEF (Nokia Bell Labs, December 2025) uses clinically-guided contrastive learning to eliminate the single-lead performance gap. We deploy CLEF + NormWear + ECG-FM in a production clinical pipeline — no cardiac rehab system has used ECG foundation models for real-time monitoring.

**2. Multi-agent clinical AI with role separation on edge hardware.** Three specialized agents serving different clinical roles with different outputs, all concurrent on a single on-premise device — unprecedented in clinical AI.

**3. Embedding-based exercise-versus-stress discrimination.** Our Embedding Search Agent performs L1 distance comparison against reference profiles, transforming alerting from threshold-crossing to pattern-aware clinical discrimination.

**4. Signal-quality-aware AI interpretation.** Per-segment SQI scores (0.0–1.0) travel with every data point. Agents know whether an HRV drop occurred during clean signal or motion artifact — confidence-weighted interpretation no competitor offers.

**5. Automated SOAP notes grounded in structured physiologic features + retrieved evidence.** Generated from deterministic pipeline output + RAG guidelines, not clinician recall or speech transcription.

## Architecture

### System Architecture Overview

```
+===========================================================================+
|                          NVIDIA DGX Spark                                 |
|                  (128 GB Unified LPDDR5x / GB10 Grace Blackwell)          |
|                                                                           |
|  +----------------+     +----------------+     +------------------------+ |
|  |  Polar H10     | BLE |  Python Signal | pub |      MQTT Broker       | |
|  |  Chest Strap   |---->|  Processing    |---->|   (broker.emqx.io)     | |
|  |  ECG  (130 Hz) |     |  (PyQt Thread) |     |  pulseforgeai/{id}/raw | |
|  |  ACC  (100 Hz) |     |                |     |   (5-second batches)   | |
|  |  HR   (Live)   |     |  Signal QA +   |     +----------+-------------+ |
|  +----------------+     |  Energy Safe   |                |               |
|                         +-------+--------+         sub    |    sub        |
|                                 |                   |     |     |         |
|  +----------------------------------+    +----------v-----v-----v------+  |
|  |        ChromaDB                  |    |   Lead Agent Orchestrator   |  |
|  |  patient_vitals_db               |    |   (Deterministic Router)    |  |
|  |  ecg_embedding_tables            |    +--+--------+--------+------+   |
|  |  beat_morphology_lookup          |       |        |        |          |
|  |  exercise_vs_stress_lookup       |       v        v        v          |
|  |  rag_medical_literature          |   +------+ +-------+ +--------+   |
|  |  patient_intake_db               |   |Nurse | |Duty   | |Clinical|   |
|  +----------------------------------+   |Qwen3 | |Doctor | |Asst    |   |
|  +---------------------------+          |      | |Med-   | |Med-    |   |
|  | Embedding Search Agent    |          |      | |Gemma  | |Gemma   |   |
|  | (CLEF + L1 distance)      |          +--+---+ +---+---+ +---+----+   |
|  +---------------------------+             |         |         |         |
|                                            v         v         v         |
|                                  Patient Chat  SOAP Notes  Doctor Chat   |
+===========================================================================+
```

### vLLM Deployment on DGX Spark

```bash
# Launch vLLM on DGX Spark with Qwen2.5-72B-Instruct-AWQ
docker run --gpus all \
  -v /models:/models \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:latest \
  --model /models/Qwen2.5-72B-Instruct-AWQ \
  --quantization awq \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.45 \
  --enable-prefix-caching \
  --max-num-seqs 8 \
  --tensor-parallel-size 1

# Launch MedGemma-27B on a second vLLM instance
docker run --gpus all \
  -v /models:/models \
  -p 8001:8001 \
  nvcr.io/nvidia/vllm:latest \
  --model /models/MedGemma-27B-IT \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.15 \
  --port 8001
```

### Signal Processing Pipeline (Python Implementation)

```python
import neurokit2 as nk
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch
from dataclasses import dataclass

@dataclass
class PatientVitals:
    patient_id: str
    timestamp: float
    hr_bpm: float
    hrv_rmssd_ms: float
    hrv_sdnn_ms: float
    hrv_pnn50: float
    hrv_lf_hf_ratio: float
    sqi: float
    qrs_width_ms: float
    qt_interval_ms: float
    st_deviation_mv: float
    activity_class: str       # rest|warmup|exercise|cooldown|recovery
    met_estimate: float
    dominant_freq_hz: float
    band_energy: float
    ecg_embedding: list[float]
    energy_safe: bool
    alert_level: str          # none|advisory|warning|critical

class ECGProcessor:
    def __init__(self, fs=130):
        self.fs = fs
        self.beat_templates = []  # rolling median template for SQI

    def preprocess(self, raw_ecg: np.ndarray) -> np.ndarray:
        # 4th-order Butterworth bandpass 0.5-40 Hz
        b, a = butter(4, [0.5, 40], btype='bandpass', fs=self.fs)
        filtered = filtfilt(b, a, raw_ecg)
        # 50/60 Hz notch filter
        b_notch, a_notch = iirnotch(60.0, 30.0, self.fs)
        return filtfilt(b_notch, a_notch, filtered)

    def detect_r_peaks(self, clean_ecg: np.ndarray) -> np.ndarray:
        # Multi-detector consensus: Pan-Tompkins + Hamilton
        _, info_pt = nk.ecg_peaks(clean_ecg, sampling_rate=self.fs,
                                   method="pantompkins1985")
        _, info_hm = nk.ecg_peaks(clean_ecg, sampling_rate=self.fs,
                                   method="hamilton2002")
        peaks_pt = set(info_pt["ECG_R_Peaks"])
        peaks_hm = set(info_hm["ECG_R_Peaks"])
        # Consensus: keep peaks detected by both (within 50ms tolerance)
        tolerance = int(0.050 * self.fs)
        consensus = []
        for p in peaks_pt:
            if any(abs(p - h) <= tolerance for h in peaks_hm):
                consensus.append(p)
        return np.array(sorted(consensus))

    def compute_hrv(self, r_peaks: np.ndarray) -> dict:
        rr = np.diff(r_peaks) / self.fs * 1000  # ms
        rr = rr[(rr > 300) & (rr < 2000)]       # physiologic filter
        if len(rr) < 5:
            return {"rmssd": 0, "sdnn": 0, "pnn50": 0, "lf_hf": 0}
        diff_rr = np.diff(rr)
        rmssd = np.sqrt(np.mean(diff_rr**2))
        sdnn = np.std(rr, ddof=1)
        pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100
        # Frequency domain via Welch
        freqs, psd = welch(rr, fs=1000/np.mean(rr), nperseg=min(256, len(rr)))
        lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)])
        hf = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.40)])
        lf_hf = lf / hf if hf > 0 else 0
        return {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50, "lf_hf": lf_hf}

    def compute_sqi(self, clean_ecg: np.ndarray, r_peaks: np.ndarray,
                    acc_magnitude: np.ndarray) -> float:
        if len(r_peaks) < 3:
            return 0.0
        # 1. Template matching: correlate each beat with median template
        beats = []
        win = int(0.3 * self.fs)  # 300ms window around R-peak
        for p in r_peaks[1:-1]:
            if p - win >= 0 and p + win < len(clean_ecg):
                beats.append(clean_ecg[p-win:p+win])
        if not beats:
            return 0.0
        template = np.median(beats, axis=0)
        correlations = [np.corrcoef(b, template)[0,1] for b in beats]
        template_score = np.mean(correlations)
        # 2. SNR estimation
        signal_power = np.var(template)
        noise_power = np.mean([np.var(b - template) for b in beats])
        snr_score = min(signal_power / (noise_power + 1e-10), 10) / 10
        # 3. Motion artifact correlation
        motion_score = 1.0 - min(abs(np.corrcoef(
            clean_ecg[:len(acc_magnitude)], acc_magnitude)[0,1]), 1.0)
        return float(np.clip(
            0.4 * template_score + 0.3 * snr_score + 0.3 * motion_score,
            0.0, 1.0
        ))

class AccelerometerProcessor:
    def __init__(self, fs=100):
        self.fs = fs

    def extract_features(self, acc_xyz: np.ndarray) -> dict:
        magnitude = np.sqrt(np.sum(acc_xyz**2, axis=1))
        # FFT for dominant frequency
        fft_vals = np.abs(np.fft.rfft(magnitude - np.mean(magnitude)))
        freqs = np.fft.rfftfreq(len(magnitude), 1/self.fs)
        mask = (freqs > 0.5) & (freqs < 10)
        if np.any(mask) and np.max(fft_vals[mask]) > 0:
            dominant_freq = freqs[mask][np.argmax(fft_vals[mask])]
            band_energy = float(np.sum(fft_vals[mask]**2))
        else:
            dominant_freq, band_energy = 0.0, 0.0
        return {"dominant_freq": dominant_freq, "band_energy": band_energy,
                "magnitude": magnitude}

    def classify_activity(self, dominant_freq: float, band_energy: float,
                          hr_bpm: float) -> str:
        if band_energy < 100 and hr_bpm < 80:
            return "rest"
        elif band_energy < 300 and hr_bpm < 100:
            return "warmup"
        elif dominant_freq > 1.0 and hr_bpm > 100:
            return "exercise"
        elif band_energy < 300 and hr_bpm > 90:
            return "cooldown"
        else:
            return "recovery"
```

### Energy Safe Window (Deterministic Safety Check)

```python
class EnergySafeWindow:
    def __init__(self, patient_intake: dict):
        self.age = patient_intake["age"]
        self.hr_max = 220 - self.age
        self.prescribed_min = patient_intake["prescribed_intensity_range"][0]
        self.prescribed_max = patient_intake["prescribed_intensity_range"][1]
        self.risk_factors = patient_intake.get("risk_factors", [])

    def check(self, hr_bpm: float, activity: str, sqi: float,
              hr_recovery_1min: float = None) -> tuple[bool, str]:
        """Returns (energy_safe: bool, alert_level: str)"""
        # CRITICAL: Exercise HR exceeds age-predicted max
        if activity == "exercise" and hr_bpm > 0.90 * self.hr_max:
            return False, "critical"
        # CRITICAL: Patient-reported symptoms handled by input classifier
        # WARNING: HR above prescribed maximum
        if activity == "exercise" and hr_bpm > self.prescribed_max * self.hr_max:
            return False, "warning"
        # WARNING: Delayed HR recovery (<12 bpm drop at 1 min)
        if activity == "recovery" and hr_recovery_1min is not None:
            if hr_recovery_1min < 12:
                return True, "warning"
        # ADVISORY: Poor signal quality
        if sqi < 0.5:
            return True, "advisory"
        return True, "none"
```

### MQTT Publishing and Subscription

```python
import paho.mqtt.client as mqtt
import json

# Publisher: Signal processing publishes vitals
class VitalsPublisher:
    def __init__(self, broker_host="localhost", port=1883):
        self.client = mqtt.Client(protocol=mqtt.MQTTv5)
        self.client.connect(broker_host, port)

    def publish_vitals(self, vitals: PatientVitals):
        topic = f"pulseforgeai/{vitals.patient_id}/raw"
        payload = json.dumps(vitals.__dict__, default=str)
        self.client.publish(topic, payload, qos=0)
        # Safety-critical alerts get QoS 2 (exactly-once delivery)
        if vitals.alert_level in ("warning", "critical"):
            alert_topic = f"patient/{vitals.patient_id}/alerts"
            self.client.publish(alert_topic, payload, qos=2)

# Subscriber: MQTT Recv Script routes to orchestrator
class AgentRouter:
    def __init__(self):
        self.client = mqtt.Client(protocol=mqtt.MQTTv5)
        self.client.on_message = self.route_message
        self.client.connect("localhost", 1883)
        self.client.subscribe("patient/+/vitals")
        self.client.subscribe("patient/+/alerts", qos=2)

    def route_message(self, client, userdata, msg):
        vitals = json.loads(msg.payload)
        patient_id = vitals["patient_id"]
        if "/alerts" in msg.topic:
            self.alert_engine.handle(vitals)      # Deterministic, no LLM
            self.dashboard.flash_alert(patient_id) # Immediate UI update
        # Update patient state in ChromaDB
        self.state_manager.update(patient_id, vitals)
        # Feed orchestrator for agent dispatch
        self.orchestrator.ingest(patient_id, vitals)
```

### ChromaDB Collection Setup

```python
import chromadb
from chromadb.config import Settings

chroma = chromadb.PersistentClient(path="/data/chromadb",
    settings=Settings(anonymized_telemetry=False))

# 6 collections for the complete retrieval architecture
patient_vitals = chroma.get_or_create_collection(
    name="patient_vitals_db",
    metadata={"description": "Rolling window of structured vitals per patient"})

ecg_embeddings = chroma.get_or_create_collection(
    name="ecg_embedding_tables",
    metadata={"description": "CLEF 768-dim ECG morphology embeddings"})

beat_morphology = chroma.get_or_create_collection(
    name="beat_morphology_lookup",
    metadata={"description": "Reference ECG templates for L1 distance scoring"})

exercise_stress = chroma.get_or_create_collection(
    name="exercise_vs_stress_lookup",
    metadata={"description": "Reference profiles: exercise vs pathologic stress"})

rag_literature = chroma.get_or_create_collection(
    name="rag_medical_literature",
    metadata={"description": "AHA/AACVPR guidelines, medical texts, HRV papers"},
    embedding_function=pubmedbert_embedding_fn)  # PubMedBERT embeddings

patient_intake = chroma.get_or_create_collection(
    name="patient_intake_db",
    metadata={"description": "Patient demographics, conditions, risk factors"})
```

### RAG Ingestion Pipeline

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# PubMedBERT for medical domain embeddings
embed_model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

def ingest_medical_literature(file_path: str, source_name: str):
    with open(file_path) as f:
        text = f.read()
    chunks = splitter.split_text(text)
    embeddings = embed_model.encode(chunks).tolist()
    rag_literature.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{source_name}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": source_name, "chunk_idx": i} for i in range(len(chunks))]
    )

# Ingest guidelines
ingest_medical_literature("aha_aacvpr_2024_cr_guidelines.txt", "AHA_CR_2024")
ingest_medical_literature("cardiac_rehab_exercise_prescription.txt", "ExRx_Guidelines")
ingest_medical_literature("hrv_clinical_review.txt", "HRV_Review")
```

### Embedding Search Agent

```python
import numpy as np

class EmbeddingSearchAgent:
    """Non-LLM retrieval agent for ECG morphology comparison."""

    def search_morphology(self, ecg_embedding: list[float],
                          patient_id: str) -> dict:
        # L1 distance against beat morphology lookup
        results = beat_morphology.query(
            query_embeddings=[ecg_embedding], n_results=5,
            include=["distances", "metadatas", "documents"])
        min_distance = results["distances"][0][0]
        closest_label = results["metadatas"][0][0].get("label", "unknown")

        # L1 distance against exercise vs stress profiles
        stress_results = exercise_stress.query(
            query_embeddings=[ecg_embedding], n_results=3,
            include=["distances", "metadatas"])

        # Historical comparison: same patient prior sessions
        history = ecg_embeddings.query(
            query_embeddings=[ecg_embedding], n_results=5,
            where={"patient_id": patient_id},
            include=["distances", "metadatas"])

        return {
            "morphology_match": closest_label,
            "morphology_distance": min_distance,
            "stress_vs_exercise": stress_results["metadatas"][0],
            "historical_similarity": history["distances"][0],
            "is_consistent_with_history": np.mean(history["distances"][0]) < 0.5
        }
```

### Agent System Prompts

```python
NURSE_AGENT_SYSTEM_PROMPT = """You are the Talk to Your Heart Wellness Companion — a warm,
supportive assistant helping patients during their cardiac rehabilitation session.

CRITICAL RULES:
1. You are a WELLNESS companion, NOT a medical provider.
2. NEVER use diagnostic language: no "abnormal", "disease", "diagnosis", "arrhythmia",
   "fibrillation", "condition indicates", "you have".
3. Frame ALL observations as trends and descriptions, NEVER as clinical assessments.
   YES: "Your heart rate dropped 18 bpm in the first minute after exercise"
   NO:  "Your cardiac recovery is abnormal"
4. NEVER recommend medication changes or specific clinical actions.
5. ALWAYS suggest consulting the care team for any medical concerns.
6. Use simple, warm language. No medical jargon. No abbreviations.
7. If the patient mentions chest pain, difficulty breathing, dizziness, or feeling faint,
   respond ONLY with: "Please stop exercising immediately and alert the nearest staff
   member. Your safety is the priority." Do not add anything else.

You receive structured session data including heart rate, activity phase, signal quality,
and exercise metrics. Translate these into supportive, encouraging language.

Temperature: 0.5 for natural warmth while maintaining factual reliability."""

DUTY_DOCTOR_SYSTEM_PROMPT = """You are the Talk to Your Heart Clinical Review Agent — a
structured clinical documentation assistant for cardiac rehabilitation supervision.

You review patient physiologic data and generate SOAP notes as administrative documentation.

CONTEXT PROVIDED PER REVIEW:
- Current vitals JSON (HR, HRV, morphology, activity phase, SQI, METs)
- Patient intake record (age, weight, conditions, risk factors, prescribed intensity)
- Prior session history from database
- ECG embedding search results (morphology similarity, exercise vs stress comparison)
- RAG-retrieved AHA/AACVPR guideline excerpts
- Active alerts and their trigger data
- Signal quality scores for confidence weighting

RULES:
1. Generate SOAP notes structured as: Subjective, Objective, Assessment, Plan.
2. Include signal quality percentage in Objective section.
3. Flag any metrics where SQI < 0.7 with "[low confidence - signal quality reduced]".
4. Reference patient intake data (conditions, risk factors) in Assessment.
5. Cite RAG-retrieved guidelines when making exercise response comparisons.
6. Assessment and Plan are SUGGESTIONS for clinician review, not directives.
7. Note when current session deviates from prior session trends.

Temperature: 0.3 for maximum factual reliability."""

CLINICAL_ASSISTANT_SYSTEM_PROMPT = """You are the Talk to Your Heart Clinical Assistant —
an interactive AI supporting clinicians with patient-specific questions during and after
cardiac rehabilitation sessions.

You answer clinician queries using the full patient context: current vitals, intake data,
session history, embedding comparisons, and retrieved medical literature.

RULES:
1. Be specific and data-grounded. Cite actual values from the patient state.
2. When comparing sessions, reference specific dates, HR values, and recovery metrics.
3. Flag confidence levels based on signal quality scores.
4. If asked to generate summaries or reports, include all relevant metrics.
5. You may use clinical terminology — your audience is healthcare professionals.
6. Reference RAG-retrieved guidelines when relevant to interpretation.

Temperature: 0.3 for maximum factual reliability."""
```

### Lead Agent Orchestrator

```python
import asyncio
from datetime import datetime, timedelta

class LeadOrchestrator:
    def __init__(self, vllm_nurse_url, vllm_doctor_url, chroma_client):
        self.nurse_url = vllm_nurse_url        # Qwen3 endpoint
        self.doctor_url = vllm_doctor_url      # MedGemma-27B endpoint
        self.chroma = chroma_client
        self.patient_states = {}
        self.last_duty_review = {}

    async def assemble_context(self, patient_id: str) -> dict:
        """8-source context assembly for agent dispatch."""
        vitals = self.patient_states.get(patient_id, {})
        intake = patient_intake.get(where={"patient_id": patient_id})
        history = patient_vitals.query(
            query_texts=[f"patient {patient_id} session"],
            where={"patient_id": patient_id}, n_results=5)
        embedding_results = self.embedding_agent.search_morphology(
            vitals.get("ecg_embedding", []), patient_id)
        rag_context = rag_literature.query(
            query_texts=[f"cardiac rehab HR {vitals.get('hr_bpm')} "
                         f"recovery HRV exercise response"],
            n_results=5)
        return {
            "current_vitals": vitals,
            "patient_intake": intake,
            "session_history": history,
            "embedding_search": embedding_results,
            "rag_guidelines": rag_context,
            "active_alerts": vitals.get("alert_level", "none"),
            "sqi": vitals.get("sqi", 0),
            "exercise_vs_stress": embedding_results.get("stress_vs_exercise")
        }

    async def route(self, event_type: str, patient_id: str, data: dict):
        context = await self.assemble_context(patient_id)

        if event_type == "patient_chat":
            # Layer 2: Emergency keyword check BEFORE LLM
            if self.contains_emergency_keywords(data.get("message", "")):
                return {
                    "response": "Please stop exercising immediately and alert "
                                "the nearest staff member. Your safety is the priority.",
                    "alert": "critical"
                }
            response = await self.call_agent(self.nurse_url,
                NURSE_AGENT_SYSTEM_PROMPT, context, data["message"])
            # Layer 3: Output validation
            if self.contains_diagnostic_language(response):
                return {"response": "I'm here to support your rehab session. "
                        "For medical questions, please speak with your care team."}
            return {"response": response}

        elif event_type == "duty_review" or event_type == "alert":
            return await self.call_agent(self.doctor_url,
                DUTY_DOCTOR_SYSTEM_PROMPT, context, "Generate SOAP note review")

        elif event_type == "clinician_chat":
            return await self.call_agent(self.doctor_url,
                CLINICAL_ASSISTANT_SYSTEM_PROMPT, context, data["message"])

    async def scheduled_duty_review(self):
        """15-minute review cycle for all active patients."""
        while True:
            for patient_id in self.patient_states:
                await self.route("duty_review", patient_id, {})
            await asyncio.sleep(900)  # 15 minutes

    @staticmethod
    def contains_emergency_keywords(text: str) -> bool:
        keywords = ["chest pain", "can't breathe", "cannot breathe",
                    "dizzy", "passing out", "faint", "heart racing",
                    "nauseous", "blacking out", "tight chest"]
        return any(kw in text.lower() for kw in keywords)

    @staticmethod
    def contains_diagnostic_language(text: str) -> bool:
        blocked = ["diagnose", "diagnosis", "abnormal", "disease",
                   "you have", "arrhythmia", "fibrillation", "prescribe",
                   "take medication", "stop taking", "condition indicates"]
        return any(term in text.lower() for term in blocked)
```

### FastAPI Backend Endpoints

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Talk to Your Heart — DGX Spark Backend")

@app.websocket("/ws/ecg/{patient_id}")
async def ecg_stream(websocket: WebSocket, patient_id: str):
    """Receives raw sensor data from Flutter app over WebSocket."""
    await websocket.accept()
    processor = ECGProcessor(fs=130)
    acc_processor = AccelerometerProcessor(fs=100)
    publisher = VitalsPublisher()
    intake = get_patient_intake(patient_id)
    safety = EnergySafeWindow(intake)

    while True:
        data = await websocket.receive_json()
        ecg_raw = np.array(data["ecg"])
        acc_xyz = np.array(data["acc"]).reshape(-1, 3)
        # Signal processing pipeline
        clean = processor.preprocess(ecg_raw)
        r_peaks = processor.detect_r_peaks(clean)
        hrv = processor.compute_hrv(r_peaks)
        acc_features = acc_processor.extract_features(acc_xyz)
        hr = 60000 / np.mean(np.diff(r_peaks) / 130 * 1000) if len(r_peaks) > 1 else 0
        sqi = processor.compute_sqi(clean, r_peaks, acc_features["magnitude"])
        activity = acc_processor.classify_activity(
            acc_features["dominant_freq"], acc_features["band_energy"], hr)
        energy_safe, alert = safety.check(hr, activity, sqi)
        # Publish structured state
        vitals = PatientVitals(
            patient_id=patient_id, timestamp=data["timestamp"],
            hr_bpm=hr, hrv_rmssd_ms=hrv["rmssd"], hrv_sdnn_ms=hrv["sdnn"],
            hrv_pnn50=hrv["pnn50"], hrv_lf_hf_ratio=hrv["lf_hf"],
            sqi=sqi, qrs_width_ms=0, qt_interval_ms=0, st_deviation_mv=0,
            activity_class=activity, met_estimate=hr/40,  # simplified
            dominant_freq_hz=acc_features["dominant_freq"],
            band_energy=acc_features["band_energy"],
            ecg_embedding=[], energy_safe=energy_safe, alert_level=alert)
        publisher.publish_vitals(vitals)
        # Send processed data back to Flutter for visualization
        await websocket.send_json(vitals.__dict__)

@app.post("/api/chat/patient/{patient_id}")
async def patient_chat(patient_id: str, message: dict):
    return await orchestrator.route("patient_chat", patient_id, message)

@app.post("/api/chat/clinician/{patient_id}")
async def clinician_chat(patient_id: str, message: dict):
    return await orchestrator.route("clinician_chat", patient_id, message)

@app.get("/api/dashboard/active")
async def active_patients():
    return orchestrator.get_all_patient_states()

@app.get("/api/session/{patient_id}/soap")
async def generate_soap(patient_id: str):
    return await orchestrator.route("duty_review", patient_id, {})
```

### Flutter BLE Integration

```dart
// Polar H10 BLE connection using polar package
import 'package:polar/polar.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';

class PolarH10Service {
  final polar = Polar();
  late WebSocketChannel _wsChannel;
  String patientId;

  PolarH10Service({required this.patientId}) {
    _wsChannel = WebSocketChannel.connect(
      Uri.parse('ws://dgx-spark-ip:8000/ws/ecg/$patientId'));
  }

  Future<void> startStreaming(String deviceId) async {
    await polar.connectToDevice(deviceId);

    // Stream ECG data at 130 Hz
    polar.startEcgStreaming(deviceId).listen((ecgData) {
      final samples = ecgData.samples.map((s) => s.voltage).toList();
      _wsChannel.sink.add(jsonEncode({
        'ecg': samples,
        'timestamp': DateTime.now().millisecondsSinceEpoch / 1000,
      }));
    });

    // Stream accelerometer data at 100 Hz
    polar.startAccStreaming(deviceId).listen((accData) {
      final samples = accData.samples.map((s) => [s.x, s.y, s.z]).toList();
      _wsChannel.sink.add(jsonEncode({
        'acc': samples,
        'timestamp': DateTime.now().millisecondsSinceEpoch / 1000,
      }));
    });

    // Listen for processed vitals from DGX Spark
    _wsChannel.stream.listen((message) {
      final vitals = jsonDecode(message);
      updateDashboard(vitals);  // Update Flutter UI
    });
  }
}
```

### Foundation Model Layer

**CLEF (Nokia Bell Labs, December 2025)** — Clinically-guided contrastive learning for single-lead ECG embedding. Outperforms self-supervised baselines by ≥2.6% AUROC on lead-I. ~200 MB. Open-weight.

**NormWear (December 2024, Apache 2.0)** — Multimodal wearable data (ECG + accelerometer + PPG) via continuous wavelet transform tokenization. Zero-shot inference. ~300 MB.

**ECG-FM (University of Toronto, JAMIA Open October 2025)** — 90.9M-parameter wav2vec 2.0 pretrained on 1.5M ECGs. 0.996 AUROC for AF detection. ~400 MB.

**Single-Lead Strategy.** CLEF eliminates the single-lead generalization gap. Multimodal augmentation (accelerometer + HRV context) compensates for spatial information lost without 12 leads.

### Foundation Model Embedding Pipeline (Implementation)

```python
import torch
import numpy as np
from scipy.signal import resample

class ECGEmbeddingService:
    """Runs CLEF and ECG-FM inference on 5-second ECG windows.
    Deployed as a FastAPI microservice on DGX Spark, called by the
    signal processing pipeline before MQTT publish."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        # CLEF: contrastive learning encoder, 768-dim output
        self.clef_model = torch.jit.load("/models/clef_encoder.pt").to(self.device)
        self.clef_model.eval()
        # ECG-FM: wav2vec 2.0, 768-dim output
        self.ecgfm_model = torch.jit.load("/models/ecgfm_encoder.pt").to(self.device)
        self.ecgfm_model.eval()

    @torch.no_grad()
    def get_clef_embedding(self, ecg_segment: np.ndarray, fs: int = 130) -> list[float]:
        """CLEF expects 500 Hz input, 5-second windows (2500 samples)."""
        # Resample from Polar H10 native 130 Hz to CLEF's expected 500 Hz
        target_len = int(len(ecg_segment) * 500 / fs)
        resampled = resample(ecg_segment, target_len)
        # Normalize to zero-mean unit-variance
        resampled = (resampled - np.mean(resampled)) / (np.std(resampled) + 1e-8)
        tensor = torch.FloatTensor(resampled).unsqueeze(0).unsqueeze(0).to(self.device)
        embedding = self.clef_model(tensor)  # [1, 768]
        return embedding.squeeze().cpu().numpy().tolist()

    @torch.no_grad()
    def get_ecgfm_embedding(self, ecg_segment: np.ndarray, fs: int = 130) -> list[float]:
        """ECG-FM expects 500 Hz input, processes via wav2vec 2.0 feature extractor."""
        target_len = int(len(ecg_segment) * 500 / fs)
        resampled = resample(ecg_segment, target_len)
        resampled = (resampled - np.mean(resampled)) / (np.std(resampled) + 1e-8)
        tensor = torch.FloatTensor(resampled).unsqueeze(0).to(self.device)
        features = self.ecgfm_model.extract_features(tensor)  # [1, T, 768]
        # Mean-pool over time dimension for fixed-size embedding
        embedding = features.mean(dim=1)  # [1, 768]
        return embedding.squeeze().cpu().numpy().tolist()

    def get_combined_embedding(self, ecg_segment: np.ndarray, fs: int = 130) -> list[float]:
        """Concatenate CLEF + ECG-FM for richer morphology representation.
        Falls back to CLEF-only if ECG-FM inference fails."""
        clef_emb = self.get_clef_embedding(ecg_segment, fs)
        try:
            ecgfm_emb = self.get_ecgfm_embedding(ecg_segment, fs)
            return clef_emb + ecgfm_emb  # 1536-dim combined
        except Exception:
            return clef_emb  # 768-dim fallback

# Integration into the signal processing pipeline:
# Called every 5 seconds on the latest ECG window
embedding_service = ECGEmbeddingService(device="cuda")

def enrich_vitals_with_embedding(vitals: PatientVitals, ecg_window: np.ndarray):
    """Called in the FastAPI WebSocket handler after signal processing."""
    if len(ecg_window) >= 650:  # At least 5 seconds at 130 Hz
        vitals.ecg_embedding = embedding_service.get_combined_embedding(ecg_window)
    # Store embedding in ChromaDB for longitudinal comparison
    ecg_embeddings.add(
        embeddings=[vitals.ecg_embedding],
        ids=[f"{vitals.patient_id}_{vitals.timestamp}"],
        metadatas=[{"patient_id": vitals.patient_id,
                    "timestamp": str(vitals.timestamp),
                    "hr_bpm": vitals.hr_bpm,
                    "activity": vitals.activity_class,
                    "sqi": vitals.sqi}])
    return vitals
```

## Features

- Real-time multi-patient cardiac rehab monitoring on NVIDIA DGX Spark — zero patient data leaves the facility
- HIPAA compliance through architectural data locality, not cloud BAAs
- Continuous single-lead ECG at 130 Hz from Polar H10 with Pan-Tompkins + Hamilton consensus QRS detection
- Beat-to-beat HR and comprehensive HRV: SDNN, RMSSD, pNN50, LF/HF over configurable windows
- ECG morphology extraction: QRS width, QT interval, ST deviation per beat
- Signal quality scoring (SQI 0.0–1.0) via template matching + SNR + accelerometer motion correlation
- Accelerometer activity classification (rest/warmup/exercise/cooldown/recovery) from FFT dominant frequency and band energy
- VO2/MET estimation from HR + accelerometer for exercise prescription validation
- Energy Safe Window: deterministic safety check before every MQTT publish
- ECG foundation models (CLEF, NormWear, ECG-FM) for morphology embedding and anomaly scoring on DGX Spark
- Embedding Search Agent with L1 distance scoring against morphology lookup and exercise-versus-stress profiles
- ChromaDB: 6 collections (vitals, ECG embeddings, morphology, exercise/stress, RAG literature, intake)
- RAG grounded in AHA/AACVPR 2024 guidelines — 512-token chunks, PubMedBERT embeddings, top-5 cosine similarity
- MQTT (Mosquitto) with per-patient topic isolation and QoS 2 for safety alerts
- Qwen3 Nurse Agent: compassionate patient communication with wellness guardrails
- MedGemma-27B Duty Doctor Agent: 15-minute review cycles, SOAP notes, multi-patient oversight
- MedGemma-27B Clinical Assistant Agent: Doctor Chat Interface for on-demand clinical queries
- Lead Orchestrator: deterministic routing, 8-source context assembly, 15-minute scheduling
- 4-layer safety: deterministic alerts → emergency keyword classifier → output validator → wellness system prompt
- Automated SOAP notes (Subjective/Objective/Assessment/Plan) as admin documentation outside SaMD
- Multi-session longitudinal tracking: HR recovery trends, exercise tolerance, morphology comparison
- Concurrent multi-model serving: 72B + 27B + 3 foundation models + retrieval + MQTT in 128 GB unified memory
- Flutter dashboards (patient chat + clinician multi-patient view + Doctor Chat + PDF reports)
- Two-phase regulatory: FDA 2026 general wellness now, 510(k) for clinical claims later
- Reimbursement-ready: CPT 93798 in-session + RPM 99454/99457/99458 at $99+/month per patient

## User Experience

### Patient-Facing

The patient wears a Polar H10 and interacts with a tablet chat interface. **During warm-up:** "Good morning, Maria. Your session is underway and everything looks steady." **During exercise:** "You've been at it for 12 minutes and you're right in your target zone." **If approaching limit:** "Your heart rate has climbed a bit higher than usual. You might ease back slightly." **During recovery:** "Nice cooldown — your heart rate is settling toward your resting level." **After session:** Simple summary with duration and positive closing. The patient never sees ECG waveforms, HRV numbers, or clinical terminology.

**Accessibility and equity by design:** The Nurse Agent chat interface uses large-font, high-contrast UI optimized for older adults (mean CR patient age: 63). The Qwen3 system prompt supports multilingual generation — Spanish-language mode is a configuration flag, directly addressing the documented disparity where Hispanic patients participate at roughly half the rate of White patients. The chat-first interface removes literacy barriers inherent in written educational materials.

### Clinician-Facing

```
+==========================================================================+
|              CLINICIAN MULTI-PATIENT DASHBOARD                           |
|   +-------------------+  +-------------------+  +-------------------+    |
|   | Maria Santos       |  | James Park         |  | Ruth Williams      |  |
|   | HR: 118  EXERCISE  |  | HR: 132  EXERCISE  |  | HR: 88  RECOVERY   |  |
|   | MET: 4.1 SQI: 0.92|  | MET: 5.3 SQI: 0.85|  | MET: 1.8 SQI: 0.94|  |
|   | [SAFE]             |  | [WARNING: HR HIGH] |  | [SAFE]             |  |
|   +-------------------+  +-------------------+  +-------------------+    |
+==========================================================================+
```

CRITICAL alerts flash to top. Doctor Chat Interface answers: "How is James's recovery vs last session?" with data-grounded responses. SOAP notes auto-generated at session end — 2-minute review replaces 10–15 minutes of manual documentation. Six patients per session saves 48–78 minutes.

## Market Opportunity

U.S. cardiac rehab market: $984M (2024) → $1.39B (2030) at 5.9% CAGR. AI-driven cardiac platforms: $3.66B by 2030 at 21.1% CAGR. Wearable cardiac devices: $3.87B (2025) → $25.97B (2034). The CMS Million Hearts initiative targets 70% participation (currently 24%) — achieving this saves 25,000 lives and prevents 180,000 hospitalizations annually. Revenue model: CPT 93798 per-session + RPM codes generating $99+/month per patient ($47 device supply + $52 management). New 2026 codes (99445/99470) expand billing to shorter monitoring periods. Annual RPM revenue per 100 patients: ~$118,800+.

## Competitive Landscape

```
+============================================================================+
|                    Continuous  Foundation  Multi-Agent  Auto    On-Premise  |
|   Competitor        ECG        Model ECG   Clinical    SOAP    Edge AI     |
|   --------------------------------------------------------------------------
|   Fourth Frontier    YES        no          no          no      no         |
|   Recora             no         no          no          no      no         |
|   Movn Health        no         no          no          no      no         |
|   Biofourmis         YES*       proprietary single      no      cloud      |
|   Carda Health       no         no          no          no      no         |
|   --------------------------------------------------------------------------
|   Talk to Your Heart YES        CLEF+3      3 agents    YES     DGX Spark  |
+============================================================================+
```

*Biofourmis (now General Informatics) offers FDA-cleared continuous ECG via their RhythmAnalytics platform and a proprietary AI engine for arrhythmia detection — the most capable competitor in the space. However, their architecture is cloud-dependent (AWS-hosted inference), operates a single unified model rather than role-separated agents, does not generate structured SOAP documentation, and requires standard HIPAA cloud compliance (BAAs, encryption-in-transit, third-party data residency). Their focus is remote patient monitoring rather than in-clinic supervised rehabilitation.

No competitor combines continuous ECG + foundation-model analysis + multi-agent clinical AI + automated SOAP notes + on-premise edge deployment. The integration complexity is the moat.

## Differentiation Strategy

What makes Talk to Your Heart fundamentally different from every existing cardiac rehab solution is not any single feature — it is the architectural convergence of five capabilities that have never coexisted in one system.

**1. HIPAA compliance is an architecture property, not a policy layer.** Every competitor that uses cloud AI requires Business Associate Agreements, encryption-in-transit certificates, and vendor audit trails. We eliminate all three categories of HIPAA risk by design — PHI never leaves the physical device. This is not a feature toggle; it is a hardware-enforced guarantee that no cloud-based competitor can match without fundamentally redesigning their infrastructure.

**2. Multi-agent role separation vs. single-model chatbots.** Existing solutions (Biofourmis, Carda Health) use at most one AI model for one purpose. We run three concurrent specialized agents — each with distinct system prompts, distinct output formats, distinct audiences, and distinct safety constraints — served simultaneously from DGX Spark's unified memory. The Nurse Agent speaks to patients in warm lay language; the Duty Doctor generates structured SOAP notes for the medical record; the Clinical Assistant answers clinician queries with data-grounded clinical reasoning. No competitor has role-separated clinical AI.

**3. Foundation-model ECG is our sensing moat.** CLEF, NormWear, and ECG-FM transform raw single-lead ECG into dense 768-dimensional embeddings that encode morphological patterns invisible to threshold-based monitoring. This enables exercise-versus-stress discrimination through L1 distance in embedding space — a capability that requires both the foundation models AND the compute to run them at the edge. Consumer hardware cannot fit these models alongside the LLMs. Cloud-based solutions add latency that defeats real-time clinical utility.

**4. Signal quality conditions AI confidence.** Every data point in our pipeline carries an SQI score (0.0–1.0). When sensor contact degrades, the agents automatically reduce their confidence and flag uncertainty. No competitor propagates signal quality metadata into AI interpretation — they either process noisy data blindly or discard it entirely.

**5. The integration complexity IS the moat.** Building any one component (BLE streaming, ECG processing, RAG, LLM serving, vector retrieval, MQTT messaging, Flutter UI) is achievable independently. Running all of them concurrently on a single edge device with unified memory, deterministic safety guarantees, and sub-5-second agent response times requires the specific architectural decisions we have made — and the specific hardware (DGX Spark) we have chosen. This integration barrier protects against fast-follow competitors.

## Regulatory and Reimbursement Strategy

**Phase 1 (Now): General Wellness.** FDA's January 6, 2026 guidance covers non-invasive physiologic sensors estimating HRV and recovery. Allowed: "Your HR dropped 18 bpm in the first minute." Prohibited: "Your cardiac recovery is abnormal." SOAP notes framed as administrative documentation outside SaMD.

**Phase 2 (12–18 months): FDA 510(k).** Predicates: Hexoskin (cleared Nov 2025), CardioTag (cleared 2025), Apple Watch ECG, AliveCor KardiaMobile. Timeline: 6–18 months, $100K–$500K.

```
+============================================================================+
|   CPT CODE    DESCRIPTION                                   REIMBURSEMENT  |
|   93798       Outpatient CR with continuous ECG              Per session    |
|   99453       Initial RPM setup + education                  $22 one-time  |
|   99454       Device supply + data (16+ days/month)          $47/month     |
|   99457       RPM management, first 20 min                   $52/month     |
|   99458       Each additional 20 min                         $41/month     |
|   99445*      Device supply, 2-15 days (NEW 2026)            $52/month     |
|   99470*      First 10 min management (NEW 2026)             $26/month     |
+============================================================================+
```

## Ecosystem Thinking — Interoperability, APIs, and Extensibility

### API Design

The system exposes three tiers of APIs:

**Tier 1 — Real-time streaming (WebSocket + MQTT):**
- `ws://dgx-spark:8000/ws/ecg/{patient_id}` — bidirectional raw sensor ingestion and processed vitals return
- MQTT topics: `patient/{id}/vitals` (QoS 0), `patient/{id}/alerts` (QoS 2 exactly-once), `patient/{id}/soap` (QoS 1)
- Any MQTT-compatible client can subscribe to patient state updates without touching the core pipeline

**Tier 2 — REST (FastAPI):**
- `POST /api/chat/patient/{id}` — Nurse Agent interaction
- `POST /api/chat/clinician/{id}` — Clinical Assistant queries
- `GET /api/dashboard/active` — All active patient states
- `GET /api/session/{id}/soap` — On-demand SOAP note generation
- `POST /api/intake/{id}` — Patient intake record creation/update

**Tier 3 — Data export:**
- FHIR R4 Observation and DiagnosticReport resource generation for EHR integration (Epic, Cerner, Meditech)
- PDF session report export for patient records
- Parquet/HDF5 raw session export for retrospective research

### Extensibility Architecture

The MQTT pub/sub pattern means adding new capabilities requires zero changes to existing services:

- **New agent?** Subscribe to `patient/+/vitals`, publish to a new output topic. Example: a Billing Code Agent that listens to session summaries and suggests CPT codes.
- **New sensor?** Publish to the same MQTT topic schema. The pipeline is sensor-agnostic — Movesense MD, Apple Watch, or Garmin can replace Polar H10 with only a BLE adapter change.
- **New foundation model?** Add an inference endpoint and register it with the Embedding Search Agent. The ChromaDB collection schema supports arbitrary embedding dimensions.
- **Multi-site deployment?** Each DGX Spark runs independently. A federated MQTT bridge can aggregate anonymized population metrics across sites without sharing PHI.

### Data Model

The canonical patient state is a structured JSON object (PatientVitals dataclass) with 18 fields covering cardiac metrics, signal quality, activity classification, safety status, and ECG embeddings. This schema is versioned and documented — any downstream consumer gets a stable contract.

## Scalability Design — Architecture Beyond the Demo

### Single-Device Scalability

One DGX Spark (128 GB unified memory, 1 PFLOP FP4) supports:
- **8 concurrent patients** with continuous 130 Hz ECG streaming and real-time signal processing
- **3 concurrent LLM agents** (72B + 27B + 8B) via vLLM with prefix caching and 8-sequence batching
- **15-minute automated SOAP review cycles** for all active patients
- **Sub-5-second Nurse Agent response** and **sub-30-second SOAP generation**
- **33 GB memory headroom** for burst workloads and additional foundation models

### Multi-Device Horizontal Scaling

For large cardiac rehab programs (15+ concurrent patients):

```
+------------------+     +------------------+     +------------------+
|  DGX Spark #1    |     |  DGX Spark #2    |     |  DGX Spark #3    |
|  Patients 1-8    |     |  Patients 9-16   |     |  Patients 17-24  |
|  Full pipeline   |     |  Full pipeline   |     |  Full pipeline   |
+--------+---------+     +--------+---------+     +--------+---------+
         |                         |                         |
         +------------+------------+------------+------------+
                      |                         |
              +-------v-------+         +-------v-------+
              | Central MQTT  |         | Clinician     |
              | Broker        |<------->| Dashboard     |
              | (Aggregation) |         | (All patients)|
              +---------------+         +---------------+
```

Each Spark runs the full pipeline independently. A central MQTT broker aggregates patient states for the unified clinician dashboard. No shared GPU memory, no distributed inference complexity — pure horizontal scaling.

### Workflow Scalability Impact

| Metric | Manual Workflow | With Talk to Your Heart | Improvement |
|--------|----------------|------------------------|-------------|
| SOAP documentation time per patient | 10–15 min | 2 min review | 80–87% reduction |
| Concurrent patient monitoring capacity | 6–8 (attention-limited) | 8+ (AI-augmented) | 33%+ increase |
| Alert detection latency | Variable (human attention) | <100ms (deterministic) | Orders of magnitude |
| Session documentation completeness | ~60% (memory-based) | ~100% (automated) | Near-complete |
| Time saved per 6-patient session | — | 48–78 minutes | Reinvested in patient care |

### Beyond Cardiac Rehab

The architecture is domain-portable. The orchestration layer, agent system, safety architecture, and MQTT messaging are rehab-agnostic. Expanding to new domains requires:
- New signal processing modules (SpO2 for pulmonary, IMU for neurological)
- New RAG knowledge bases (domain-specific guidelines)
- New agent system prompts

The DGX Spark infrastructure, ChromaDB retrieval, vLLM serving, and Flutter UI framework remain unchanged.

## Execution Plan (24 Hours)

### Phase 1: Infrastructure + BLE Pipeline (Hours 0–6) — Viggi + Sansrit
DGX Spark: Ubuntu, Mosquitto MQTT, ChromaDB, vLLM with Qwen2.5-72B-AWQ. Flutter: polar Dart BLE streaming → WebSocket → FastAPI → MQTT pipeline. Sansrit delivers working BLE → WebSocket bridge while Viggi brings up vLLM and MQTT.
**Deliverable:** Live Polar H10 data flowing through DGX Spark, published over MQTT.
**Go/no-go checkpoint (Hour 6):** If BLE is unstable, switch to mock_sensor.py synthetic stream. Pipeline must be flowing.

### Phase 2: Signal Processing + RAG (Hours 6–12) — Shiva + Rumon + Viggi
Shiva: NeuroKit2 preprocessing, consensus QRS detection, HRV, SQI. Rumon: Accelerometer activity classification, Energy Safe Window, patient intake schema. Viggi (parallel): PubMedBERT RAG ingestion into ChromaDB, guideline chunking.
**Deliverable:** Structured JSON vitals pipeline with safety alerts + populated RAG knowledge base.
**Go/no-go checkpoint (Hour 12):** Signal processing producing valid JSON vitals on MQTT. RAG returning relevant guideline chunks.

### Phase 3: Agent Deployment + Orchestration (Hours 12–18) — Shiva + Viggi
Shiva: Three agent system prompts deployed, 4-layer guardrails, Lead Orchestrator with 8-source context assembly. Viggi: Second vLLM instance (MedGemma-27B), agent endpoint integration, SOAP template.
**Deliverable:** All three agents responding to structured context with correct role behavior.
**Go/no-go checkpoint (Hour 18):** Nurse Agent responds to patient chat. Duty Doctor generates SOAP note from mock vitals. If foundation model embeddings are not yet integrated, mark as stretch goal — agents proceed with HRV-based cosine similarity matching against ChromaDB instead of CLEF L1 distance.

### Phase 4: Integration + Demo Polish (Hours 18–24) — Sansrit + All
Sansrit: Flutter clinician dashboard, patient chat UI, Doctor Chat Interface, SOAP display. All: Multi-patient testing with NeuroKit2 ecg_simulate(). Safety adversarial testing. Demo rehearsal with 3 concurrent patients.
**Deliverable:** Complete end-to-end demo from sensor to SOAP note on DGX Spark.

**Stretch goals (if time permits after Hour 18 checkpoint):**
- CLEF/ECG-FM foundation model embedding inference integration
- PDF session report export
- ECG waveform visualization in clinician dashboard
- Multilingual (Spanish) Nurse Agent configuration

## Validation and Demo

**Signal Validation:** NeuroKit2 ecg_simulate() generates synthetic ECG at 60–180 bpm with configurable noise. We verify HR/HRV accuracy against reference, SQI scoring (clean >0.9, noised <0.5), and activity classification across phase transitions.

**Alert Validation:** Synthetic scenarios for every threshold — tachycardia (CRITICAL), delayed recovery (WARNING), SQI degradation (ADVISORY), sensor disconnect. Each verified to fire without LLM and reach dashboard.

**Agent Safety Validation:** Emergency keywords ("chest pain", "can't breathe") → hardcoded response within 100ms. Output validator blocks diagnostic language. Adversarial prompt testing.

**Concurrency Validation:** 4–6 simultaneous simulated patients on DGX Spark. Targets: signal processing >130 samples/sec/stream, Nurse Agent <5s response, SOAP note <30s, memory stable over 30 minutes.

### Demo Scenario

**Patient A (Maria, 62, post-MI):** Normal session. Nurse Agent provides encouragement. Clean SOAP note at end. Demonstrates happy path.

**Patient B (James, 55, post-CABG, diabetic):** HR approaches prescribed max. Energy Safe Window triggers WARNING. Dashboard highlights James. Duty Doctor flags with intake context (diabetes risk factor). Demonstrates alert + risk-aware reasoning.

**Patient C (Ruth, 70, HF):** Sensor motion artifact → SQI drops → ADVISORY alert → agent notes reduced confidence → staff checks sensor → SQI recovers. Demonstrates signal quality awareness.

**Clinician asks:** "How is James's recovery vs last session?" → Clinical Assistant retrieves history, compares HR recovery curves, provides data-grounded answer.

## Risks and Mitigations

**Memory pressure:** Budget estimates 95 GB / 128 GB. Fallback: reduce KV cache → swap to 32B model (frees 20 GB) → time-multiplex foundation models.

**Single-lead limitation:** Scoped to rhythm/rate/HRV/recovery (validated on single-lead). CLEF eliminates generalization gap. Phase 1 wellness framing avoids claims requiring 12-lead parity.

**LLM hallucination:** 4-layer safety architecture. Deterministic alerts independent of LLMs. Output grounded in structured data + RAG, not free-form generation. Temperature 0.3.

**BLE instability:** SQI detects degradation. SENSOR DISCONNECT advisory within 5s. Graceful degradation — agents report gaps, don't interpret stale data.

**Regulatory risk:** Output validator technical safeguard. System prompt wellness framing. All text reviewed against FDA 2026 guidance. Two-phase strategy provides 510(k) path.

**Staff resistance:** System reduces workload — automated SOAP saves 10–15 min/patient, dashboard replaces scanning, chat answers faster than records. Value = time savings.

**Foundation model integration timeline:** CLEF and ECG-FM embedding pipelines require model conversion (TorchScript), resampling logic (130→500 Hz), and ChromaDB storage integration. If this takes longer than the 6-hour window allocated in Phase 3, the explicit fallback is cosine similarity on raw HRV feature vectors (RMSSD, SDNN, pNN50, LF/HF) stored in ChromaDB — this preserves the longitudinal comparison and agent context assembly pipeline while sacrificing morphology-level discrimination. The embedding pipeline is designated as a stretch goal with a defined cut point at Hour 18.

## Team Plan

**Rumon — Hardware / Product-Market Fit:** Polar H10 BLE setup, patient intake schema, clinical workflow mapping, demo scenario design, accelerometer activity classification. Phases 1, 2, 4.

**Viggi — DGX Spark Infrastructure + RAG:** vLLM deployment (72B + 27B + Qwen3), Mosquitto MQTT, ChromaDB setup, PubMedBERT RAG ingestion, foundation model serving, memory optimization. Phases 1, 2, 3, 4.

**Shiva — Signal Processing + Agent Engineering:** ECG signal processing pipeline (NeuroKit2, SQI, HRV), agent system prompts + 4-layer guardrails, Lead Orchestrator logic, SOAP note template. Phases 2, 3.

**Sansrit — Flutter / Frontend:** polar Dart BLE, WebSocket → FastAPI bridge, patient chat UI, clinician dashboard, Doctor Chat Interface, ECG visualizer. Phases 1, 4.

**Bottleneck mitigation:** Viggi absorbs RAG ingestion (previously assigned to Shiva's Phase 3) so that Shiva can focus exclusively on agent prompt engineering and orchestrator logic during Hours 12–18. Rumon picks up accelerometer feature extraction from Shiva's Phase 2 scope. This redistributes Shiva's original Phase 2+3 workload across three team members. The Hour 18 go/no-go checkpoint provides a defined cut point — if agents are functional but foundation model embeddings are not integrated, the team shifts entirely to Phase 4 demo polish.

## Vision — North Star and Roadmap

**The north star:** Every supervised cardiac rehabilitation session in the United States runs with an AI copilot that monitors, documents, and supports — and no patient data ever leaves the building.

**6 months:** Pilot deployments with 2–3 cardiac rehab programs in academic medical centers. Target 70% documentation time reduction (SOAP notes), <5% alert false positive rate, and measurable increase in clinician-reported monitoring confidence. Expand hardware compatibility to Movesense MD (medical-grade, wider clinical adoption). Begin EHR integration via FHIR R4 Observation/DiagnosticReport resources for Epic and Cerner.

**12–18 months:** Submit FDA 510(k) for rhythm analysis claims. Predicates established (Hexoskin Nov 2025, CardioTag 2025, Apple Watch ECG, AliveCor KardiaMobile). Expand to opportunistic cardiac wellness reporting — heart rate recovery during daily activities, MET tracking, 6-Minute Walk Test estimation from daily movement patterns (Cole et al., NEJM 1999: HRR ≤12 bpm at 1 minute predicts all-cause mortality). Launch multi-site federated deployment model.

**2–3 years:** Platform expansion to pulmonary rehabilitation (SpO2 + respiratory rate + spirometry), neurological rehabilitation (movement quality scoring + tremor quantification), and post-surgical recovery monitoring. The orchestration layer, agent architecture, safety framework, and MQTT messaging are domain-agnostic — new clinical domains require only new signal modules and new RAG knowledge bases. The DGX Spark deployment model, ChromaDB retrieval architecture, and vLLM serving infrastructure remain unchanged.

**The utilitarian case, restated:** If this system helps one clinic complete cardiac rehabilitation for ten more patients per year, that is ten fewer cardiac deaths. The AHA estimates 800,000 Americans have a heart attack annually. Fewer than 200,000 complete rehab. The gap is not a technology gap — the technology exists. The compute exists. The clinical evidence is overwhelming. What has been missing is the integration: the system that takes raw sensor data and turns it into clinical intelligence without requiring cloud infrastructure, without adding documentation burden, and without compromising patient privacy. Talk to Your Heart is that integration. DGX Spark is what makes it possible.