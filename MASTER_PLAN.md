# Talk to Your Heart
## On-Campus Multi-Agent Cardiac Rehabilitation Intelligence — NVIDIA DGX Spark

> **North Star:** Every improvement in clinic throughput, monitoring quality, and documentation speed directly translates into fewer deaths — because the 73% of eligible patients who never complete cardiac rehab are dying of a process failure, not a knowledge failure.

---

## MASTER PLAN — 12 DIMENSIONS

---

## 1. Problem Definition

Every year in the United States, hundreds of thousands of people survive a cardiac event only to face a second, quieter crisis: the rehabilitation that could keep them alive is failing them.

Cardiac rehabilitation (CR) is a **Class Ia recommended therapy** proven to reduce all-cause mortality by **13%** and hospitalizations by **31%** — yet only **24% of eligible Medicare beneficiaries** ever attend a single session, and fewer than **27% of those who start** complete the full 36-session program. Fewer than **1% of U.S. hospitals** meet the CMS Million Hearts target of 70% participation.

**The gap between what CR can do and what it actually delivers represents one of the largest preventable loss-of-life failures in modern cardiology.** The CMS Million Hearts initiative estimates closing this gap would save **25,000 lives** and prevent **180,000 hospitalizations** every single year.

### Inside the Clinic: The Supervision Problem

A single supervising clinician may oversee **6–10 patients exercising simultaneously**, each wearing a heart rate monitor, each recovering at a different rate. That clinician is expected to:

- Watch every screen simultaneously
- Catch subtle shifts in heart rate recovery before they become emergencies
- Notice when effort drifts outside the prescription
- Document everything — often by hand, after the session, from memory

Wearable sensors generate rich physiologic data (continuous ECG, beat-to-beat HRV, movement intensity, recovery dynamics) — but that data flows into fragmented displays with **no unified interpretation, no intelligent alerting beyond threshold alarms, and no automated documentation.**

### The Cloud AI Trap

Cloud-based AI introduces three structural problems for this use case:

1. **HIPAA exposure** — PHI in transit and at rest on third-party infrastructure
2. **Latency** — 100–500ms network round-trips make real-time alerting architecturally wrong
3. **Operational dependency** — outages during sessions create clinical risk, not just inconvenience

### Quantified Stakes

| Metric | Value |
|--------|-------|
| Eligible Medicare CR patients | 366,000 studied |
| Current completion rate | 27% |
| CR participation difference in hospitalizations | 48 fewer per 1,000 beneficiaries/year |
| CR participation difference in cost | $1,005 lower Medicare spend/beneficiary/year |
| Each session's impact on 1-year readmission | 1.8% lower incidence |
| Hospital-level enrollment variation | 10-fold |
| Black/Hispanic participation gap | ~50% of White participation rate |
| Lives saved if gap closes | 25,000/year |
| Hospitalizations prevented if gap closes | 180,000/year |

**Who experiences this:**
- **Patients:** Post-MI, post-CABG, post-valve, heart failure — the highest-risk population in cardiology, navigating complex recovery with minimal real-time support
- **Supervising clinicians:** RNs and exercise physiologists managing 6–10 patients from a single station with no AI assistance and 10–15 minutes of documentation per patient per session
- **Health systems:** Compliance pressure, reimbursement risk, and staff burnout from documentation load

---

## 2. Solution — Talk to Your Heart

**Talk to Your Heart** is an on-campus cardiac rehabilitation intelligence platform built on **NVIDIA DGX Spark**. The system is purpose-built for supervised in-clinic sessions where multiple patients exercise simultaneously and clinicians need live support, structured monitoring, and automated documentation.

### What It Does

```
+==============================================================+
|                    WHAT THE SYSTEM DELIVERS                  |
+==============================================================+
|  FOR PATIENTS                  |  FOR CLINICIANS             |
|  - Real-time encouragement     |  - Multi-patient dashboard  |
|  - Effort-calibrated coaching  |  - Automated SOAP notes     |
|  - Post-session summary        |  - Doctor Chat Interface    |
|  - Emergency safety routing    |  - Alert intelligence       |
|--------------------------------|-----------------------------|
|  FOR THE CLINIC                |  FOR COMPLIANCE             |
|  - Session documentation       |  - Zero cloud PHI exposure  |
|  - Longitudinal trend data     |  - Deterministic safety     |
|  - CPT/RPM billing support     |  - On-campus data residency |
+==============================================================+
```

### Three-Agent Architecture

The platform deploys three role-specific AI agents coordinated by a Lead Agent Orchestrator:

**Nurse Agent (Qwen3-32B-FP4)**
Patient-facing communication. Translates physiologic state into warm, understandable language. Calibrated encouragement to actual effort. Strict wellness-framing guardrails — never diagnoses, never recommends medication changes, routes emergency keywords to hardcoded safety responses without LLM involvement.

**Duty Doctor Agent (MedGemma-27B, 87.7% MedQA)**
Multi-patient clinical oversight. Runs on 15-minute cycles or event triggers. Reviews every active patient. Cross-references intake data, RAG-retrieved AHA/AACVPR guidelines, and embedding search results. Generates structured SOAP notes as administrative documentation.

**Clinical Assistant Agent (MedGemma-27B)**
Clinician-facing interactive reasoning. Powers the Doctor Chat Interface for targeted queries: What changed in this patient's HRV? Is recovery slower than baseline? Generate a session summary. Transforms each session into a searchable, interpretable clinical record.

### Core Design Principle

> **Deterministic pipelines handle sensing and safety.**
> **Retrieval systems handle memory and evidence.**
> **Agents handle communication and workflow.**

Critical clinical alerts are **always** generated by deterministic rule-based logic. The LLM never fires an alert — it only interprets one after it fires.

---

## 3. Technical Architecture

### System Overview

```
+===========================================================================+
|                          NVIDIA DGX Spark                                 |
|              (128 GB Unified LPDDR5x / GB10 Grace Blackwell)              |
|                                                                           |
|  +----------------+     +------------------+     +--------------------+  |
|  |  Polar H10     | BLE |  Python Signal   | pub |   MQTT Broker      |  |
|  |  Chest Strap   |---->|  Processing      |---->|  (Mosquitto Local) |  |
|  |  ECG  130 Hz   |     |  (FastAPI)       |     | patient/{id}/vitals|  |
|  |  ACC  100 Hz   |     |                  |     | patient/{id}/alerts|  |
|  |  RR intervals  |     |  SQI + Safety    |     +--------+-----------+  |
|  +----------------+     +--------+---------+              |              |
|                                   |                  sub  |  sub         |
|                                   v                   |   |   |          |
|  +-----------------------------+  |    +--------------v---v---v-------+  |
|  |        ChromaDB             |  |    |  Lead Agent Orchestrator     |  |
|  |  patient_vitals_db          |<-+    |  (Deterministic Router)      |  |
|  |  ecg_embedding_tables       |       +-+----------+-----------+-----+  |
|  |  beat_morphology_lookup     |         |          |           |        |
|  |  exercise_vs_stress_lookup  |         v          v           v        |
|  |  rag_medical_literature     |     +-------+  +--------+  +--------+  |
|  |  patient_intake_db          |     | Nurse |  |  Duty  |  |Clinical|  |
|  +-----------------------------+     | Qwen3 |  |Doctor  |  | Asst.  |  |
|  +-----------------------------+     | 32B   |  |Med-    |  |Med-    |  |
|  |  Embedding Search Agent     |     | FP4   |  |Gemma   |  |Gemma   |  |
|  |  CLEF + L1 distance         |     +---+---+  +---+----+  +---+----+  |
|  |  ECG-FM + NormWear          |         |          |            |       |
|  +-----------------------------+         v          v            v       |
|                                    Patient    SOAP Notes    Doctor Chat  |
|                                    Chat UI    Auto-Gen      Interface    |
+===========================================================================+
```

### Memory Allocation on DGX Spark (128 GB Unified LPDDR5x @ 273 GB/s)

```
+=====================================================+
|         DGX SPARK MEMORY ALLOCATION                 |
+=====================================================+
|  Qwen3-32B-FP4 (primary LLM)         ~18 GB        |
|  MedGemma-27B (INT4, Duty Doctor)    ~14 GB        |
|  MedGemma-27B (INT4, Clinical Asst)  ~14 GB        |
|  CLEF ECG foundation model            ~0.2 GB      |
|  ECG-FM foundation model              ~0.4 GB      |
|  NormWear multimodal model            ~0.3 GB      |
|  PubMedBERT embedding model           ~0.4 GB      |
|  ChromaDB indices + vector data       ~4 GB        |
|  vLLM KV cache (two instances)        ~20 GB       |
|  MQTT broker + auxiliary services     ~1 GB        |
|  Signal processing buffers            ~2 GB        |
|  OS + system overhead                 ~8 GB        |
|  ---------------------------------------------------
|  TOTAL ESTIMATED                      ~82 GB       |
|  REMAINING HEADROOM                   ~46 GB       |
+=====================================================+
```

Unified coherent memory means **zero-copy handoff** between CPU signal processing and GPU model inference. The CPU writes structured patient state into the same physical memory the GPU reads for vLLM context. On discrete-GPU systems this requires explicit PCIe transfers, adding latency to every inference call.

### Signal Processing Data Contract

Every MQTT message emits a fully typed, versioned JSON payload. Agents never receive raw waveforms.

```json
{
  "schema_version": "1.0",
  "patient_id": "pt_001",
  "session_id": "S001_20260328",
  "timestamp_iso": "2026-03-28T20:37:00Z",
  "ecg": {
    "hr_bpm": 118,
    "rr_ms_mean": 508,
    "hrv_rmssd_ms": 14.2,
    "hrv_sdnn_ms": 22.1,
    "hrv_pnn50": 3.1,
    "hrv_lf_hf_ratio": 3.8,
    "qrs_width_ms": 94,
    "qt_interval_ms": 378,
    "st_deviation_mv": -0.03,
    "signal_quality_index": 0.87
  },
  "activity": {
    "activity_class": "exercise",
    "dominant_frequency_hz": 1.9,
    "band_energy": 0.61,
    "met_estimate": 4.2
  },
  "safety": {
    "energy_safe": true,
    "alert_level": "none",
    "hr_recovery_1min_bpm": null,
    "prescribed_range_pct": 72
  },
  "embedding": {
    "ecg_embedding_ready": true,
    "embedding_dim": 768,
    "model": "CLEF"
  }
}
```

### vLLM Deployment on DGX Spark

```bash
# Instance 1: Qwen3-32B-FP4 (Nurse Agent)
# NVIDIA pre-validated on DGX Spark — fastest path to live demo
docker run --gpus all \
  -v /models:/models \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:latest \
  --model /models/Qwen3-32B-FP4 \
  --quantization fp4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.18 \
  --enable-prefix-caching \
  --max-num-seqs 8

# Instance 2: MedGemma-27B-IT (Duty Doctor + Clinical Assistant)
# Two agents share one vLLM instance — context routing via system prompt
docker run --gpus all \
  -v /models:/models \
  -p 8001:8001 \
  nvcr.io/nvidia/vllm:latest \
  --model /models/MedGemma-27B-IT \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.22 \
  --port 8001

# Emergency fallback: medgemma-4b (7 GB, always fits)
docker run --gpus all -p 8002:8002 nvcr.io/nvidia/vllm:latest \
  --model /models/MedGemma-4B-IT --gpu-memory-utilization 0.06 --port 8002
```

### Signal Processing Pipeline

```python
import neurokit2 as nk
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch
from dataclasses import dataclass, asdict
import json

@dataclass
class PatientVitals:
    patient_id: str
    session_id: str
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
    prescribed_range_pct: float


class ECGProcessor:
    def __init__(self, fs=130):
        self.fs = fs

    def preprocess(self, raw_ecg: np.ndarray) -> np.ndarray:
        b, a = butter(4, [0.5, 40], btype='bandpass', fs=self.fs)
        filtered = filtfilt(b, a, raw_ecg)
        b_n, a_n = iirnotch(60.0, 30.0, self.fs)
        return filtfilt(b_n, a_n, filtered)

    def detect_r_peaks(self, clean_ecg: np.ndarray) -> np.ndarray:
        # Consensus: Pan-Tompkins + Hamilton (50ms tolerance)
        _, i_pt = nk.ecg_peaks(clean_ecg, sampling_rate=self.fs, method="pantompkins1985")
        _, i_hm = nk.ecg_peaks(clean_ecg, sampling_rate=self.fs, method="hamilton2002")
        tol = int(0.050 * self.fs)
        peaks_pt = set(i_pt["ECG_R_Peaks"])
        peaks_hm = set(i_hm["ECG_R_Peaks"])
        consensus = [p for p in peaks_pt if any(abs(p - h) <= tol for h in peaks_hm)]
        return np.array(sorted(consensus))

    def compute_hrv(self, r_peaks: np.ndarray) -> dict:
        rr = np.diff(r_peaks) / self.fs * 1000
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) < 5:
            return {"rmssd": 0, "sdnn": 0, "pnn50": 0, "lf_hf": 0}
        diff_rr = np.diff(rr)
        rmssd   = np.sqrt(np.mean(diff_rr ** 2))
        sdnn    = np.std(rr, ddof=1)
        pnn50   = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100
        freqs, psd = welch(rr, fs=1000 / np.mean(rr), nperseg=min(256, len(rr)))
        lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)])
        hf = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.40)])
        return {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50, "lf_hf": lf / hf if hf > 0 else 0}

    def compute_sqi(self, clean_ecg, r_peaks, acc_magnitude) -> float:
        if len(r_peaks) < 3:
            return 0.0
        win = int(0.3 * self.fs)
        beats = [clean_ecg[p - win:p + win] for p in r_peaks[1:-1]
                 if p - win >= 0 and p + win < len(clean_ecg)]
        if not beats:
            return 0.0
        template = np.median(beats, axis=0)
        corrs = [np.corrcoef(b, template)[0, 1] for b in beats]
        snr = min(np.var(template) / (np.mean([np.var(b - template) for b in beats]) + 1e-10), 10) / 10
        motion = 1.0 - min(abs(np.corrcoef(clean_ecg[:len(acc_magnitude)], acc_magnitude)[0, 1]), 1.0)
        return float(np.clip(0.4 * np.mean(corrs) + 0.3 * snr + 0.3 * motion, 0.0, 1.0))


class EnergySafeWindow:
    """Deterministic safety check — no LLM involvement."""
    def __init__(self, intake: dict):
        self.hr_max       = 220 - intake["age"]
        self.p_min        = intake["prescribed_intensity_range"][0]
        self.p_max        = intake["prescribed_intensity_range"][1]
        self.risk_factors = intake.get("risk_factors", [])

    def check(self, hr: float, activity: str, sqi: float,
              hr_recovery_1min: float = None) -> tuple[bool, str]:
        if activity == "exercise" and hr > 0.90 * self.hr_max:
            return False, "critical"
        if activity == "exercise" and hr > self.p_max * self.hr_max:
            return False, "warning"
        if activity == "recovery" and hr_recovery_1min is not None and hr_recovery_1min < 12:
            return True, "warning"
        if sqi < 0.5:
            return True, "advisory"
        return True, "none"
```

### MQTT Event Bus

```python
import paho.mqtt.client as mqtt
import json

class VitalsPublisher:
    def __init__(self, broker="localhost", port=1883):
        self.client = mqtt.Client(protocol=mqtt.MQTTv5)
        self.client.connect(broker, port)

    def publish(self, vitals: PatientVitals):
        payload = json.dumps(asdict(vitals), default=str)
        self.client.publish(f"patient/{vitals.patient_id}/vitals", payload, qos=0)
        if vitals.alert_level in ("warning", "critical"):
            self.client.publish(f"patient/{vitals.patient_id}/alerts", payload, qos=2)

# MQTT topic schema
TOPICS = {
    "raw_ecg":       "ttyh/raw/ecg/{patient_id}",
    "raw_acc":       "ttyh/raw/acc/{patient_id}",
    "features_live": "ttyh/features/live/{patient_id}",
    "features_5s":   "ttyh/features/window_5s/{patient_id}",
    "features_30s":  "ttyh/features/window_30s/{patient_id}",
    "alerts":        "ttyh/alerts/{patient_id}",
    "soap":          "ttyh/reports/soap/{patient_id}",
    "session":       "ttyh/reports/session/{patient_id}",
}
```

### ChromaDB Collections

```python
import chromadb
from chromadb.config import Settings

chroma = chromadb.PersistentClient(
    path="/data/chromadb",
    settings=Settings(anonymized_telemetry=False))

COLLECTIONS = {
    "patient_vitals":    "Rolling window of structured vitals per patient",
    "ecg_embeddings":    "CLEF 768-dim ECG morphology embeddings per session",
    "beat_morphology":   "Reference ECG templates for L1 distance scoring",
    "exercise_stress":   "Reference profiles: exercise vs pathologic stress patterns",
    "rag_literature":    "AHA/AACVPR 2024 guidelines, HRV papers (PubMedBERT)",
    "patient_intake":    "Demographics, conditions, risk factors, prescriptions",
}

for name, desc in COLLECTIONS.items():
    chroma.get_or_create_collection(name=name, metadata={"description": desc})
```

### Embedding Search Agent (Non-LLM Retrieval)

```python
import numpy as np

class EmbeddingSearchAgent:
    def search(self, ecg_embedding: list, patient_id: str) -> dict:
        morph = chroma.get_collection("beat_morphology").query(
            query_embeddings=[ecg_embedding], n_results=5,
            include=["distances", "metadatas"])
        stress = chroma.get_collection("exercise_stress").query(
            query_embeddings=[ecg_embedding], n_results=3,
            include=["distances", "metadatas"])
        history = chroma.get_collection("ecg_embeddings").query(
            query_embeddings=[ecg_embedding], n_results=5,
            where={"patient_id": patient_id},
            include=["distances", "metadatas"])
        return {
            "morphology_match":       morph["metadatas"][0][0].get("label"),
            "morphology_distance":    morph["distances"][0][0],
            "stress_classification":  stress["metadatas"][0],
            "historical_similarity":  float(np.mean(history["distances"][0])),
            "consistent_with_history": np.mean(history["distances"][0]) < 0.5
        }
```

### Lead Agent Orchestrator

```python
import asyncio

EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "dizzy",
    "passing out", "faint", "heart racing", "nauseous", "blacking out", "tight chest"
]
DIAGNOSTIC_TERMS = [
    "diagnose", "diagnosis", "abnormal", "disease", "you have",
    "arrhythmia", "fibrillation", "prescribe", "take medication",
    "stop taking", "condition indicates"
]

NURSE_PROMPT = """You are the Talk to Your Heart Wellness Companion — warm, supportive,
helping patients during cardiac rehabilitation.

CRITICAL RULES:
1. WELLNESS companion only — NOT a medical provider.
2. NEVER use diagnostic language: no 'abnormal', 'disease', 'arrhythmia', 'you have'.
3. Frame ALL observations as trends: YES: 'Your heart rate dropped 18 bpm after exercise'.
   NO: 'Your cardiac recovery is abnormal'.
4. NEVER recommend medication changes.
5. ALWAYS refer medical concerns to the care team.
6. If patient mentions chest pain, dizziness, or shortness of breath — respond ONLY with:
   'Please stop exercising immediately and alert the nearest staff member.'
Temperature: 0.5"""

DUTY_DOCTOR_PROMPT = """You are the Talk to Your Heart Clinical Review Agent — generating
structured SOAP notes as administrative documentation for cardiac rehabilitation sessions.

Context provided: current vitals JSON, patient intake, session history, ECG embedding
results, RAG-retrieved AHA/AACVPR guidelines, active alerts with trigger data.

RULES:
1. SOAP structure: Subjective / Objective / Assessment / Plan.
2. Include SQI score in Objective. Flag metrics where SQI < 0.7 as [low confidence].
3. Reference patient intake (conditions, risk factors) in Assessment.
4. Cite RAG guidelines when making exercise response comparisons.
5. Assessment and Plan are suggestions for clinician review — not clinical directives.
6. Note deviations from prior session trends.
Temperature: 0.3"""

CLINICAL_ASSISTANT_PROMPT = """You are the Talk to Your Heart Clinical Assistant —
interactive AI for clinicians during and after cardiac rehabilitation sessions.

Answer queries using full patient context: current vitals, intake, session history,
embedding comparisons, retrieved guidelines.

RULES:
1. Be specific and data-grounded. Cite actual values.
2. Compare sessions by date, HR values, recovery metrics.
3. Flag confidence based on SQI scores.
4. Use clinical terminology — audience is healthcare professionals.
Temperature: 0.3"""


class LeadOrchestrator:
    def __init__(self, nurse_url, doctor_url, chroma_client):
        self.nurse_url  = nurse_url    # Qwen3-32B-FP4
        self.doctor_url = doctor_url   # MedGemma-27B
        self.chroma     = chroma_client
        self.patient_states = {}
        self.embedding_agent = EmbeddingSearchAgent()

    async def assemble_context(self, patient_id: str) -> dict:
        vitals   = self.patient_states.get(patient_id, {})
        intake   = self.chroma.get_collection("patient_intake").get(
                       where={"patient_id": patient_id})
        history  = self.chroma.get_collection("patient_vitals").query(
                       query_texts=[f"patient {patient_id}"],
                       where={"patient_id": patient_id}, n_results=5)
        embed    = self.embedding_agent.search(
                       vitals.get("ecg_embedding", []), patient_id)
        rag      = self.chroma.get_collection("rag_literature").query(
                       query_texts=[f"cardiac rehab HR {vitals.get('hr_bpm')} "
                                    f"recovery HRV exercise"],
                       n_results=5)
        return {
            "current_vitals":     vitals,
            "patient_intake":     intake,
            "session_history":    history,
            "embedding_search":   embed,
            "rag_guidelines":     rag,
            "active_alert":       vitals.get("alert_level", "none"),
            "sqi":                vitals.get("sqi", 0),
        }

    async def route(self, event: str, patient_id: str, data: dict) -> dict:
        ctx = await self.assemble_context(patient_id)

        if event == "patient_chat":
            msg = data.get("message", "")
            # Layer 1: Emergency keyword — hardcoded, no LLM
            if any(k in msg.lower() for k in EMERGENCY_KEYWORDS):
                return {"response": "Please stop exercising immediately and alert "
                                    "the nearest staff member. Your safety is the priority.",
                        "alert": "critical", "source": "hardcoded"}
            response = await self.call_agent(self.nurse_url, NURSE_PROMPT, ctx, msg)
            # Layer 2: Output validator — blocks diagnostic language
            if any(t in response.lower() for t in DIAGNOSTIC_TERMS):
                return {"response": "I'm here to support your rehab session. "
                                    "For medical questions, please speak with your care team.",
                        "source": "validator_fallback"}
            return {"response": response, "source": "nurse_agent"}

        elif event in ("duty_review", "alert"):
            return await self.call_agent(
                self.doctor_url, DUTY_DOCTOR_PROMPT, ctx, "Generate SOAP note review")

        elif event == "clinician_chat":
            return await self.call_agent(
                self.doctor_url, CLINICAL_ASSISTANT_PROMPT, ctx, data["message"])

    async def scheduled_duty_review(self):
        while True:
            for pid in list(self.patient_states.keys()):
                await self.route("duty_review", pid, {})
            await asyncio.sleep(900)  # 15-minute cycle
```

### FastAPI Backend

```python
from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI(title="Talk to Your Heart — DGX Spark Backend", version="1.0")

@app.websocket("/ws/ecg/{patient_id}")
async def ecg_stream(websocket: WebSocket, patient_id: str):
    await websocket.accept()
    processor  = ECGProcessor(fs=130)
    acc_proc   = AccelerometerProcessor(fs=100)
    publisher  = VitalsPublisher()
    intake     = get_patient_intake(patient_id)
    safety     = EnergySafeWindow(intake)
    while True:
        data     = await websocket.receive_json()
        ecg_raw  = np.array(data["ecg"])
        acc_xyz  = np.array(data["acc"]).reshape(-1, 3)
        clean    = processor.preprocess(ecg_raw)
        r_peaks  = processor.detect_r_peaks(clean)
        hrv      = processor.compute_hrv(r_peaks)
        acc_feat = acc_proc.extract_features(acc_xyz)
        hr       = 60000 / np.mean(np.diff(r_peaks) / 130 * 1000) if len(r_peaks) > 1 else 0
        sqi      = processor.compute_sqi(clean, r_peaks, acc_feat["magnitude"])
        activity = acc_proc.classify_activity(acc_feat["dominant_freq"], acc_feat["band_energy"], hr)
        safe, alert = safety.check(hr, activity, sqi)
        vitals = PatientVitals(patient_id=patient_id, session_id=data["session_id"],
                               timestamp=data["timestamp"], hr_bpm=hr,
                               hrv_rmssd_ms=hrv["rmssd"], hrv_sdnn_ms=hrv["sdnn"],
                               hrv_pnn50=hrv["pnn50"], hrv_lf_hf_ratio=hrv["lf_hf"],
                               sqi=sqi, qrs_width_ms=0, qt_interval_ms=0, st_deviation_mv=0,
                               activity_class=activity, met_estimate=hr / 40,
                               dominant_freq_hz=acc_feat["dominant_freq"],
                               band_energy=acc_feat["band_energy"],
                               ecg_embedding=[], energy_safe=safe, alert_level=alert,
                               prescribed_range_pct=hr / safety.hr_max * 100)
        publisher.publish(vitals)
        await websocket.send_json(asdict(vitals))

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

@app.get("/api/session/{patient_id}/report")
async def session_report(patient_id: str):
    """Generate PDF-ready session summary for CPT 93798 billing documentation."""
    return await orchestrator.generate_session_report(patient_id)
```

### Flutter BLE Integration

```dart
import 'package:polar/polar.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';

class PolarH10Service {
  final polar = Polar();
  late WebSocketChannel _ws;
  final String patientId;
  final String sessionId;

  PolarH10Service({required this.patientId, required this.sessionId}) {
    _ws = WebSocketChannel.connect(
        Uri.parse('ws://dgx-spark-ip:8000/ws/ecg/$patientId'));
  }

  Future<void> startStreaming(String deviceId) async {
    await polar.connectToDevice(deviceId);

    polar.startEcgStreaming(deviceId).listen((data) {
      _ws.sink.add(jsonEncode({
        'ecg': data.samples.map((s) => s.voltage).toList(),
        'session_id': sessionId,
        'timestamp': DateTime.now().millisecondsSinceEpoch / 1000,
      }));
    });

    polar.startAccStreaming(deviceId).listen((data) {
      _ws.sink.add(jsonEncode({
        'acc': data.samples.map((s) => [s.x, s.y, s.z]).toList(),
        'session_id': sessionId,
        'timestamp': DateTime.now().millisecondsSinceEpoch / 1000,
      }));
    });

    // Receive processed vitals for UI update
    _ws.stream.listen((msg) {
      final vitals = jsonDecode(msg);
      _updateDashboard(vitals);
    });
  }
}
```

### Foundation Model Layer

| Model | Source | Size | Use |
|-------|--------|------|-----|
| **CLEF** | Nokia Bell Labs, Dec 2025 | ~200 MB | Single-lead ECG embedding, clinically-guided contrastive learning |
| **ECG-FM** | U. Toronto, JAMIA Open Oct 2025 | ~400 MB | 90.9M params, wav2vec 2.0, 1.5M ECGs, 0.996 AUROC AF detection |
| **NormWear** | Dec 2024, Apache 2.0 | ~300 MB | Multimodal wearable (ECG+ACC+PPG), zero-shot inference |
| **PubMedBERT** | Microsoft BioNLP | ~400 MB | Medical RAG embeddings for AHA/AACVPR guidelines |

**Single-Lead Strategy:** CLEF eliminates the single-lead generalization gap via clinically-guided contrastive learning (≥2.6% AUROC improvement over self-supervised baselines on lead-I). Multimodal augmentation from NormWear (accelerometer + HRV context) compensates for spatial information absent without 12 leads.

---

## 4. Why NVIDIA DGX Spark

DGX Spark is the **only commercially available device** that makes this system possible. Every design decision traces to a specific capability:

### The Core Constraint: Simultaneous Multi-Model Serving

| Requirement | Details |
|-------------|---------|
| Primary LLM | Qwen3-32B-FP4 (~18 GB) |
| Medical LLM | MedGemma-27B × 2 roles (~28 GB) |
| ECG foundation models | CLEF + ECG-FM + NormWear (~0.9 GB) |
| RAG embedding model | PubMedBERT (~0.4 GB) |
| KV cache (two vLLM) | ~20 GB |
| ChromaDB + services | ~5 GB |
| Signal buffers + OS | ~10 GB |
| **Total** | **~82 GB** |

### Why No Alternative Works

| Alternative | Why It Fails |
|-------------|-------------|
| **Cloud (AWS/GCP/Azure)** | PHI in transit violates HIPAA architecture; 100–500ms network latency per inference; ongoing cost dependency |
| **RTX 4090 (24 GB VRAM)** | Cannot fit the primary 32B LLM in a single GPU |
| **A6000 (48 GB VRAM)** | Fits one LLM but not medical model + foundation models + KV cache simultaneously |
| **Apple M4 Ultra (192 GB)** | 128 GB unified memory but ~20× less GPU compute than Blackwell — multi-agent serving degrades to 4–8 tok/s |
| **DGX H100 (80 GB/GPU)** | Would work but is a rack-mounted data center device — not deployable in a clinic room |

### DGX Spark's Unique Value

- **128 GB unified LPDDR5x @ 273 GB/s** — zero-copy CPU→GPU handoff
- **1 PFLOP FP4** (Blackwell GB10) — multi-agent serving at clinical latency
- **Desktop form factor** — deployable on-premises in clinic without data center infrastructure
- **NVIDIA inference stack** — native vLLM support, NIM containers, pre-validated models
- **HIPAA by architecture** — data never leaves the building, no BAAs required for cloud AI

---

## 5. Innovation — Five First-in-Class Capabilities

**1. Foundation-Model ECG Analysis on Wearable Single-Lead Signal**
CLEF (Nokia Bell Labs, December 2025) uses clinically-guided contrastive learning to eliminate the single-lead performance gap. Deploying CLEF + NormWear + ECG-FM in a production clinical pipeline is unprecedented — no cardiac rehab system has used ECG foundation models for real-time in-session monitoring.

**2. Multi-Agent Clinical AI with Role Separation on Edge Hardware**
Three specialized agents — patient communication (Qwen3), clinical oversight (MedGemma-27B), and clinician reasoning (MedGemma-27B) — running concurrently on a single on-premise device. Role boundaries enforced by system prompt engineering and output validation. No clinical AI system has deployed multi-agent role separation at the edge.

**3. Embedding-Based Exercise-vs-Stress Discrimination**
The Embedding Search Agent performs L1 distance comparison against reference profiles, transforming alerting from threshold-crossing (HR > X) to pattern-aware clinical discrimination (this ECG morphology cluster is consistent with physiologic exercise stress, not pathologic ischemia). This is not threshold alerting with renamed variables — it is a fundamentally different approach to cardiac safety monitoring.

**4. Signal-Quality-Aware AI Interpretation**
SQI scores (0.0–1.0) travel with every data point through the entire pipeline. Agents explicitly know whether an HRV drop occurred during clean signal (SQI 0.92) or motion artifact (SQI 0.31). SOAP notes flag low-confidence measurements. No competitor offers confidence-weighted interpretation at this granularity.

**5. Automated SOAP Notes Grounded in Structured Physiologic Features + Retrieved Evidence**
Generated from deterministic pipeline output + RAG guidelines, not clinician speech transcription or free-form recall. The structured data contract (typed JSON payload) makes the notes auditable and reproducible — every line in the SOAP note traces to a specific sensor value and a specific guideline citation.

---

## 6. Scalability Design

The demo runs on a single DGX Spark serving one clinic. The architecture is designed for scale-out from day one.

### Path to Multi-Clinic Deployment

```
+=========================================================+
|              SCALING ARCHITECTURE                       |
+=========================================================+
|  SINGLE CLINIC (Demo/MVP)                               |
|  1× DGX Spark                                          |
|  6-10 concurrent patients                              |
|  On-premise MQTT + ChromaDB                            |
|  Single vLLM instance per model                        |
+---------------------------------------------------------+
|  MULTI-CLINIC (Phase 2, 6 months)                      |
|  1× DGX Spark per clinic                               |
|  Federated ChromaDB with clinic-local sharding         |
|  Central aggregate analytics (de-identified)            |
|  Clinic-to-clinic model weight sharing (no PHI)        |
+---------------------------------------------------------+
|  HEALTH SYSTEM (Phase 3, 12-18 months)                 |
|  DGX Spark cluster behind hospital network             |
|  NVIDIA NIM microservices for model versioning         |
|  FHIR R4 export for EHR integration                   |
|  Federated learning: improve models without pooling PHI|
+---------------------------------------------------------+
|  HYBRID CLOUD (Optional — non-HIPAA analytics only)    |
|  De-identified population analytics → cloud            |
|  Per-clinic raw PHI stays on DGX Spark                 |
|  Model updates from cloud → pushed to edge devices     |
+=========================================================+
```

### Vertical Scaling (More Patients per Clinic)

- **Memory headroom (46 GB)** absorbs additional KV cache for higher concurrency
- **vLLM prefix caching** reuses shared system prompt tokens across patients (~40% KV cache reduction)
- **Duty Doctor agent** batches multi-patient reviews — single LLM call reviews all 10 patients in one context window
- **Embedding Search Agent** is CPU-bound, not GPU-bound — scales independently

### Horizontal Scaling (More Clinics)

- Each DGX Spark is fully self-contained — no inter-device dependencies in the critical path
- ChromaDB can be configured for cross-clinic read replicas for population analytics
- MQTT topic namespacing (`clinic/{id}/patient/{id}/...`) isolates data by facility

---

## 7. Ecosystem Thinking — Interoperability and Extensibility

### EHR Integration Layer (FHIR R4)

```python
# FHIR R4 export — session data → Epic/Cerner
from fhirclient.models import observation, patient

def export_session_to_fhir(patient_vitals: PatientVitals,
                            soap_note: str) -> dict:
    """
    Maps structured session data to FHIR R4 Observation resources.
    Enables direct import into Epic/Cerner via SMART on FHIR.
    """
    obs = observation.Observation()
    obs.status = "final"
    obs.code = {
        "coding": [{"system": "http://loinc.org", "code": "8867-4",
                    "display": "Heart rate"}]
    }
    obs.valueQuantity = {"value": patient_vitals.hr_bpm, "unit": "beats/min"}
    obs.effectiveDateTime = patient_vitals.timestamp
    return obs.as_json()
```

### Open Extension Points

| Extension Point | Interface | Purpose |
|-----------------|-----------|---------|
| **Additional wearables** | WebSocket JSON contract | Plug in Garmin, Apple Watch, Withings via unified data contract |
| **Custom alert rules** | YAML rule DSL loaded at startup | Clinical staff can configure institution-specific thresholds |
| **Model swap** | vLLM OpenAI-compatible `/v1/chat` | Replace Qwen3/MedGemma with any OpenAI-compatible model |
| **RAG document ingestion** | REST `POST /api/rag/ingest` | Add institution-specific protocols without code changes |
| **Webhook alerts** | `POST /api/webhooks/register` | Push alerts to pager systems, Slack, nurse call systems |
| **PDF report export** | `GET /api/session/{id}/report.pdf` | Direct output for CPT 93798 billing documentation |

### API Surface (Production-Ready)

```
POST   /api/patients/{id}/intake         Register patient
GET    /api/patients/{id}/sessions       List sessions
POST   /api/chat/patient/{id}            Patient chat (Nurse Agent)
POST   /api/chat/clinician/{id}          Clinician query (Clinical Assistant)
GET    /api/session/{id}/soap            SOAP note
GET    /api/session/{id}/report.pdf      PDF session report
GET    /api/dashboard/active             Live multi-patient state
GET    /api/fhir/R4/Observation/{id}     FHIR export
POST   /api/rag/ingest                   Add documents to RAG
POST   /api/webhooks/register            Register alert webhooks
GET    /api/health                       System health + memory usage
```

---

## 8. User Impact

### Primary Beneficiaries

| User | Current State | With Talk to Your Heart |
|------|---------------|------------------------|
| **Patient (post-MI/CABG)** | No feedback during session, confusing discharge papers | Real-time encouragement calibrated to their actual effort, understandable session summaries |
| **Supervising RN/EP** | Manually watching 6–10 screens, 10–15 min documentation/patient | Alert intelligence surfaces what needs attention; automated SOAP notes; 2-min review instead of 15 |
| **Cardiologist** | Reviews hand-written notes days later | Same-session longitudinal data, trend comparison, ECG embedding history |
| **Health System** | Compliance burden, poor CR completion rates | Improved monitoring quality, structured documentation for billing, reduced liability |

### Quantified Clinical Impact

- **Documentation time saved:** 10–15 min → ~2 min per patient per session
- **6 patients per session:** 48–78 minutes of clinician time recovered per session
- **CR sessions per week per clinic:** ~20 sessions → 16–26 clinician-hours/week recovered
- **Patient safety:** Deterministic alerting catches HR threshold violations in <1s (vs. human detection latency of 30–180s in a busy room)
- **SOAP note quality:** Grounded in structured sensor data + RAG guidelines, not memory
- **Longitudinal tracking:** First system to offer ECG morphology comparison across sessions via embedding similarity

### Population-Scale Impact

If Talk to Your Heart improves CR completion by even 5 percentage points (27% → 32%):
- Based on 366,000 studied Medicare beneficiaries
- ~18,300 additional completers
- At 1.8% lower 1-year readmission per session × 36 sessions: meaningful readmission reduction
- At $1,005 lower annual spend per CR participant: ~$18.4M in Medicare savings in that cohort

---

## 9. Market Awareness

### Market Size

| Segment | 2024 | 2030 | CAGR |
|---------|------|------|------|
| U.S. Cardiac Rehab | $984M | $1.39B | 5.9% |
| AI-Driven Cardiac Platforms | — | $3.66B | 21.1% |
| Wearable Cardiac Devices | $3.87B | $25.97B | 23.8% |

### Competitive Landscape

```
+================================================================================+
|              Continuous  Foundation  Multi-Agent  Auto-    On-Premise          |
| Company       ECG        Model ECG   Clinical AI  SOAP     Edge AI    Position |
+================================================================================+
| Fourth Frontier   YES     no          no           no       no        consumer |
| Recora            no      no          no           no       no        home CR  |
| Movn Health       no      no          no           no       no        home CR  |
| Biofourmis        partial proprietary single        no       no        enterprise|
| Carda Health      no      no          no           no       no        home CR  |
| BioTelemetry      partial no          no           no       no        hospital |
+--------------------------------------------------------------------------------+
| Talk to Your Heart YES   CLEF+3       3 agents     YES      DGX Spark  in-clinic|
+================================================================================+
```

### Strategic Positioning

**Biofourmis** is the closest competitor in enterprise clinical AI — but runs cloud-based, uses proprietary models, and lacks in-clinic supervised session support. Their moat is EHR integration breadth; ours is clinical-session intelligence depth and on-premise HIPAA architecture.

**No competitor combines:** continuous ECG + foundation-model embedding + multi-agent clinical AI + automated SOAP notes + on-premise edge deployment. The integration complexity is the moat.

### Go-to-Market

1. **Direct to CR programs** (3–5 pilot sites): University health systems with existing DGX Spark or DGX infrastructure commitments are the ideal first customers — hardware is already procurement-approved.
2. **Through NVIDIA healthcare partners:** NVIDIA's healthcare ecosystem (Epic, Cerner, Philips) provides distribution channels without a direct sales force.
3. **CPT/RPM billing creates self-funding:** Each 100-patient clinic generates ~$118,800/year in RPM revenue under current codes — the system pays for itself.

---

## 10. Feasibility and 24-Hour Execution Plan

### MVP Scope vs. Full Scope

| Component | MVP (Demo) | Full Scope |
|-----------|-----------|------------|
| BLE acquisition | Polar H10 → Python mock OR real device | Real H10 + multi-device |
| Signal processing | NeuroKit2 pipeline, SQI, HRV | + ST analysis, foundation model embeddings |
| MQTT | Mosquitto local | + QoS 2 alert delivery |
| ChromaDB | RAG + vitals | + ECG embeddings + exercise/stress lookup |
| vLLM | Qwen3-32B-FP4 (single instance) | + MedGemma-27B dual-role |
| Agents | Nurse + Clinical Assistant | + Duty Doctor 15-min cycle |
| Flutter UI | Simulated patient chat + clinician dashboard | + PDF reports + BLE native |
| Safety | Emergency keyword hardcoded response | + output validator + 4-layer |
| Foundation models | CLEF embedding generation | + ECG-FM + NormWear |

### 24-Hour Build Timeline

```
+=========================================================================+
|  HOUR   TASK                                              OWNER         |
+=========================================================================+
|  0-2    DGX Spark: Ubuntu, Mosquitto MQTT, ChromaDB      Viggi         |
|         vLLM + Qwen3-32B-FP4 model download + test       Viggi         |
+-------------------------------------------------------------------------+
|  2-6    FastAPI server skeleton + WebSocket endpoint     Sansrit        |
|         Polar H10 BLE (Python polar_python client)       Sansrit        |
|         Flutter: BLE page + WebSocket sender             Sansrit        |
+-------------------------------------------------------------------------+
|  6-10   ECG preprocessing (NeuroKit2 pipeline)           Shiva          |
|         HRV computation (RMSSD, SDNN, LF/HF)            Shiva          |
|         SQI scoring (template match + SNR + motion)      Shiva          |
+-------------------------------------------------------------------------+
|  10-14  EnergySafeWindow deterministic check             Rumon          |
|         MQTT publisher + alert QoS 2                     Rumon          |
|         Alert dashboard (Flutter flash on WARNING)        Rumon          |
+-------------------------------------------------------------------------+
|  14-18  ChromaDB: RAG ingestion (AHA guidelines)         Viggi          |
|         Nurse Agent: system prompt + endpoint            Viggi          |
|         Patient chat UI (Flutter)                        Sansrit        |
+-------------------------------------------------------------------------+
|  18-22  Clinical Assistant Agent (Doctor Chat)           Shiva          |
|         Clinician multi-patient dashboard (Flutter)      Rumon          |
|         Lead Orchestrator: routing + context assembly    Viggi          |
+-------------------------------------------------------------------------+
|  22-26  CLEF embedding integration (if time allows)      Shiva          |
|         SOAP note generation + display                   Viggi          |
|         MedGemma-27B second vLLM instance                Rumon          |
+-------------------------------------------------------------------------+
|  26-30  Multi-patient simulation (3 patients, ecg_sim)  All            |
|         Adversarial safety testing (emergency keywords)  Rumon          |
|         Demo rehearsal: Maria + James + Ruth scenario    All            |
+-------------------------------------------------------------------------+
|  30-36  Polish: latency optimization, UI cleanup          Sansrit        |
|         PDF session report generation                    Viggi          |
|         Backup: fallback to MedGemma-4B if 27B issues   Shiva          |
+-------------------------------------------------------------------------+
|  36-48  Buffer + final integration testing               All            |
|         Presentation preparation                         All            |
+=========================================================================+
```

### Critical Path

The single critical path item is **DGX Spark vLLM serving** (Hours 0–2). Everything else builds on top of it. If vLLM is live by Hour 2, the project is on track.

### Fallback Models (Always Available)

| Primary | If Fails | Fallback |
|---------|----------|----------|
| Qwen3-32B-FP4 | Load error / OOM | nvidia/Qwen3-32B-FP4 → Qwen3-14B-FP4 |
| MedGemma-27B | Too slow for demo | MedGemma-4B-IT (4 GB, always fits) |
| CLEF embeddings | Not ready in 24h | Random unit vector (demo still works) |
| Flutter BLE | Hardware issue | NeuroKit2 ecg_simulate() over mock WebSocket |

---

## 11. Risk Assessment

### Risk Matrix

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|------------|-------------|
| **vLLM OOM — models don't fit** | Medium | Critical | Budget validated at ~82 GB / 128 GB with 46 GB headroom; use FP4 not FP8 | Drop MedGemma-27B → MedGemma-4B (saves 28 GB) |
| **BLE pairing failure (Polar H10)** | Medium | High | Test hardware before hackathon; have device ID pre-registered | NeuroKit2 ecg_simulate() → mock WebSocket for full demo |
| **Inference latency too slow for demo** | Medium | High | Qwen3-32B-FP4 is pre-validated on DGX Spark at 15–20 tok/s; vLLM prefix caching | Pre-generate 2–3 SOAP notes offline; stream tokens live |
| **MedGemma-27B model access/download** | Low | High | Pre-download before hackathon begins; mirror to local storage | MedGemma-4B as drop-in replacement |
| **Memory pressure under concurrent load** | Low | Medium | 15-min Duty Doctor cycle batches patients; Nurse Agent is patient-sequential | Reduce KV cache utilization flag in vLLM |
| **SQI false positives flagging clean signal** | Low | Medium | Validated parameters on synthetic ECG pre-hackathon | Raise SQI threshold from 0.5 → 0.4 |
| **Flutter WebSocket stability** | Low | Low | Reconnection logic built in; local WiFi only | Direct REST polling as fallback |
| **CLEF not loading in 24h** | Medium | Low | Demo still works with random embeddings; embedding search degrades gracefully | Skip CLEF demo, emphasize LLM agents |

### Non-Negotiable Safety Properties

These are verified independently of LLM availability:

1. Emergency keywords → hardcoded response without LLM invocation (testable with unit test)
2. Energy Safe Window fires before MQTT publish (deterministic, synchronous)
3. Dashboard alert flash on WARNING/CRITICAL does not depend on agent response

---

## 12. Differentiation Strategy

### The Integration Moat

The individual components (RAG, ECG foundation models, multi-agent LLMs, MQTT) are not novel in isolation. **The moat is the integration:**

```
+=========================================================+
|              WHY THIS IS HARD TO REPLICATE              |
+=========================================================+
| 1. Clinical context design (not just an LLM wrapper)    |
|    - 4-layer safety architecture                        |
|    - Role-specific agent personas with guardrails       |
|    - Wellness-vs-diagnostic language boundary           |
|                                                         |
| 2. Signal quality as a first-class citizen              |
|    - SQI travels with every data point                  |
|    - Agents know confidence level of their inputs       |
|    - Not an afterthought                                |
|                                                         |
| 3. On-premise architecture designed from day one        |
|    - Not a cloud system with a privacy toggle           |
|    - HIPAA compliance through architecture, not BAAs    |
|    - Deployable in a clinic without IT infrastructure   |
|                                                         |
| 4. Structured data contract between signal and agent    |
|    - Typed JSON payload, not raw waveform text          |
|    - Auditable, reproducible SOAP notes                 |
|    - Every agent output traces to a sensor value        |
|                                                         |
| 5. Regulatory-first framing from prototype              |
|    - General wellness now (no 510(k) required)          |
|    - 510(k) path identified with specific predicates    |
|    - Not retrofitting compliance onto a demo            |
+=========================================================+
```

### Sustainable Differentiation

- **Data advantage:** Multi-session ECG embeddings per patient create a longitudinal morphology record that improves embedding search accuracy over time — a data flywheel competitors cannot replicate without longitudinal CR data
- **DGX Spark positioning:** NVIDIA's push into clinical edge AI aligns hardware roadmap with this system's requirements — future Grace Blackwell generations will expand the memory/compute envelope, not constrain it
- **Regulatory moat:** 510(k) clearance (Phase 2) with established predicates (Hexoskin, CardioTag, Apple Watch ECG) creates a 6–18 month barrier for undifferentiated competitors

---

## 13. Features Summary

- On-campus cardiac rehab monitoring — zero patient data leaves the facility
- HIPAA compliance through architectural data locality, not cloud BAAs
- Continuous single-lead ECG at 130 Hz from Polar H10
- Pan-Tompkins + Hamilton consensus QRS detection
- Beat-to-beat HRV: RMSSD, SDNN, pNN50, LF/HF ratio over configurable windows
- ECG morphology: QRS width, QT interval, ST deviation per beat
- SQI (0.0–1.0): template matching + SNR estimation + accelerometer motion correlation
- Activity classification (rest/warmup/exercise/cooldown/recovery) from FFT dominant frequency + band energy
- MET estimation from HR + accelerometer for exercise prescription validation
- Energy Safe Window: deterministic safety check before every MQTT publish
- ECG foundation models (CLEF, NormWear, ECG-FM) for morphology embedding
- Embedding Search Agent: L1 distance against morphology + exercise-vs-stress profiles
- ChromaDB: 6 collections (vitals, ECG embeddings, morphology, exercise/stress, RAG, intake)
- RAG grounded in AHA/AACVPR 2024 guidelines — PubMedBERT embeddings, top-5 retrieval
- MQTT (Mosquitto) with per-patient topic isolation, QoS 2 for safety alerts
- Qwen3-32B-FP4 Nurse Agent: patient communication with wellness guardrails
- MedGemma-27B Duty Doctor Agent: 15-minute review cycles, SOAP notes
- MedGemma-27B Clinical Assistant Agent: Doctor Chat Interface
- Lead Orchestrator: deterministic routing, 8-source context assembly
- 4-layer safety: deterministic alerts → emergency keyword classifier → output validator → wellness system prompt
- Automated SOAP notes as administrative documentation outside SaMD scope
- Multi-session longitudinal tracking: HR recovery trends, morphology comparison
- Concurrent multi-model serving in 128 GB unified memory
- Flutter dashboards: patient chat + multi-patient clinician view + Doctor Chat + PDF reports
- FHIR R4 export for Epic/Cerner EHR integration
- Webhook API for integration with pager systems and nurse call systems
- Two-phase regulatory: FDA general wellness now, 510(k) for clinical claims
- CPT 93798 in-session + RPM 99454/99457/99458 at $99+/month per patient

---

## 14. User Experience

### Patient-Facing Flow

The patient wears a Polar H10 and interacts with a tablet chat interface.

```
WARM-UP:  "Good morning, Maria. Your session is underway and everything looks steady."
EXERCISE: "You've been at it for 12 minutes and you're right in your target zone."
EFFORT+:  "Your heart rate has climbed a bit higher than usual — you might ease back slightly."
RECOVERY: "Nice cooldown — your heart rate is settling toward your resting level."
END:      "Great session today, Maria. You completed your target and your heart settled
           down steadily after exercise. See you next session."
```

The patient never sees ECG waveforms, HRV numbers, or clinical terminology.

### Clinician Dashboard

```
+==========================================================================+
|              CLINICIAN MULTI-PATIENT DASHBOARD — DGX Spark               |
|   +-------------------+  +-------------------+  +-------------------+    |
|   | Maria Santos       |  | James Park         |  | Ruth Williams      |  |
|   | HR: 118  EXERCISE  |  | HR: 132  EXERCISE  |  | HR: 88  RECOVERY   |  |
|   | MET: 4.1 SQI: 0.92|  | MET: 5.3 SQI: 0.85|  | MET: 1.8 SQI: 0.94|  |
|   | Rx: 60–80% HRmax  |  | [! WARNING: HR HIGH]|  | HRR: 22 bpm/min    |  |
|   | [SAFE]             |  | 91% of HRmax       |  | [SAFE]             |  |
|   +-------------------+  +-------------------+  +-------------------+    |
|                                                                           |
|   [ DOCTOR CHAT ]  "How is James's recovery vs last session?"            |
|   > Last session (03-21): HRR at 1 min = 18 bpm. Today (03-28): 14 bpm. |
|     Slower by 4 bpm. SQI today: 0.85 (adequate confidence). Intake note: |
|     T2DM — autonomic neuropathy may contribute to blunted recovery.      |
|     AHA CR guidelines suggest monitoring HRR trend over 3–5 sessions.    |
+==========================================================================+
```

---

## 15. Demo Scenario

Three concurrent patients demonstrate every major system capability:

**Patient A — Maria Santos, 62, post-MI, no complications**
Normal session. Nurse Agent provides warm encouragement calibrated to her 72% HRmax effort. Clean SOAP note generated at end. Demonstrates happy-path clinical documentation.

**Patient B — James Park, 55, post-CABG, T2DM**
HR approaches prescribed maximum during exercise. Energy Safe Window triggers WARNING. Dashboard highlights James in amber. Duty Doctor review incorporates intake (T2DM), RAG-retrieved exercise prescription guidance, and 4-session HR recovery trend. Clinician asks Doctor Chat: "Is James's recovery blunted vs baseline?" — Clinical Assistant retrieves history and delivers data-grounded comparison. Demonstrates alert intelligence + risk-aware reasoning + historical context.

**Patient C — Ruth Williams, 70, HFrEF, sensor movement during exercise**
Motion artifact → SQI drops to 0.31 → ADVISORY alert → Nurse Agent tells Ruth: "Let me check on something — there may be a sensor shift." Staff adjusts strap → SQI recovers to 0.88. Subsequent SOAP note flags the gap: "[low confidence - SQI 0.31 during minutes 8–11]." Demonstrates signal-quality-aware interpretation — the system knows what it doesn't know.

---

## 16. Regulatory and Reimbursement

### Regulatory Pathway

**Phase 1 (Now): FDA General Wellness — no 510(k) required**
FDA guidance (updated January 6, 2026) covers non-invasive physiologic sensors estimating HRV and recovery metrics.

| Allowed Language | Prohibited Language |
|-----------------|-------------------|
| "Your HR dropped 18 bpm in the first minute after exercise" | "Your cardiac recovery is abnormal" |
| "Your session effort was in your target zone" | "This suggests reduced autonomic function" |
| "HR trend over 5 sessions shows improvement" | "You have a condition that..." |

SOAP notes framed as administrative documentation — outside Software as a Medical Device scope.

**Phase 2 (12–18 months): FDA 510(k) De Novo**
Predicates: Hexoskin (cleared Nov 2025), CardioTag (cleared 2025), Apple Watch ECG (cleared 2018), AliveCor KardiaMobile (cleared 2012). Timeline: 6–18 months, $100K–$500K.

### Reimbursement

```
+============================================================================+
|  CPT CODE   DESCRIPTION                                   REIMBURSEMENT    |
+============================================================================+
|  93798      Outpatient CR with continuous ECG              Per session      |
|  99453      Initial RPM setup + education                  $22 one-time    |
|  99454      Device supply + data transmission (16+ d/mo)   $47/month       |
|  99457      RPM management, first 20 min                   $52/month       |
|  99458      Each additional 20 min                         $41/month       |
|  99445*     Device supply, 2–15 days (NEW 2026)            $52/month       |
|  99470*     First 10 min management (NEW 2026)             $26/month       |
+============================================================================+
|  Per 100 patients: CPT 93798/session + RPM ~$118,800+/year                 |
+============================================================================+
```

---

*Talk to Your Heart — On-Campus Multi-Agent Cardiac Rehabilitation Intelligence*
*Built on NVIDIA DGX Spark | HIPAA-compliant by architecture | No PHI leaves the building*
