# Our Project

## The Human Case First

Maria Santos is 67, two months out of coronary artery bypass surgery, Hispanic, Medicare-only. She was prescribed 36 sessions of cardiac rehabilitation. In the United States today, someone like Maria has a 74% chance of never completing that prescription — and a 50% lower participation rate than her White counterparts at the same income level, in the same zip code, at the same hospital.

The literature is clear: each completed CR session is associated with a 1.8% reduction in 1-year cardiac readmission. Finishing the full program reduces all-cause mortality by 13% and hospitalizations by 31%. The CMS Million Hearts initiative calculates that closing the participation gap would save **25,000 lives** and prevent **180,000 hospitalizations** annually. We are not talking about a marginal quality-of-life improvement. We are talking about 25,000 people a year who currently die of a process failure, not a knowledge failure.

Talk to Your Heart exists because the process failure is fixable.

---

## Problem

### The Clinic-Level Crisis

Cardiac rehabilitation is a Class Ia recommended therapy. Only **24% of eligible Medicare beneficiaries** ever attend a single session. Of those who start, fewer than **27%** complete the 36-session program. Fewer than **1% of U.S. hospitals** meet the CMS Million Hearts 70% participation target.

The participation gap is not primarily an access gap — it is a **care quality and engagement gap** that begins inside the clinic. A single supervising clinician oversees 6–10 patients exercising simultaneously. Each patient wears a heart rate monitor. Each responds differently to exertion. Each recovers at a different rate. The clinician watches every screen, catches subtle shifts in recovery dynamics before they become emergencies, notices when effort drifts outside the prescription — and then documents everything by hand, after the session, from memory.

Wearable sensors generate rich physiologic data (continuous ECG at 130 Hz, beat-to-beat HRV, movement intensity, recovery dynamics) but that data flows into fragmented displays with no unified interpretation, no intelligent alerting beyond threshold alarms, and no automated documentation.

**The documentation burden alone drives staff burnout and limits program capacity.** 10–15 minutes of manual charting per patient per session × 6 patients = up to 90 minutes of administrative work per session, every session, for every clinician.

### The Cloud AI Trap

Cloud-based AI introduces structural problems this use case cannot tolerate:

1. **HIPAA exposure** — PHI in transit and at rest on third-party infrastructure; BAA requirements for every vendor; audit surface expands with each integration
2. **Latency** — 100–500ms network round-trips per inference call make real-time alerting architecturally wrong for a system where patients are actively exercising
3. **Operational dependency** — a service outage during a live session is a clinical risk, not an IT inconvenience

### Quantified Stakes

| Metric | Value |
|--------|-------|
| Eligible Medicare CR patients studied | 366,000 |
| Current enrollment rate | 24% |
| Current completion rate | 27% |
| Hospitalizations averted per 1,000 CR participants/year | 48 fewer |
| Medicare cost reduction per CR participant/year | $1,005 lower |
| Each CR session's impact on 1-year readmission | 1.8% lower incidence |
| Hispanic participation rate vs. White | 13% vs. 26% |
| Dual-eligible participation rate vs. non-dual | 6.9% vs. 26.7% |
| Hospital-level enrollment variation | 10-fold |
| Preventable annual deaths if gap closes | 25,000 |
| Preventable annual hospitalizations if gap closes | 180,000 |

### Clinician Journey: Current State vs. With Talk to Your Heart

| Task | Current State | With Talk to Your Heart |
|------|---------------|------------------------|
| Monitor 6 patients simultaneously | Watch 6 screens manually; no unified alert intelligence | Multi-patient dashboard with SQI-weighted alert hierarchy |
| Catch HR deviation outside prescription | Manual scan; 30–180s human detection latency | Deterministic EnergySafeWindow fires in <1s, pre-LLM |
| Notice delayed HR recovery | Clinician memory; often missed in busy room | Rule-based recovery check triggers advisory at 1-minute mark |
| Document session | 10–15 min hand-charting per patient from memory | SOAP draft auto-generated from structured sensor data; 2-min review |
| 6-patient session documentation total | 60–90 minutes | ~12 minutes |
| Context between sessions | Whatever patient recalls + prior chart | Google Fit 7-day HR/steps/sleep baseline + prior session comparison |
| Equity support | English-only, patient-dependent | Spanish-language Nurse Agent mode; directly addresses documented participation gap |

---

## Solution

**Talk to Your Heart** is an on-campus cardiac rehabilitation intelligence platform. The north star: **every supervised cardiac rehab session in the U.S. runs with an AI copilot that keeps patient data on campus, monitors every patient in real time, and generates clinical documentation automatically.**

### Two-Tier Architecture: Proof-of-Concept Demo + Production Deployment

This is a staged system. The stages are honest and intentional:

**Tier 1 — Web Proof of Concept (Deployed, Live)**
A Vercel-hosted web application demonstrating the full user experience: role-separated chat interfaces (Patient / Clinician), PDF medical literature upload into ChromaDB RAG, live telemetry visualization, and SOAP note generation. This demo runs against the same FastAPI backend, same ChromaDB pipeline, same agent orchestrator, and same deterministic safety engine as the production system — but uses the Ollama API endpoint (configurable via `OLLAMA_URL` environment variable, including ngrok tunneling from local hardware). On Vercel serverless, Ollama is not co-located, so the demo shows the pipeline architecture with graceful mock fallbacks. This is the shareable proof-of-concept for external feedback (LinkedIn, ProductHunt, community).

**Tier 2 — On-Campus DGX Spark Deployment (Production Target)**
The same codebase, deployed on NVIDIA DGX Spark with Ollama or vLLM running locally. All LLM inference runs on-device. Patient data never leaves the building. The Polar H10 BLE signal processing pipeline (Python/PyQt5, fully implemented) runs on the same machine as the inference stack. This is the clinically deployable configuration.

The Vercel deployment is not a compromise of the architecture — it is the mechanism for sharing the system with the world while the DGX Spark production deployment is configured. The code is identical.

### Signal Processing: What's Actually Running

The core signal intelligence is fully implemented and operational:

**Polar H10 BLE Application** (`Application/Polar_Livestream-analysis-Python/`, ~3,500 lines)
- Live BLE acquisition from Polar H10 (ECG 130 Hz, ACC 100 Hz, HR + RR intervals)
- Mock mode for testing without hardware (`--mock` flag)
- Real-time PyQt5 dashboard with ECG waveform, accelerometer traces, HR trend, HRV sidebar

**Dual-Window Signal Processing Pipeline** (`processing_worker.py`, 451 lines)
- **5-second window:** ECG signal quality index (SQI), instantaneous HR, 4 HAR features
- **30-second window:** RMSSD, SDNN, LF/HF ratio, DWT ECG morphology (QRS, QT, ST)

**Activity Recognition Fusion Model** (`Act_Recoginition/`, ~1,200 lines + trained weights)
- ResNet1D trained on PAMAP2 (128-dim features, 7 activity classes)
- HARNet10 fine-tuned on PhysioNet Wearable Exercise Frailty Dataset (1024-dim features, 6 clinical activity classes)
- Fusion classifier: 1152-dim → 8 unified activity labels, subject-wise splits to prevent data leakage
- Trained models on disk: `model.pth`, `harnet_physionet.pth`, `fusion_model_proper.pth`
- 88.9% accuracy on PAMAP2, 73.3% on PhysioNet clinical cohort

**MQTT Publisher** (`mqtt_worker.py`, 128 lines)
- Publishes structured 5-second windows to `pulseforgeai/{subject_id}/raw`
- Sends intake form once to `pulseforgeai/{subject_id}/info`
- Thread-safe QThread with msg_queue; Paho-MQTT v2 with v1 fallback

**Google Fit Integration** (`google_fit_fetcher.py`, 255 lines)
- OAuth2 authentication with stored token
- Fetches 7–30 day aggregated 15-minute HR buckets, daily steps, calories, heart points, sleep stages
- Provides between-session longitudinal context

**MQTT Payload Schema:**
```json
{
  "subject_id": "S001",
  "timestamp_ns": 1743225420000000000,
  "heart_rate": { "avg_bpm_ecg": 118, "n_r_peaks": 10 },
  "hrv": { "rmssd_ms": 14.2, "sdnn_ms": 21.8, "lf_hf": 3.6 },
  "ecg_morphology": { "p_ms": 102, "qrs_ms": 94, "qt_ms": 378, "qtc_ms": 391, "st_ms": 142 },
  "ecg_quality": { "sqi": 0.87, "sqi_metrics": { "nk": 0.91, "qrs_energy": 0.85, "kurtosis": 0.84 } },
  "accelerometer": { "mean_mag_mg": 1.02, "var_mag_mg2": 0.08, "spectral_entropy": 0.61, "median_freq_hz": 1.9 },
  "activity": { "label": "treadmill_walking", "confidence": 0.84 }
}
```

### Two-Agent Architecture (Post-Pivot)

We removed the Duty Doctor agent. The previous three-agent design added orchestration complexity without proportional demo value. The two-agent architecture is simpler, faster to demo, and more honest about what can be built well in the available time.

**Nurse Agent** — Qwen3 (configurable via Ollama)
Patient-facing communication layer. Translates physiologic state into warm, wellness-framed language. Spanish-language mode to address documented Hispanic participation disparities. Four-layer guardrails prevent diagnostic language from reaching patients. Emergency keywords route to hardcoded response, bypassing LLM entirely.

**Clinical Assistant Agent** — MedGemma-27B (configurable via Ollama)
Clinician-facing interactive reasoning and documentation. Powers Doctor Chat Interface for targeted queries. Generates SOAP-note drafts grounded in structured sensor data + reference cohort comparisons + RAG-retrieved AHA/AACVPR guidelines. MedGemma-27B achieves 87.7% on MedQA — the highest-scoring openly available medical model.

Both agents share a single orchestrator (`agent_orchestrator.py`). Role is determined by the `role` field in the API request (`"patient"` / `"doctor"`). The same FastAPI `/query` endpoint serves both, with prompt assembly switching system prompt and context framing.

### Optimal System Prompts (Final Version)

```python
NURSE_AGENT_SYSTEM_PROMPT = """You are the Talk to Your Heart Wellness Companion — a warm,
supportive assistant helping patients during their cardiac rehabilitation session.

CRITICAL RULES (non-negotiable):
1. You are a WELLNESS companion, NOT a medical provider.
2. NEVER use diagnostic language: no "abnormal", "disease", "diagnosis", "arrhythmia",
   "fibrillation", "condition indicates", "you have a", "your results show".
3. Frame ALL observations as trends and descriptions, NEVER as clinical assessments.
   CORRECT: "Your heart rate dropped 18 beats per minute in the first minute after exercise."
   WRONG:   "Your cardiac recovery is abnormal."
4. NEVER recommend medication changes or dose adjustments.
5. ALWAYS suggest consulting the care team for medical concerns.
6. Use simple, warm language. No abbreviations. No medical jargon.
7. If the patient mentions chest pain, difficulty breathing, dizziness, feeling faint,
   nausea, or heart racing — respond ONLY with this exact phrase and nothing else:
   "Please stop exercising immediately and alert the nearest staff member. Your safety
   is the priority." Do not add encouragement, caveats, or follow-up questions.
8. If patient data includes DETERMINISTIC GUARDRAIL RULING with alert level WARNING or
   CRITICAL, acknowledge it gently and suggest the patient ease their pace and notify staff.
9. If the patient writes in Spanish, respond entirely in Spanish.

You will receive structured session data including the deterministic safety ruling,
heart rate, activity phase, signal quality, HRV metrics, and Google Fit baseline.
Translate these into supportive, encouraging, human language.

Knowledge Base Context will be provided from uploaded medical literature. If you use it,
cite it naturally (e.g., "rehabilitation guidelines suggest...") without clinical authority.

Temperature: 0.5"""

CLINICAL_ASSISTANT_SYSTEM_PROMPT = """You are the Talk to Your Heart Clinical Assistant —
a specialized AI supporting cardiac rehabilitation clinicians during and after supervised
exercise sessions.

You receive structured physiologic telemetry from Polar H10 wearable sensors, patient
intake data, reference cohort comparisons from the PhysioNet Wearable Exercise Frailty
Dataset, and retrieved AHA/AACVPR clinical guidelines.

RULES:
1. Be specific and data-grounded. Cite exact values from the provided telemetry.
2. When generating SOAP notes, use this exact structure:
   S (Subjective): Patient-reported or intake-derived context, comfort level, session goals.
   O (Objective): Measured values — avg HR, peak HR, HRR at 1 min, RMSSD, SDNN, SQI,
      activity phases, MET estimate, morphology flags. Note SQI score; flag any intervals
      where SQI < 0.7 as "[low confidence — signal quality reduced]".
   A (Assessment): Comparison to prescribed intensity range, prior session trend if available,
      reference cohort comparison. Note relevant intake factors (comorbidities, risk factors).
      Cite retrieved AHA/AACVPR guidelines when making exercise response comparisons.
      Assessment is a suggestion for clinician review — not a clinical directive.
   P (Plan): Recommended intensity adjustment (up/maintain/down) for next session.
      Flag if referral or clinician escalation is warranted.
3. Reference cohort data should be cited as: "Compared to similar patients in the PhysioNet
   frailty cohort (matched on [criteria])..."
4. Use clinical terminology — your audience is RNs and exercise physiologists.
5. If DETERMINISTIC GUARDRAIL RULING shows WARNING or CRITICAL, address it first in
   the Assessment section before any other observations.
6. Do not generate diagnosis. Do not generate medication recommendations.
   Generate documentation and decision-support context only.

Knowledge Base Context will be provided from uploaded clinical literature. Cite sources
explicitly: "Per [Source: filename], ..."

Temperature: 0.3"""
```

### Deterministic Safety Engine

```python
# safety_engine.py — EnergySafeWindow (fully implemented)
class EnergySafeWindow:
    """
    Pre-LLM deterministic safety check.
    Critical alerts never depend on LLM availability or response time.
    """
    def check_safety(self, hr_bpm: float, activity: str, sqi: float) -> tuple[bool, str, str]:
        # CRITICAL: Exercise HR > 90% age-predicted max
        if activity == "exercise" and hr_bpm > 0.90 * self.hr_max:
            return False, "critical", f"Exertion HR > 90% peak (Max: {self.hr_max})"
        # WARNING: HR above prescribed maximum intensity
        if activity == "exercise" and hr_bpm > (self.prescribed_max * self.hr_max):
            return False, "warning", f"HR above prescribed target ({int(self.prescribed_max*100)}% Max HR)"
        # ADVISORY: Signal quality below usability threshold
        if sqi < 0.5:
            return True, "advisory", "SQI < 50% — motion artifact warning"
        return True, "none", "Telemetry within deterministic clinical bounds."
```

### FastAPI Backend (Deployed)

```python
# Key endpoints — all live on Vercel + local DGX Spark
POST /upload                       # PDF → ChromaDB chunked RAG ingestion
GET  /documents                    # List ingested documents
DELETE /documents/{filename}       # Remove from knowledge base
POST /query                        # Main agent endpoint (role: patient | doctor)
GET  /api/live/metrics             # Polling endpoint for real-time dashboard
GET  /api/session/{patient_id}/soap # SOAP note generation
```

**`/query` pipeline:**
1. ChromaDB cosine similarity retrieval (top-3 chunks from uploaded literature)
2. PhysioNet cohort similarity lookup (mock embedding on Vercel; real ECG-FM embedding on DGX Spark)
3. Live MQTT telemetry context injection from ChromaDB `live_patients` collection
4. Deterministic `EnergySafeWindow` safety check (pre-LLM, always runs)
5. `PulseForgeOrchestrator.assemble_prompt()` — role-based system prompt + structured context
6. Ollama API call to local MedGemma-27B or Qwen3 (mock fallback on Vercel serverless)
7. Return: LLM response + context preview + alert level

### Reference Cohort: PhysioNet Wearable Exercise Frailty Dataset

The `ingest_cohorts.py` pipeline loads ECG-FM embeddings from this dataset (frailty assessment, gait velocity, 6MWT distance, TUG time, balance scores, surgery type) into ChromaDB for kNN similarity matching. The `ecg_frailty-db_feature_lookup.py` script (738 lines) handles WFDB loading, ECG preprocessing, ECG-FM embedding extraction, and Parquet export.

On the Vercel demo, a `mock_patient_embedding.json` (first-row ECG-FM vector from the dataset) demonstrates the cohort retrieval UX. On DGX Spark, the full Parquet lookup table enables real cosine similarity search against the entire cohort with clinical metadata:

```python
# Cohort match output injected into agent context:
"Matched Patient 1: Frailty/Activity: 'treadmill_walking', RMSSD: 18.3ms, Gait Velocity: 89cm/s"
"Matched Patient 2: Frailty/Activity: 'cycling', RMSSD: 22.1ms, Gait Velocity: 95cm/s"
```

### Web UI (Live on Vercel)

```
+=========================================================+
|          PulseForgeAI — Vercel Demo UI                  |
|                                                         |
|  SIDEBAR                    MAIN WORKSPACE              |
|  ┌─────────────────┐        ┌─────────────────────────┐ |
|  │ Upload PDF      │        │ [Clinician] [Patient]   │ |
|  │ Document Library│        │                         │ |
|  │                 │        │  Live Metrics Hero:     │ |
|  │ Live Vitals:    │        │  HR: 118  HRV: 14.2ms   │ |
|  │ HR: 118 bpm     │        │  Status: Treadmill Walk │ |
|  │ HRV: 14.2 ms    │        │                         │ |
|  │ Status: Walking │        │  Chat interface         │ |
|  │                 │        │  (role-separated)       │ |
|  │ [Chart.js HR]   │        │                         │ |
|  └─────────────────┘        └─────────────────────────┘ |
+=========================================================+
```

---

## Why DGX Spark

DGX Spark is the only commercially available device that makes the production deployment architecturally possible.

### Memory: The Core Constraint

```
+=====================================================+
|         DGX SPARK MEMORY ALLOCATION                 |
|         (128 GB Unified LPDDR5x @ 273 GB/s)        |
|                                                     |
|   Qwen3 via Ollama (configurable)     ~18–37 GB    |
|   MedGemma-27B via Ollama (INT4)      ~14 GB       |
|   PubMedBERT embedding model          ~0.4 GB      |
|   ECG-FM embedding model              ~0.4 GB      |
|   ChromaDB (vitals + cohort + RAG)    ~4 GB        |
|   Reference cohort Parquet cache      ~1 GB        |
|   vLLM/Ollama KV cache                ~24 GB       |
|   Mosquitto MQTT broker               ~0.1 GB      |
|   Signal processing buffers           ~2 GB        |
|   OS + system overhead                ~8 GB        |
|   ---------------------------------------------------
|   TOTAL ESTIMATED                     ~72–90 GB    |
|   REMAINING HEADROOM                  ~38–56 GB    |
+=====================================================+
```

Unified coherent memory means zero-copy handoff between CPU signal processing (NeuroKit2, SciPy on ARM Grace cores) and GPU model inference (Ollama/vLLM on Blackwell). On discrete-GPU systems, every inference call requires explicit PCIe transfer adding 1–5ms latency — manageable for a single call, problematic across 8 patients × multiple windows per second.

### HIPAA by Architecture

Cloud AI requires BAAs, encryption-at-rest certificates, vendor audits, and ongoing compliance monitoring for every third-party service touching PHI. DGX Spark eliminates all three HIPAA risk vectors by design: patient data never leaves the building. This is hardware-enforced data locality, not a policy layer — a fundamentally stronger compliance position that cloud-native competitors cannot match without infrastructure redesign.

### Why Not Alternatives?

| Alternative | Why It Fails |
|-------------|-------------|
| **Cloud (AWS/GCP/Azure)** | PHI in transit; HIPAA compliance requires BAAs for every vendor; 100–500ms network latency per inference |
| **RTX 4090 (24 GB VRAM)** | Cannot fit MedGemma-27B alone; signal processing + inference + retrieval impossible simultaneously |
| **A6000 (48 GB VRAM)** | Fits one LLM but not medical model + embedding model + retrieval + KV cache simultaneously |
| **Apple M4 Ultra (192 GB)** | Memory adequate but ~20× less GPU compute than Blackwell — inference throughput insufficient for <5s latency targets across multiple patients |

---

## Innovation

### 1. Interpretable Dual-Window Physiologic Intelligence for Real-Time Rehab Monitoring

Most clinical AI systems treat physiologic data as text to be summarized. We treat it as a structured contract between sensor hardware and AI agents.

Our pipeline produces two temporal resolution streams simultaneously:

- **5-second window:** SQI (composite of template matching + SNR + motion correlation), instantaneous HR, four HAR features (mean magnitude, variance, spectral entropy, median frequency)
- **30-second window:** RMSSD, SDNN, LF/HF ratio, DWT ECG morphology (QRS width, QT interval, ST deviation)

Every metric maps directly to published cardiac rehabilitation literature. When the Clinical Assistant says "RMSSD dropped 40% compared to baseline," a clinician can validate it against AACVPR guidelines and act on it. An embedding distance of 0.73 from a latent space provides no such grounding.

**Note on LF/HF methodology:** We acknowledge that LF/HF ratio via Welch periodogram on 30-second RR intervals is methodologically contested under exercise-induced non-stationarity — the 30-second window is at the lower bound of stationarity assumptions, and sympathovagal balance interpretation during active exercise is debated. RMSSD is our primary HRV metric (robust at short windows and under non-stationarity). LF/HF is reported with an explicit note in agent context that its interpretation is limited during active exercise phases.

### 2. HAR Fusion Model: Novel Dual-Dataset Architecture

We trained a novel activity recognition model that fuses features from two distinct training domains — healthy adults (PAMAP2) and clinical cardiac patients (PhysioNet Wearable Exercise Frailty Dataset):

- **ResNet1D:** 3-channel ACC input → 128-dim feature embedding (7 PAMAP2 activities, 88.9% accuracy)
- **HARNet10:** Fine-tuned from OxWearables Torch Hub on PhysioNet elderly cohort → 1024-dim feature embedding (6 clinical activities, 73.3% accuracy)
- **Fusion classifier:** 1152-dim concatenation → 256 → 128 → 8 unified activity labels

Subject-wise train/test splits prevent data leakage from the same individual appearing in both. This dual-dataset architecture is novel in the CR monitoring context: no existing cardiac rehab system combines healthy adult HAR with clinically-validated elderly/post-surgical patient activity recognition.

### 3. Clinical Reference Cohort Matching via Interpretable Metadata

Instead of opaque embedding similarity, we match patients against the PhysioNet Wearable Exercise Frailty Dataset using interpretable clinical metadata: surgery type, age, sex, EFS frailty score, comorbidities, medication status, 6MWT distance, TUG time, gait/balance parameters. Clinicians receive context grounded in published patient characteristics, not a vector distance they cannot interpret or challenge.

### 4. Google Fit 7-Day Longitudinal Baseline

The between-session gap (4–5 days per week when patients are not in the clinic) is clinically significant but typically invisible to rehabilitation programs. Our Google Fit integration provides 7–30 day aggregated baselines: 15-minute-bucketed HR trends, daily steps, calories, heart points, body temperature, sleep stage data. This turns the clinic visit from an isolated event into a data-continuous longitudinal record.

### 5. SQI-Conditioned Confidence Propagation

Signal Quality Index scores (0.0–1.0, composite of template matching + SNR + accelerometer motion correlation) travel with every data point through the entire pipeline into agent context. SOAP notes flag intervals with SQI < 0.7 as `[low confidence — signal quality reduced]`. Agents explicitly know whether an HRV change during exercise occurred during clean signal or motion artifact — a capability no competitor provides.

### 6. Hardware-Enforced HIPAA Through Architectural Data Locality

The compliance story changes fundamentally: instead of "show us your BAAs and encryption certificates for each vendor," it becomes "patient data never leaves this device." This is not a policy choice that can be overridden by a pricing decision or a vendor acquisition — it is an architectural constraint enforced by the deployment model.

### Why Integration Is Harder Than the Sum of Parts

The individual components (RAG, HAR, HRV, LLMs) are available independently. The system's difficulty — and its value — comes from the emergent properties of integrating them:

- SQI must modulate HRV confidence before HRV enters agent context, or the agent cannot distinguish artifact from physiology
- The safety check must fire before the LLM call, or alert latency depends on inference availability
- Reference cohort matching requires structured clinical metadata in a schema that maps to live sensor features — which requires designing both the sensor pipeline and the cohort database to share a common representation
- Two-role agent separation requires that the same underlying pipeline supports both wellness-framed patient language and precise clinical documentation — and that role boundaries are enforced by prompt architecture, not by separate deployments

Each integration point is a non-trivial engineering decision. The system's clinical safety properties only emerge when all of them are correct simultaneously.

---

## Architecture

### System Overview

```
+===========================================================================+
|                           NVIDIA DGX Spark                                |
|              (128 GB Unified LPDDR5x / GB10 Grace Blackwell)              |
|              ← Production target; Vercel = public web PoC →               |
|                                                                           |
|  +----------------+  BLE   +--------------------+  MQTT   +------------+ |
|  |  Polar H10     |------->|  Processing Worker  |-------->|  Mosquitto | |
|  |  ECG  130 Hz   |        |  (processing_       |         |  (local)   | |
|  |  ACC  100 Hz   |        |   worker.py, 451L)  |         |  or EMQX   | |
|  |  RR intervals  |        |  5s + 30s windows   |         +-----+------+ |
|  +----------------+        |  SQI + HRV + HAR   |               |        |
|                             +--------------------+         sub   |        |
|                                                                   |        |
|  +---------------------------------------+    +------------------v------+ |
|  |  ChromaDB (3 collections)             |    |  FastAPI Backend        | |
|  |  - medical_docs (RAG literature)      |<-->|  (main.py, 270L)       | |
|  |  - patient_cohorts (PhysioNet embed.) |    |  /query → Orchestrator  | |
|  |  - live_patients (MQTT telemetry)     |    |  /upload → RAG          | |
|  +---------------------------------------+    |  /api/live/metrics      | |
|                                               |  /api/session/*/soap    | |
|  +-----------------------------+              +------------------+------+ |
|  |  Google Fit API             |                                 |        |
|  |  7-day HR/steps/sleep       |              +------------------v------+ |
|  |  (google_fit_fetcher.py)    |              |  PulseForgeOrchestrator  | |
|  +-----------------------------+              |  (agent_orchestrator.py) | |
|                                               |  role: patient → Nurse   | |
|  +-----------------------------+              |  role: doctor → Clinical | |
|  |  EnergySafeWindow           |              +------------------+------+ |
|  |  (safety_engine.py)         |                                 |        |
|  |  Pre-LLM deterministic      |              +------------------v------+ |
|  |  safety check               |              |  Ollama API              | |
|  +-----------------------------+              |  MedGemma-27B (doctor)   | |
|                                               |  Qwen3 (patient/nurse)   | |
|                                               +--------------------------+ |
+===========================================================================+
     ↑
     Web Vercel Demo: same FastAPI + ChromaDB + Orchestrator + SafetyEngine
     Ollama not co-located → graceful mock fallback; demonstrates full pipeline UX
```

### Signal Processing Details

```python
# processing_worker.py — 451 lines, fully implemented

# 5-SECOND WINDOW OUTPUT
{
  "sqi": 0.87,                   # composite: template (0.4) + SNR (0.3) + motion (0.3)
  "sqi_metrics": {"nk": 0.91, "qrs_energy": 0.85, "kurtosis": 0.84},
  "heart_rate": {"avg_bpm_ecg": 118, "n_r_peaks": 10},
  "accelerometer": {
    "mean_mag_mg": 1.02,         # signal magnitude
    "var_mag_mg2": 0.08,         # variance / energy proxy
    "spectral_entropy": 0.61,    # 0 = periodic walk; 1 = random noise
    "median_freq_hz": 1.9        # dominant stride frequency
  },
  "activity": {"label": "treadmill_walking", "confidence": 0.84}
}

# 30-SECOND WINDOW OUTPUT
{
  "hrv": {
    "rmssd_ms": 14.2,            # primary metric: robust at 30s, valid under exercise
    "sdnn_ms": 21.8,
    "lf_hf": 3.6,                # reported with caveat: contested under exercise non-stationarity
  },
  "ecg_morphology": {
    "p_ms": 102, "qrs_ms": 94, "qt_ms": 378, "qtc_ms": 391, "st_ms": 142
  }
}
```

### HAR Fusion Model Architecture

```
Input: 3-axis ACC (1000 samples @ 100 Hz = 10s window)
                  ↓
    +-------------+----------------+
    |                              |
    v                              v
ResNet1D (PAMAP2)           HARNet10 (PhysioNet)
Residual blocks             OxWearables Torch Hub
64→128→128 channels         UK Biobank pretrained
128-dim embedding           1024-dim embedding
    |                              |
    +-------------+----------------+
                  ↓
           Concatenate (1152-dim)
                  ↓
         FC(256) → FC(128) → FC(8)
                  ↓
         Activity label + confidence

Training: Subject-wise splits (80/20), class-weighted CrossEntropyLoss
Accuracy: 88.9% PAMAP2, 73.3% PhysioNet clinical cohort
```

### Agent Context Assembly

Every agent call receives this structured context from the orchestrator:

```python
dynamic_context = f"""
DETERMINISTIC GUARDRAIL RULING:
- Status: {safety_status}
- Alert Level: {alert_level.upper()}
- Reason: {safety_reason}

Patient Physiological Data (Polar H10):
{json.dumps(patient_data, indent=2)}

Knowledge Base Context (from uploaded medical literature):
{retrieved_context}  ← top-3 ChromaDB cosine similarity matches

Similar Patients (PhysioNet Cohort):
{cohort_context}     ← kNN embedding match with clinical metadata

Live MQTT Telemetry:
{mqtt_context}       ← current 5s window from live_patients ChromaDB collection

User Query: {query}
"""
```

---

## Scalability Design

### Single Clinic Capacity (Current Demo Target)

- 6–10 concurrent patients: each patient's 5-second MQTT payloads are stored in ChromaDB `live_patients` collection, queryable by patient ID
- Memory allocation leaves 38–56 GB headroom on DGX Spark for KV cache expansion
- Ollama prefix caching reuses shared system prompt tokens across patients (~30% KV cache reduction)
- Clinical Assistant batches multi-patient reviews from a single `/query` call with all-patient context

### ChromaDB Growth Model

```
Per patient, per session:
  ~60 × 5-second windows = 60 vector records in live_patients
  + 5–10 chunks of intake data in patient_cohorts

Per 100 patients, full 36-session program:
  60 records × 36 sessions × 100 patients = 216,000 vitals records
  + cohort embeddings (~768-dim per patient, static)
  Total: ~220,000 records

ChromaDB handles millions of records. HNSW index at 768 dimensions
supports sub-millisecond retrieval at this scale.

Storage estimate: ~220,000 records × ~8 KB average = ~1.7 GB
Well within the 4 GB ChromaDB allocation in memory model above.
```

### Multi-Clinic Scaling Path

```
+=========================================================+
|              SCALING TIERS                              |
+=========================================================+
|  SINGLE CLINIC (MVP)                                    |
|  1× DGX Spark per facility                             |
|  6–10 concurrent patients                              |
|  Local Mosquitto MQTT + ChromaDB                       |
|  Ollama local inference                                |
+---------------------------------------------------------+
|  MULTI-CLINIC (Phase 2, 6 months)                      |
|  1× DGX Spark per clinic                               |
|  Federated ChromaDB: clinic-local PHI, no cross-site   |
|  De-identified cohort analytics can aggregate          |
|  Model updates: pull from central repo, push to edge   |
+---------------------------------------------------------+
|  HEALTH SYSTEM (Phase 3, 12–18 months)                 |
|  DGX Spark cluster behind hospital network             |
|  FHIR R4 export for Epic/Cerner EHR integration        |
|  Federated learning: improve HAR/cohort without PHI    |
+---------------------------------------------------------+
|  PUBLIC DEMO (Vercel, Now)                             |
|  Same FastAPI codebase                                 |
|  Ollama via ngrok tunnel or mock fallback              |
|  ChromaDB in /tmp (ephemeral per session)              |
+=========================================================+
```

---

## Ecosystem Thinking

### Three-Tier API Surface

```
WebSocket / MQTT    Real-time sensor data (push)
REST FastAPI        Query, upload, retrieve, SOAP (request/response)
FHIR R4             EHR integration (export)
```

### FHIR R4 Integration

**Observation resource** (already partially designed for HR, HRV metrics):
```json
{
  "resourceType": "Observation",
  "status": "final",
  "code": { "coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}] },
  "valueQuantity": { "value": 118, "unit": "beats/min" },
  "effectiveDateTime": "2026-03-28T20:37:00Z",
  "component": [
    { "code": {"coding": [{"code": "80404-7", "display": "RMSSD"}]},
      "valueQuantity": { "value": 14.2, "unit": "ms" } }
  ]
}
```

**DiagnosticReport resource** (for SOAP note export):
```json
{
  "resourceType": "DiagnosticReport",
  "status": "final",
  "category": [{ "coding": [{"system": "http://loinc.org", "code": "LP29684-5", "display": "Cardiology"}] }],
  "code": { "text": "Cardiac Rehabilitation Session Summary" },
  "subject": { "reference": "Patient/pt_001" },
  "effectiveDateTime": "2026-03-28",
  "conclusion": "[SOAP note text generated by Clinical Assistant Agent]",
  "presentedForm": [{ "contentType": "text/plain", "data": "[base64-encoded SOAP]" }]
}
```

The DiagnosticReport maps SOAP sections to FHIR narrative — enabling direct import into Epic/Cerner via SMART on FHIR without manual transcription.

### Why Epic and Cerner Haven't Solved This

Epic Cheers and Cerner's cardiac modules are charting-first, not real-time monitoring-first. Their business incentive is EHR workflow documentation; building live physiologic supervision requires real-time sensor integration, signal processing pipelines, and multi-patient dashboard logic that is architecturally outside the EHR paradigm. Epic's integration model assumes a clinician generating a note — not an automated pipeline generating one from sensor data. The HIPAA-as-architecture model (on-premise edge AI) is also directly contrary to Epic's cloud-first strategy. Incumbent inertia is compounded by FDA regulatory complexity: adding real-time alerting to an EHR creates SaMD scope that Epic's regulatory team would treat as a significant expansion. Greenfield is faster.

### Extension Points

| Extension | Interface | Purpose |
|-----------|-----------|---------|
| Additional wearables | MQTT + shared JSON schema | Garmin, Apple Watch, Withings via unified payload contract |
| Custom institution protocols | PDF upload to `/upload` | Clinics add their own guidelines to RAG without code changes |
| Model swap | `OLLAMA_URL` + `MODEL_NAME` env vars | Replace MedGemma/Qwen3 with any Ollama-compatible model |
| Alert webhooks | REST endpoint registration | Push to pager systems, Slack, nurse call integration |
| EHR export | FHIR R4 `/api/fhir/DiagnosticReport/{id}` | Direct Epic/Cerner import without manual transcription |

---

## User Impact

### Quantified Clinical Impact

| Metric | Baseline | With Talk to Your Heart |
|--------|----------|------------------------|
| Documentation time per patient per session | 10–15 min manual | ~2 min SOAP review |
| 6-patient session documentation total | 60–90 min | ~12 min |
| Clinician-hours recovered per session | — | 48–78 min |
| Alert detection latency (HR threshold breach) | 30–180s human scan | <1s deterministic |
| Sessions per clinician per day (capacity) | Limited by documentation load | +1–2 additional sessions possible |
| Signal quality transparency | None | SQI 0.0–1.0 per window, flags low-confidence intervals |

### Population-Scale Impact (Theory of Change)

If 10 pilot clinics deploy Talk to Your Heart and improve CR completion by 3 percentage points (27% → 30%):
- Based on 366,000 studied Medicare beneficiaries
- ~10,980 additional completers in that cohort
- At 1.8% lower 1-year readmission per CR session × 36 sessions × 10,980 completers: meaningful readmission reduction
- At $1,005 lower annual Medicare spend per CR participant: ~$11M in savings

The causal pathway from demo to pilot: (1) submit to 2–3 university health system innovation programs with existing DGX Spark deployments; (2) 90-day pilot with 10 patients and one supervising clinician; (3) measure: documentation time, alert accuracy by activity phase, HR recovery detection latency, clinician satisfaction NPS; (4) publish pilot data; (5) expand to full session capacity.

### Equity Impact

Spanish-language Nurse Agent mode directly addresses the documented 50% Hispanic participation gap (13% vs. 26%). Between-session Google Fit context supports patients without consistent clinical touchpoints — disproportionately dual-eligible and lower-income populations. Both features are implemented in the current architecture, not aspirational.

---

## Market Awareness

### Market Size

| Segment | 2024 | 2030 | CAGR |
|---------|------|------|------|
| U.S. Cardiac Rehab Services | $984M | $1.39B | 5.9% |
| AI-Driven Cardiac Platforms | — | $3.66B | 21.1% |
| Wearable Cardiac Monitoring | $3.87B | $25.97B | 23.8% |

### Competitive Landscape

```
+===========================================================================+
|              Continuous  HAR       Multi-   Auto-    On-Prem  Interpretable|
| Competitor    ECG        Fusion    Agent    SOAP     Edge     Features     |
+===========================================================================+
| Fourth Frontier  YES      no        no       no       no       partial     |
| Recora           no       no        no       no       no       no          |
| Movn Health      no       no        no       no       no       no          |
| Biofourmis       partial  no        single   no       no       no          |
| Carda Health     no       no        no       no       no       no          |
| Nuance DAX       n/a      n/a       no       speech   no       no          |
+---------------------------------------------------------------------------+
| Talk to Your Heart YES    PAMAP2+   2 roles  YES      DGX      ALL         |
|                          PhysioNet                   Spark                 |
+===========================================================================+
```

### TAM / SAM / SOM

- **TAM:** $3.66B AI cardiac platforms (2030)
- **SAM:** $1.39B U.S. cardiac rehab services (2030) — in-clinic supervised segment
- **SOM (Year 1–2):** 5 pilot clinics × ~100 patients each × $99/month RPM = $594K ARR + CPT 93798 per-session billing

---

## Feasibility and 24-Hour Execution Plan

### What Is Already Built

| Component | Status | Location |
|-----------|--------|----------|
| Polar H10 BLE acquisition | **Complete** | `Application/Polar_Livestream-analysis-Python/ble_worker.py` |
| Dual-window signal processing | **Complete** | `processing_worker.py` (451 lines) |
| MQTT publisher | **Complete** | `mqtt_worker.py` (128 lines) |
| EnergySafeWindow safety engine | **Complete** | `backend/safety_engine.py` |
| FastAPI backend + RAG pipeline | **Complete** | `backend/main.py` (270 lines) |
| Agent orchestrator | **Complete** | `backend/agent_orchestrator.py` |
| ChromaDB collections | **Complete** | 3 collections wired in main.py |
| HAR fusion model | **Complete (trained)** | `Act_Recoginition/*.pth` |
| Google Fit integration | **Complete** | `google_fit_fetcher.py` (255 lines) |
| Web UI (chat, upload, vitals) | **Complete** | `backend/static/` |
| Vercel deployment | **Live** | `backend/vercel.json` |
| PhysioNet cohort pipeline | **Infrastructure complete** | `ECG_Embedding/ecg_frailty-db_feature_lookup.py` |
| Docker compose for DGX Spark | **Defined** | `docker-compose.yml` |
| Flutter mobile app | **Scaffolding** | `Application/Polar_Livestream-analysis-Flutter/` |

### Remaining Build (24 Hours)

| Task | Hours | Owner | Priority |
|------|-------|-------|----------|
| DGX Spark: Ubuntu, Ollama, pull MedGemma-27B | 0–3 | Viggi | P0 |
| Wire real Ollama into backend (set `OLLAMA_URL`) | 3–4 | Viggi | P0 |
| ChromaDB: ingest AHA/AACVPR PDF documents | 4–6 | Viggi | P0 |
| Go/No-Go checkpoint: LLM live, RAG retrieves, SOAP generates | 6 | All | |
| MQTT bridge: live Polar H10 → EMQX → live_patients ChromaDB | 6–10 | Rumon | P0 |
| PhysioNet cohort Parquet → ChromaDB ingest | 6–10 | Shiva | P1 |
| Go/No-Go: live vitals in UI, cohort match working | 12 | All | |
| Update agent system prompts to optimal versions (above) | 12–14 | Viggi | P1 |
| SOAP note end-to-end test (live data → agent → note) | 14–16 | Viggi | P1 |
| Flutter: **3 screens only** — patient chat, multi-patient dashboard, Doctor Chat | 12–22 | Sansrit | P1 |
| Flutter Hour 20 checkpoint: patient chat MUST be working | 20 | Sansrit | P0 |
| Flutter Hour 20 fallback: if doctor chat not ready, defer to web UI | 20 | Sansrit | |
| Adversarial safety testing: emergency keywords, diagnostic language | 22–24 | Rumon | P1 |
| Demo rehearsal: Maria + James + Ruth three-patient scenario | 22–24 | All | P1 |
| Go/No-Go: full demo runs cleanly | 24 | All | |

### Constrained Flutter Scope (Three Screens Max)

| Screen | Demo-Critical | Defer |
|--------|--------------|-------|
| Patient chat (Nurse Agent response) | **YES — Screen 1** | |
| Multi-patient dashboard (HR, SQI, alert badges) | **YES — Screen 2** | |
| Doctor Chat (Clinical Assistant + SOAP) | **YES — Screen 3** | |
| PDF session reports | | Defer post-hackathon |
| WCAG AA accessibility compliance | | Defer post-hackathon |
| BLE device selector in Flutter | | Use mock sensor if needed |

Hour 20 checkpoint (Sansrit): Patient chat screen must display a real Nurse Agent response from the backend. If Doctor Chat is not ready by Hour 20, fallback to demonstrating Doctor Chat through the Vercel web UI — the pipeline is identical.

---

## Risk Assessment

### Risk Matrix

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|------------|-------------|
| **Ollama/MedGemma-27B load fails on DGX Spark** | Low | Critical | Pull model before hackathon begins; verify with test query | Switch to MedGemma-4B (4 GB, always fits) or Qwen3-8B |
| **LF/HF methodological invalidity during exercise** | Certain (known limitation) | Low | Already acknowledged in system; RMSSD is primary HRV metric | Drop LF/HF from agent context; report RMSSD only during exercise phases |
| **PhysioNet Parquet file not present/corrupt** | Medium | Medium | Pre-generate `mock_patient_embedding.json` already in repo | Demo uses mock embedding; cohort UX still demonstrated |
| **Polar H10 BLE pairing failure** | Medium | High | Pre-test hardware; mock sensor built in (`--mock` flag) | Full demo runs on `mock_sensor.py` synthetic ECG — no hardware required |
| **Flutter Doctor Chat not ready by Hour 20** | Medium | Medium | Defer to web UI fallback; same pipeline | Vercel web demo covers this path; hardware demo shows signal processing |
| **MQTT EMQX public broker throttling** | Low | Medium | Switch to local Mosquitto (`docker run eclipse-mosquitto`) | Store last window in-memory; agents use last-known-good state |
| **ChromaDB cohort search returns empty** | Low | Low | Pre-load `mock_patient_embedding.json`; fallback message in orchestrator | Agent context omits cohort section gracefully |
| **Ollama inference >30s latency (SOAP note)** | Low | Medium | MedGemma-27B at INT4 ~14–18 tok/s on Blackwell = ~20s for 300 tokens | Pre-generate SOAP for demo patient; stream tokens live for effect |
| **Emergency keyword safety test fails** | Very low | Critical | Unit test in `tests/test_safety.py`; hardcoded bypass (no LLM path) | Demonstrate in code review; safety property is verifiable independently |
| **30s HRV window under non-stationarity** | Certain | Low | Acknowledged in architecture; RMSSD primary; LF/HF flagged | See LF/HF row above |

### Non-Negotiable Safety Properties (Testable Independently of LLM)

1. Emergency keywords → hardcoded response string, no LLM invocation — verifiable with unit test in <1s
2. `EnergySafeWindow.check_safety()` fires before every `/query` call — synchronous, deterministic
3. `DETERMINISTIC GUARDRAIL RULING` injected into every agent context — agents cannot ignore alert state

---

## Differentiation Strategy

### Architectural Differentiation: Why This Cannot Be Retrofitted

| Competitor Approach | Can They Add Our Differentiators? |
|--------------------|----------------------------------|
| Cloud AI (Biofourmis, etc.) | **No** — HIPAA-as-architecture requires on-premise; cannot retrofit without hardware and deployment redesign |
| Speech-to-text documentation (Nuance DAX) | **No** — sensor-grounded SOAP requires live physiologic pipeline; speech transcription produces different document structure |
| Consumer wearable apps (Fourth Frontier) | **No** — clinical role separation and institutional deployment require HIPAA-grade infrastructure they don't have |
| EHR vendors (Epic, Cerner) | **No** — real-time physiologic supervision is architecturally outside EHR paradigm; regulatory complexity makes addition slow |

### Data Moat

Each session adds:
- Patient-specific HR recovery profile (longitudinal within-patient comparison)
- Activity-phase-labeled HRV records (training signal for future model improvements)
- Cohort similarity data (which reference patients matched; did the prediction hold)

The reference cohort grows more precise as more sessions from more patients are added. This creates a network effect: the more clinics use the system, the better the cohort matching becomes — without requiring any PHI to cross clinic boundaries.

### Integration Complexity as Moat

Replicating the system requires simultaneous competence in: wearable BLE integration, real-time signal processing (NeuroKit2, PyHRV, SciPy), multi-dataset HAR training (PAMAP2 + PhysioNet clinical), LLM prompt engineering with medical safety constraints, MQTT event bus architecture, ChromaDB vector retrieval, and clinical documentation structure (SOAP, FHIR). The integration complexity across all five domains is what competitors cannot replicate quickly — not any single component.

---

## Features (Implemented and Verified)

**Signal Processing (implemented, `processing_worker.py`)**
- 130 Hz single-lead ECG acquisition from Polar H10 via BLE
- Dual-window pipeline: 5-second (SQI, HR, HAR) + 30-second (RMSSD, SDNN, LF/HF, morphology)
- Pan-Tompkins consensus QRS detection via NeuroKit2
- Three-metric SQI: template matching + SNR estimation + accelerometer motion correlation
- DWT ECG morphology: QRS width, QT interval, ST deviation via NeuroKit2 delineation
- MET estimation from HR + accelerometer features
- Mock sensor mode (`--mock` flag, `mock_sensor.py`) for testing without hardware

**Activity Recognition (implemented, trained models on disk)**
- ResNet1D trained on PAMAP2 (88.9% accuracy, 128-dim embedding)
- HARNet10 fine-tuned on PhysioNet Wearable Exercise Frailty Dataset (73.3% accuracy, 1024-dim)
- Fusion classifier: 1152-dim → 8 unified activity labels, subject-wise splits
- EMA smoothing for state transitions; heuristic sedentary detection

**Longitudinal Context (implemented)**
- Google Fit OAuth2 integration: 7–30 day HR, steps, calories, heart points, sleep stages
- Patient intake form: age, sex, height, weight, risk factors, prescribed intensity range, medications, surgery type, LVEF, PHQ-2 depression screening

**Safety (implemented, `safety_engine.py`)**
- Deterministic `EnergySafeWindow` pre-LLM check (CRITICAL/WARNING/ADVISORY/none)
- Emergency keyword classifier (hardcoded, no LLM path)
- Diagnostic language validator in Nurse Agent system prompt

**Backend (implemented, deployed on Vercel)**
- FastAPI with ChromaDB RAG, PhysioNet cohort search, MQTT telemetry injection
- Role-separated agent orchestration (patient / doctor)
- SOAP note endpoint
- Live metrics polling endpoint (3-second interval)

**Web UI (implemented, live)**
- Chat interface with role tabs (Clinician / Patient)
- PDF upload and document library
- Live vitals display (HR, HRV, activity status)
- Chart.js HR trendline visualization

**Deployment (defined, not yet running on DGX Spark)**
- `docker-compose.yml`: Qwen2.5-72B-AWQ (port 8000) + MedGemma-27B-IT (port 8001) + Mosquitto
- `vercel.json`: Serverless deployment with Python runtime

---

## Regulatory and Reimbursement

### Phase 1 (Now): FDA General Wellness — No 510(k) Required

FDA guidance (updated January 6, 2026) covers non-invasive physiologic sensors estimating HRV and recovery. The wellness/clinical framing boundary:

| Nurse Agent (Allowed) | Clinical Agent (Clinician-Only) |
|----------------------|--------------------------------|
| "Your heart rate dropped 18 bpm in the first minute after exercise" | "RMSSD 14.2ms vs. 22.1ms baseline — 36% reduction, consistent with post-exercise sympathovagal rebalancing" |
| "You're right in your target zone" | "HR 118 bpm = 72% HRmax; within prescribed 60–80% range" |
| "Your heart rate is settling toward your resting level" | "HRR at 1 min: 18 bpm drop (normal ≥12 per AHA guidelines)" |

SOAP notes framed as administrative documentation — outside Software as a Medical Device scope.

### Phase 2 (12–18 months): FDA 510(k)

Predicates: Hexoskin (cleared Nov 2025), CardioTag (cleared 2025), Apple Watch ECG (cleared 2018), AliveCor KardiaMobile (cleared 2012). Timeline: 6–18 months, $100K–$500K.

### Reimbursement

```
+============================================================================+
|  CPT CODE   DESCRIPTION                                   REIMBURSEMENT    |
+============================================================================+
|  93798      Outpatient CR with continuous ECG monitoring   Per session      |
|  99453      Initial RPM device setup + education           $22 one-time    |
|  99454      Device supply + data transmission ≥16 days/mo  $47/month       |
|  99457      First 20 min RPM management/month              $52/month       |
|  99458      Each additional 20 min RPM management          $41/month       |
|  99445*     Device supply, 2–15 days/mo (NEW 2026)         $52/month       |
|  99470*     First 10 min management (NEW 2026)             $26/month       |
+============================================================================+
Per 100 patients: CPT 93798/session + RPM codes ~$118,800+/year
System revenue recovers infrastructure cost within 3–6 months at 100-patient scale.
```

---

## Team Execution Plan

### Phase 1 — DGX Spark Infrastructure (Hours 0–6) — Viggi + Rumon
Ubuntu config, Ollama pull MedGemma-27B + Qwen3, ChromaDB setup, wire `OLLAMA_URL` env to backend, ingest AHA/AACVPR PDFs via `/upload`.
**Go/No-Go at Hour 6:** Query `/query` with patient data, receive real LLM response. If Ollama fails: MedGemma-4B fallback.

### Phase 2 — Signal Intelligence (Hours 6–12) — Shiva + Rumon
Run Polar H10 BLE pipeline, verify 5-second and 30-second window outputs, test MQTT → EMQX → `live_patients` ChromaDB write, PhysioNet cohort Parquet ingestion into `patient_cohorts`.
**Go/No-Go at Hour 12:** Live vitals visible in Vercel UI dashboard, cohort match returns clinical metadata.

### Phase 3 — Agent Intelligence (Hours 12–18) — Viggi + Shiva
Update system prompts to optimal versions (above). Test Nurse Agent wellness framing. Test Clinical Assistant SOAP output. Test emergency keyword hardcoded bypass. Adversarial prompt injection tests.
**Go/No-Go at Hour 18:** Nurse Agent returns wellness-framed response. Clinical Assistant generates structured SOAP. Emergency keyword triggers hardcoded response only.

### Phase 4 — Flutter + Integration (Hours 18–24) — Sansrit + All
Flutter: patient chat (Screen 1) → multi-patient dashboard (Screen 2) → Doctor Chat (Screen 3).
**Hour 20 checkpoint:** Patient chat must show Nurse Agent response. Doctor Chat deferred to web UI if not ready.
Full end-to-end demo rehearsal with Maria + James + Ruth synthetic scenario.
**Go/No-Go at Hour 24:** Three-patient demo runs cleanly on DGX Spark.

---

## Demo Scenario

Three concurrent patients demonstrate every major system capability:

**Patient A — Maria Santos, 67, post-CABG, Hispanic, dual-eligible Medicare/Medicaid**
Normal session in Spanish-language mode. Nurse Agent: *"Buenos días, María. Tu sesión está en marcha y todo se ve estable."* Clean SOAP note at end. Demonstrates equity feature, happy-path documentation, Spanish-language guardrails.

**Patient B — James Park, 55, post-CABG, T2DM, prescribed 60–80% HRmax**
HR reaches 91% HRmax during exercise. `EnergySafeWindow` fires CRITICAL before any LLM call. Dashboard flashes alert in <1 second. Clinical Assistant SOAP: "WARNING: Exertion HR exceeded 90% age-predicted maximum. Assessment: T2DM intake flag — autonomic neuropathy may contribute to blunted HR regulation. Plan: Reduce treadmill intensity; monitor HRR at 1-minute mark." Demonstrates deterministic alert architecture + risk-aware clinical documentation.

**Patient C — Ruth Williams, 70, HFrEF, motion artifact during exercise**
Electrode contact degrades → SQI drops to 0.28 → ADVISORY alert. Nurse Agent: "Let me check on something — it looks like the sensor may have shifted slightly. The staff will take a quick look." Staff adjusts. SQI recovers to 0.91. SOAP note: "Signal quality reduced during minutes 8–11 (SQI 0.28) [low confidence — signal quality reduced]. HRV metrics from this interval excluded from trend analysis." Demonstrates SQI-conditioned confidence propagation — the system knows the limits of its own data.

**Clinician asks Doctor Chat:** "How does Maria's HR recovery compare to similar post-CABG patients in the reference cohort?"
→ Clinical Assistant retrieves PhysioNet cohort matches: "Matched 3 patients with similar characteristics (CABG, 65–70 years, EFS score 2–3). Median HRR at 1 min for cohort: 15 bpm. Maria today: 18 bpm — within normal range for matched cohort. Prior session (3/21): 16 bpm. Trend is stable. [Source: PhysioNet frailty cohort, match criteria: surgery_type=CABG, age_range=62-72]"

---

*Talk to Your Heart — On-Campus Cardiac Rehabilitation Intelligence*
*Vercel web demo (proof of concept) + NVIDIA DGX Spark (production deployment target)*
*Patient data never leaves the building.*
