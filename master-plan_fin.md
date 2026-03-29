# Talk to Your Heart
## On-Campus Cardiac Rehabilitation Intelligence — Proof of Concept

> **North Star:** Every supervised cardiac rehab session runs with an AI copilot that keeps patient data on campus, monitors every patient in real time, and generates clinical documentation automatically. Every improvement in clinic throughput, monitoring quality, and documentation speed directly translates into fewer deaths — because the 73% of eligible patients who never complete cardiac rehab are dying of a process failure, not a knowledge failure.

> **Current Stage:** Web-based proof of concept deployed on Vercel with Ollama inference tunneling, demonstrating the clinical intelligence workflow. Production target: NVIDIA DGX Spark edge deployment for hardware-enforced HIPAA compliance.

---

## 1. Problem Definition

Every year in the United States, hundreds of thousands of people survive a cardiac event only to face a second, quieter crisis: the rehabilitation that could keep them alive is failing them.

Cardiac rehabilitation (CR) is a **Class Ia recommended therapy** proven to reduce all-cause mortality by **13%** and hospitalizations by **31%** — yet only **24% of eligible Medicare beneficiaries** ever attend a single session, and fewer than **27% of those who start** complete the full 36-session program. Fewer than **1% of U.S. hospitals** meet the CMS Million Hearts target of 70% participation. The CMS Million Hearts initiative estimates closing this gap would save **25,000 lives** and prevent **180,000 hospitalizations** every single year.

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

---

## 2. Solution — Talk to Your Heart

**Talk to Your Heart** is an on-campus cardiac rehabilitation intelligence platform. The proof of concept runs as a web application backed by FastAPI and Ollama, with the production target being deployment on **NVIDIA DGX Spark** for hardware-enforced HIPAA compliance.

The system ingests live physiologic data from Polar H10 chest straps, performs signal processing producing interpretable clinical features including HRV metrics, ECG morphology, HAR activity classification, and signal quality indices, publishes structured patient state over MQTT, integrates Google Fit 7-day longitudinal baselines for between-session context, matches patients against a clinical reference cohort derived from the PhysioNet Wearable Exercise Frailty Dataset, and retrieves AHA/AACVPR guidelines via RAG to assemble rich context for two specialized AI agents.

### Two-Agent Architecture

After consulting with Dr. Amrutha (UCSF radiologist with AI expertise) and potential clinical users, we pivoted from multi-agent orchestration (removed the Duty Doctor as a separate cyclic agent) to a focused two-agent model where each agent operates independently with specialized system prompts and deep RAG integration:

**Nurse Agent (Patient-Facing Wellness Companion)**
Translates complex physiologic state into warm, understandable language. Provides patient education and encouragement calibrated to actual effort. Supports configurable language (including Spanish-language mode to directly address documented Hispanic participation disparities). Operates under strict wellness-framing guardrails: never diagnoses, never recommends medication changes, routes emergency-keyword input to hardcoded safety responses without LLM involvement.

**Clinical Assistant Agent (Clinician-Facing Intelligence)**
Powers the Doctor Chat Interface for targeted questions and generates structured session summaries and SOAP-note drafts grounded in measured physiologic data, patient intake context, reference cohort comparisons, and RAG-retrieved guidelines. Handles both interactive clinical Q&A and automated documentation — absorbing the Duty Doctor's SOAP generation responsibility into a single, more capable agent.

### Why Two Agents Instead of Three

The pivot from three agents to two was driven by user research:

1. **Clinician feedback:** Supervising clinicians want one intelligent interface, not two separate clinical agents to monitor. The Clinical Assistant handles both interactive queries ("How does James compare to similar patients?") and documentation (SOAP generation) in a single conversation context.
2. **Resource efficiency:** Running two LLM instances instead of three reduces memory pressure and simplifies orchestration while maintaining role separation where it matters — the patient/clinician boundary.
3. **Proof of concept clarity:** For the hackathon demo and external feedback (LinkedIn, ProductHunt), demonstrating two well-defined agents with clear role boundaries is more compelling than three agents with overlapping responsibilities.

### Core Design Principle

> **Deterministic pipelines handle sensing and safety.**
> **Retrieval systems handle memory and evidence.**
> **Agents handle communication and workflow.**

Critical clinical alerts are **always** generated by deterministic rule-based logic. The LLM never fires an alert — it only interprets one after it fires.

---

## 3. What Is Built (Proof of Concept)

### Implemented and Deployed

The following components are implemented in the repository and verifiably working:

#### Backend (`backend/`)
- **FastAPI application** (`main.py`, 271 lines) — serves the web UI, handles PDF uploads for RAG, routes queries through the agent orchestrator, provides live metrics polling, SOAP note generation
- **Agent Orchestrator** (`agent_orchestrator.py`, 60 lines) — role-based prompt assembly with deterministic safety injection for Nurse (patient) and Clinical Assistant (doctor) personas
- **Safety Engine** (`safety_engine.py`, 44 lines) — `EnergySafeWindow` implementing deterministic safety boundary checks: critical (HR > 90% HR_max), warning (HR > prescribed max), advisory (SQI < 0.5)
- **MQTT Pipeline** (`mqtt_pipeline.py`, 93 lines) — `VitalsPublisher` with QoS 0 for vitals and QoS 2 for critical alerts, `PulseForgeSubscriber` for Lead Orchestrator alert consumption
- **MQTT Subscriber** (`mqtt_subscriber.py`, 77 lines) — live EMQX broker subscription that upserts wearable telemetry directly into ChromaDB `live_patients` collection for real-time RAG context
- **Literature Ingestion** (`ingest_literature.py`, 104 lines) — offline PDF processing pipeline using PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`) embeddings via sentence-transformers, 1000-char chunks with 100-char overlap into ChromaDB `medical_docs` collection
- **Cohort Ingestion** (`ingest_cohorts.py`, 112 lines) — PhysioNet Wearable Exercise Frailty Dataset parquet ingestion into ChromaDB `patient_cohorts` collection with cosine similarity space, including HRV RMSSD, exercise labels, gait velocity metadata
- **Ollama Integration** (`utils.py`, 41 lines) — tunneled inference to local Ollama instance with ngrok/LocalTunnel bypass headers

#### Web Dashboard (`backend/static/`)
- **Clinical Reasoning Dashboard** (`index.html`, 210 lines + `app.js`, 21,740 lines + `style.css`, 21,743 lines) — glassmorphism-styled web UI with:
  - Role-based chat (Clinician / Patient toggle)
  - PDF upload for knowledge base ingestion with document library management
  - Live Polar H10 telemetry hero card (HR, HRV RMSSD, kinematic load) with Chart.js scrolling visualization
  - Real-time MQTT WebSocket streaming from EMQX broker
  - Model selection (MedGemma 27B / Llama 3.2)
  - SOAP note generation endpoint
  - PDF report generation

#### Signal Processing (`Application/Polar_Livestream-analysis-Python/`)
- **PyQt5 Desktop Dashboard** (`main.py` + `polar_ecg/`) — full Polar H10 BLE acquisition application with:
  - `processing_worker.py` (17,241 bytes) — dual-window signal processing: 5-second (SQI, instantaneous HR, HAR features) + 30-second (RMSSD, SDNN, LF/HF, DWT morphology delineation including P-width, QRS-width, ST-width, QT/QTc)
  - `ble_worker.py` (13,991 bytes) — Polar H10 BLE connection management with ECG (130 Hz), ACC (100 Hz), and HR stream handling
  - `mqtt_worker.py` (5,134 bytes) — QThread MQTT publisher with paho-mqtt v2, thread-safe queue
  - `google_fit_fetcher.py` (11,471 bytes) — Google Fit REST API integration for 7-day historical baselines (15-min HR bucketing, sleep stages, body temperature, steps, calories, heart points)
  - `har_inference.py` (8,080 bytes) — HAR fusion model inference for activity classification
  - `mock_sensor.py` (4,472 bytes) — synthetic ECG/ACC/HR generator for testing without hardware
  - `intake_form.py` (18,805 bytes) — clinical intake form UI combining patient demographics, cardiac history, comorbidities, medications, PHQ-2 depression screening
  - `dashboard.py` (37,127 bytes) — real-time visualization dashboard with ECG waveform, HRV metrics, activity classification, SQI display
  - `data_exporter.py` (6,338 bytes) — session data export utilities
  - `intake_state.json` (34,849 bytes) — combined clinical profile + Google Fit 7-day longitudinal data schema

#### Activity Recognition (`Act_Recoginition/`)
- **Dual-model HAR fusion** — ResNet1D (PAMAP2, 88.9% accuracy) + HARNet10 SSL pretrained on UK Biobank (PhysioNet, 73.3% accuracy), 1152-dim feature fusion, subject-wise 80/20 splits preventing sample leakage across 47 subjects
- 8-class unified activity classifier: sitting, walking, running, cycling, stair climbing, treadmill walking, timed up and go, nordic walking
- Trained model weights: `fusion_model_proper.pth` (3 MB), `harnet_physionet.pth` (44 MB), `model.pth` (1.8 MB)

#### ECG Foundation Model Feature Extraction (`ECG_Embedding/`)
- **ECG-FM pipeline** (`ECG_frailty-db_feature_lookup.py`, 739 lines) — complete end-to-end pipeline for:
  - Loading WFDB records from PhysioNet Wearable Exercise Frailty Dataset
  - ECG-FM (fairseq-signals) 768-dim embedding extraction with chunked GPU inference
  - 5-second ECG metrics: SQI (NeuroKit2 + Welch PSD QRS energy + vital_sqi kurtosis), instantaneous HR
  - 30-second HRV metrics: RMSSD, SDNN, mean HR, LF/HF (Lomb-Scargle), DWT morphology (P-width, QRS-width, ST-width, QT-width, QTc)
  - 5-second ACC HAR features: mean magnitude, variance/energy, spectral entropy, median frequency
  - Per-segment parquet generation joining ECG-FM embeddings + signal metrics + exercise annotations + patient clinical metadata

#### System Prompts (`system_prompts.md`)
- **Production-grade agent system prompts** (505 lines) with:
  - Complete data source schemas for all 5 ChromaDB collections
  - SQI-conditioned confidence reporting rules
  - RAG citation protocol ([Source: X, chunk N] for Clinical Assistant, invisible grounding for Nurse)
  - Emergency keyword response protocol
  - SOAP note generation format (Subjective/Objective/Assessment/Plan with SQI annotations)
  - Google Fit data interpretation guidelines
  - Missing data handling requirements
  - 4-layer safety architecture documentation

#### Infrastructure
- **Docker Compose** (`docker-compose.yml`) — multi-service deployment: nurse agent vLLM (Qwen2.5-72B-AWQ), clinical agent vLLM (MedGemma-27B-IT), Mosquitto MQTT broker
- **Vercel deployment** (`vercel.json`) — serverless hosting for web proof of concept
- **Live deployment:** https://pulse-forge-8ibmyxj7v-paudel54s-projects.vercel.app/

### Deployed via Ollama (Current Proof of Concept)

The current demo uses **MedGemma-27B** via Ollama with ngrok/LocalTunnel tunneling. This demonstrates the clinical intelligence workflow (RAG retrieval → context assembly → role-based response → safety guardrails) without requiring DGX Spark hardware. The architecture is designed for a clean migration path: replace the Ollama endpoint URL with vLLM endpoint URLs when deploying to DGX Spark.

---

## 4. Why Interpretable Clinical Metrics Over Foundation Model Embeddings

We made a deliberate architectural decision to center the system on **interpretable, deterministic clinical features** rather than opaque foundation model embeddings (CLEF, ECG-FM, etc.) for the patient-facing pipeline. The rationale is both clinical and engineering:

**Clinical interpretability.** Every metric our system produces — RMSSD, SDNN, LF/HF ratio, QRS width, HR recovery, MET estimate, signal quality index — has a direct mapping to published clinical literature and established cardiac rehab guidelines. When the Clinical Assistant tells a clinician "RMSSD dropped 40% during recovery compared to the patient's baseline," the clinician understands what that means, can validate it against their own training, and can act on it with confidence. An embedding distance of 0.73 from a latent space provides no such clinical grounding.

**Deterministic reproducibility.** Our signal processing pipeline produces identical outputs for identical inputs. This is essential for clinical documentation, audit trails, and regulatory positioning.

**Latency and reliability.** Deterministic feature computation processes each 5-second window in <10ms on CPU, leaving GPU resources entirely available for LLM agent inference.

**Clinical reference matching.** We match patients against the PhysioNet Wearable Exercise Frailty Dataset using interpretable clinical metadata — surgery type, age, gender, comorbidities, EFS frailty score, medication status — and physiologic feature profiles. This gives clinicians context grounded in published clinical data with known patient characteristics, not an opaque vector distance.

**Note on ECG-FM embeddings:** The ECG-FM pipeline (`ECG_Embedding/ECG_frailty-db_feature_lookup.py`) generates 768-dim embeddings and feeds them into the ChromaDB cohort matching system. These embeddings are used for **similarity retrieval in the reference cohort** (finding patients with similar ECG morphology patterns), not for direct clinical decision-making. The clinician-facing output always surfaces interpretable features, not raw embedding distances.

### Known Limitation: LF/HF Ratio on 30-Second Windows

The LF/HF ratio computed via Welch periodogram (or Lomb-Scargle in the batch pipeline) over 30-second RR intervals during exercise is methodologically contested. The Task Force of the European Society of Cardiology (1996) recommends minimum 2-minute windows for reliable frequency-domain HRV analysis, and exercise-induced non-stationarity violates the stationarity assumptions underlying spectral analysis. Our implementation acknowledges this:

- **Mitigation:** LF/HF values from 30-second exercise windows are reported with a confidence qualifier in the Clinical Assistant's SQI-conditioned output. The system prioritizes time-domain metrics (RMSSD, SDNN) which are more reliable at shorter window lengths.
- **Fallback:** If LF/HF metrics prove unreliable during active exercise phases, the system degrades gracefully — time-domain HRV and HR recovery metrics remain the primary indicators for clinical interpretation.
- **Production plan:** Extend to 2-minute overlapping windows for frequency-domain analysis when the 30-second constraint is relaxed in non-real-time contexts (e.g., post-session SOAP generation).

---

## 5. Architecture

### System Architecture Overview

```
+=================================================================+
|                     PROOF OF CONCEPT                            |
|     (Vercel + Ollama Tunneling → Target: DGX Spark)             |
|                                                                  |
|  +----------------+     +--------------------+                   |
|  |  Polar H10     | BLE |  Python Signal     |                   |
|  |  Chest Strap   |---->|  Processing        |                   |
|  |  ECG  (130 Hz) |     |  (processing_      |                   |
|  |  ACC  (100 Hz) |     |   worker.py)       |                   |
|  |  HR   (Live)   |     |  SQI + HRV + HAR   |                   |
|  +----------------+     +--------+-----------+                   |
|                                  |                               |
|  +----------------+              | MQTT pub                      |
|  | Google Fit API |     +--------v-----------+                   |
|  | (7-day hist)   |     |    MQTT Broker     |                   |
|  | HR, Steps,     |     |  (EMQX / Mosquitto)|                   |
|  | Sleep, Temp,   |     +--------+-----------+                   |
|  | Calories       |              | sub                           |
|  +-------+--------+    +--------v-----------+                    |
|          |              | mqtt_subscriber.py |                    |
|          |              | Upserts to ChromaDB|                    |
|  +-------v--------------+--------+----------+                    |
|  |              ChromaDB                      |                   |
|  |  medical_docs (RAG literature)             |                   |
|  |  patient_cohorts (ECG-FM + clinical meta)  |                   |
|  |  live_patients (real-time MQTT state)       |                   |
|  +---------------------+---------------------+                   |
|                         |                                        |
|  +---------------------v---------------------+                   |
|  | FastAPI + Agent Orchestrator               |                   |
|  |  → EnergySafeWindow (deterministic safety) |                   |
|  |  → RAG retrieval from ChromaDB             |                   |
|  |  → Cohort KNN similarity matching          |                   |
|  |  → Role-based prompt assembly              |                   |
|  +---+---------------------+-----------------+                   |
|      |                     |                                     |
|      v                     v                                     |
|  +---------+         +-----------+                               |
|  | Nurse   |         | Clinical  |                               |
|  | Agent   |         | Assistant |                               |
|  | (Ollama |         | (Ollama   |                               |
|  |  or     |         |  or       |                               |
|  |  vLLM)  |         |  vLLM)    |                               |
|  +----+----+         +-----+-----+                               |
|       |                    |                                     |
|       v                    v                                     |
|  Patient Chat         Doctor Chat                                |
|  + Education          + SOAP Notes                               |
|                       + Session Reports                          |
+=================================================================+
```

### Production Target: DGX Spark Memory Allocation (128 GB Unified LPDDR5x)

```
+=====================================================+
|         DGX SPARK MEMORY ALLOCATION                 |
|         (128 GB Unified LPDDR5x @ 273 GB/s)        |
|                                                     |
|   MedGemma-27B (INT4 quantized)     ~14 GB          |
|   Qwen3 (nurse agent variant)       ~ 8 GB          |
|   PubMedBERT embedding model        ~ 0.4 GB        |
|   ChromaDB indices + data           ~ 4 GB          |
|   Reference cohort feature store    ~ 1 GB          |
|   vLLM KV cache                     ~20 GB          |
|   MQTT broker + services            ~ 1 GB          |
|   Signal processing buffers         ~ 2 GB          |
|   OS + system overhead              ~ 8 GB          |
|   ----------------------------------------          |
|   TOTAL ESTIMATED                   ~58 GB          |
|   REMAINING HEADROOM                ~70 GB          |
+=====================================================+
```

The two-agent architecture significantly reduces memory pressure compared to the three-agent plan (82 GB → 58 GB), providing 70 GB of headroom for model upgrades, increased KV cache for higher patient concurrency, or deploying larger models as they become available.

### Why DGX Spark for Production

DGX Spark is the target production deployment platform because it uniquely enables:

1. **Unified memory** — zero-copy handoff between CPU signal processing and GPU model inference
2. **128 GB LPDDR5x** — sufficient for concurrent multi-model serving that discrete GPU systems cannot match
3. **Desktop form factor** — deployable in a clinical equipment closet without data center infrastructure
4. **HIPAA by architecture** — patient data never leaves the building, no BAAs required for cloud AI
5. **NVIDIA inference stack** — native vLLM support, NIM containers, pre-validated models

**Why not alternatives:**

| Alternative | Why It Fails |
|-------------|-------------|
| **Cloud (AWS/GCP/Azure)** | PHI in transit violates HIPAA architecture; 100–500ms latency |
| **RTX 4090 (24 GB VRAM)** | Cannot fit MedGemma-27B + Nurse agent simultaneously |
| **Apple M4 Ultra (192 GB)** | Memory but ~20x less GPU compute — multi-agent serving infeasible at clinical latency |

---

## 6. Five Implemented Innovations

**1. Interpretable dual-window physiologic intelligence for real-time rehab monitoring.**
Working implementation: 5-second window (SQI via NeuroKit2 + Welch PSD QRS energy + kurtosis, instantaneous HR, 4 HAR features) + 30-second window (RMSSD, SDNN, LF/HF via Lomb-Scargle, DWT morphology: P-width, QRS-width, ST-width, QT/QTc). Every feature maps directly to published cardiac rehabilitation literature.

**2. Clinical reference cohort matching using PhysioNet Wearable Exercise Frailty Dataset.**
Working implementation: ECG-FM 768-dim embeddings + clinical metadata (surgery type, age, EFS frailty score, comorbidities, gait/balance parameters, veloergometry outcomes) ingested into ChromaDB with cosine similarity retrieval. Clinicians see interpretable patient comparisons, not opaque distances.

**3. Multi-source RAG with role-separated agents on a single platform.**
Working implementation: PubMedBERT-embedded AHA/AACVPR guidelines in ChromaDB, live MQTT wearable telemetry synchronized into ChromaDB for real-time context, agent orchestrator with role-based prompt assembly and deterministic safety injection. Two agents with different system prompts, different audiences, and different guardrails operating from the same knowledge base.

**4. Signal-quality-aware AI interpretation.**
Working implementation: SQI scores (0.0–1.0) propagated through the pipeline into agent context. SQI computed from three metrics: NeuroKit2 template quality, QRS band energy ratio (Welch PSD 5–15 Hz / 1–40 Hz), and vital_sqi kurtosis. When sqi < 0.50, Safety Engine fires advisory alerts and the Clinical Assistant annotates metrics with "[LOW CONFIDENCE — Signal quality degraded]".

**5. Google Fit longitudinal baseline integration for between-session context.**
Working implementation: Google Fit REST API (`fitness.googleapis.com`) fetches 7-day historical baselines — 15-minute bucketed HR arrays, daily steps, calories, heart points, body temperature, sleep hours with stage segmentation (light/deep/REM/awake). Provides between-session context for the 4–5 days/week when the patient is not in the clinic.

---

## 7. Safety Architecture (4 Layers — All Implemented)

```
Layer 1: Energy Safe Window (deterministic, pre-LLM)
  → safety_engine.py — fires alerts based on:
    CRITICAL: HR > 90% age-predicted max during exercise
    WARNING:  HR > prescribed intensity ceiling during exercise
    ADVISORY: SQI < 0.50 (motion artifact)
  → Fires BEFORE any LLM invocation

Layer 2: Emergency Keyword Classifier (deterministic, pre-LLM)
  → agent_orchestrator.py — intercepts "chest pain", "can't breathe",
    "dizzy", "passing out", etc. before LLM processes
  → Returns hardcoded safety response without LLM involvement

Layer 3: Output Validator (deterministic, post-LLM)
  → Blocks Nurse output containing diagnostic language:
    "abnormal", "disease", "arrhythmia", "you have", etc.
  → Falls back to safe default response

Layer 4: System Prompt Guardrails (probabilistic, in-LLM)
  → system_prompts.md — role boundaries enforced via
    instruction following with banned word lists
```

---

## 8. Vercel Web Demo (Proof of Concept)

The deployed Vercel application demonstrates the complete clinical intelligence workflow:

**Live at:** https://pulse-forge-8ibmyxj7v-paudel54s-projects.vercel.app/

### What the Demo Shows

1. **PDF Knowledge Base Upload** — upload AHA/AACVPR guidelines or clinical papers, system chunks and embeds into ChromaDB for RAG retrieval
2. **Role-Based Chat** — toggle between Clinician (Clinical Assistant) and Patient (Nurse Agent) personas with different system prompts and guardrails
3. **Live Sensor Telemetry** — real-time Polar H10 HR, HRV, and activity status via MQTT WebSocket streaming from EMQX broker, visualized with Chart.js
4. **RAG-Grounded Responses** — clinical queries retrieve relevant guideline chunks and reference cohort matches, cited in responses
5. **SOAP Note Generation** — automated structured clinical documentation from the `/api/session/{patient_id}/soap` endpoint
6. **Deterministic Safety** — EnergySafeWindow evaluates telemetry before LLM invocation; safety status injected into every prompt

### Architecture Gap: Vercel vs. DGX Spark

The Vercel deployment uses Ollama tunneling for LLM inference, which means patient data transits through an ngrok/LocalTunnel connection. This is acceptable for a proof of concept / hackathon demo but **does not satisfy HIPAA requirements**. The DGX Spark production deployment eliminates this gap entirely — all inference runs on-campus with zero data egress.

| Aspect | Vercel PoC | DGX Spark Production |
|--------|-----------|---------------------|
| LLM hosting | Ollama via tunnel | vLLM on-device |
| Data transit | Through tunnel (not HIPAA) | Zero egress (HIPAA compliant) |
| Latency | 500ms–2s (tunnel overhead) | <5s end-to-end |
| Models | MedGemma-27B via Ollama | MedGemma-27B + Qwen3 via vLLM |
| MQTT | EMQX public broker | Mosquitto local broker |
| Scalability | Single patient demo | 6–8 concurrent patients |

---

## 9. User Experience

### Patient-Facing (Nurse Agent)

The patient wears a Polar H10 and interacts with the web chat interface. The Nurse Agent receives the full context (vitals, intake, Google Fit baseline, risk flags) but responds only in warm, accessible language:

```
WARM-UP:  "Good morning, Maria. Your session is underway and everything
           looks nice and steady."
EXERCISE: "You've been going for 12 minutes now and you're doing great!
           Your heart is working at a nice, steady pace."
EFFORT+:  "Your heart rate has climbed a bit higher than your usual
           target. You might want to ease back just a touch."
RECOVERY: "Wonderful job today! Your heart rate is settling down nicely.
           Every session like this makes a difference."
```

The patient never sees HRV numbers, morphology metrics, or clinical terminology. Spanish-language mode addresses documented 50% Hispanic participation gap.

### Clinician-Facing (Clinical Assistant)

```
+==========================================================================+
|              CLINICAL REASONING DASHBOARD                                 |
|   Upload clinical PDFs → Ask medical questions → Get grounded answers     |
|                                                                           |
|   [Live Telemetry: HR 118 bpm | HRV 24.8 ms | Exercise Phase]           |
|                                                                           |
|   Clinician: "How does this patient compare to similar post-CABG cases?" |
|                                                                           |
|   Clinical Assistant: "Compared to reference cohort patients with        |
|   similar profiles (post-CABG, age 55-65, n=5 matches from PhysioNet    |
|   Wearable Exercise Frailty Dataset), this patient's HR recovery of      |
|   14 bpm/min falls within the interquartile range [12-18 bpm/min].       |
|   [Source: AHA_CR_2024, chunk 12] AHA guidelines recommend monitoring    |
|   HR recovery trend over 3-5 sessions. SQI: 0.87 [high confidence]."    |
|                                                                           |
|   [Generate SOAP Note]  [Generate PDF Report]                             |
+==========================================================================+
```

---

## 10. Data Pipeline: PhysioNet Wearable Exercise Frailty Dataset

### Why This Dataset

The PhysioNet Wearable Exercise Frailty Dataset provides the ground truth reference cohort for our clinical matching system. It contains:

- **47 post-cardiac-surgery patients** with rich clinical metadata
- **ECG at 130 Hz** (same frequency as Polar H10) + **ACC at 100 Hz**
- **Exercise annotations** (rest, cycling, walking, timed-up-and-go, stair climbing)
- **Clinical outcomes:** 6MWT distance, TUG time, veloergometry, gait/balance parameters
- **Frailty assessment:** Edmonton Frail Scale (EFS) scores
- **Medications:** beta-blockers, ACE inhibitors, calcium channel blockers
- **Demographics + comorbidities:** heart failure NYHA, atrial fibrillation, COPD, depression

### Processing Pipeline (Implemented)

```
PhysioNet Dataset → WFDB loader → ECG-FM embeddings (768-dim)
                                 → 5s ECG metrics (SQI, HR)
                                 → 30s HRV metrics (RMSSD, SDNN, LF/HF, morphology)
                                 → 5s ACC HAR features (magnitude, variance, entropy, freq)
                                 → Exercise activity labels from .atr annotations
                                 → Join with subject-info.csv clinical metadata
                                 → Output: Parquet lookup table
                                 → Ingest into ChromaDB (cosine similarity)
```

---

## 11. Competitive Landscape

```
+============================================================================+
|                    Continuous  Interpretable  Multi-Agent  Auto    On-Prem  |
|   Competitor        ECG        Clinical AI    Role-Sep     SOAP    Edge     |
|   --------------------------------------------------------------------------
|   Fourth Frontier    YES        no             no          no      no       |
|   Recora             no         no             no          no      no       |
|   Biofourmis*        YES        proprietary    no          no      no       |
|   Movn Health        no         no             no          no      no       |
|   --------------------------------------------------------------------------
|   Talk to Your Heart YES        Interpretable  2 agents    YES     DGX      |
|                                 + Ref Cohort                       Spark    |
+============================================================================+
* Biofourmis (now General Informatics): cloud-dependent, no role separation,
  no SOAP, no edge. HIPAA-as-architecture cannot be retrofitted onto cloud-native.
```

Defensibility: integration complexity across interpretable signal pipeline + reference cohort matching + role-separated agents + edge deployment. 12–18 months to replicate. Privacy-preserving architecture is an offensive advantage, not just compliance.

---

## 12. Market Opportunity

**TAM:** $3.66B AI-driven cardiac platforms by 2030. **SAM:** $1.39B U.S. cardiac rehab programs. **SOM (Year 1–2):** $4.2M — 35 clinics at $120K ACV.

Revenue: hardware deployment ($40–60K) + annual license ($60–80K) + RPM billing ($99+/month/patient, ~$118,800/year per 100 patients). CMS permanently enabled virtual CR supervision in CY 2026.

### Reimbursement

```
+============================================================================+
|   CODE    DESCRIPTION                                      REIMBURSEMENT   |
|   93798   Outpatient CR with continuous ECG                 Per session     |
|   99453   Initial RPM setup + education                     $22 one-time   |
|   99454   Device supply + data transmission (16+ d/mo)      $47/month      |
|   99457   RPM management, first 20 min                      $52/month      |
|   99458   Each additional 20 min                            $41/month      |
|   99445   Device supply, 2–15 days (NEW 2026)               $52/month      |
|   99470   First 10 min management (NEW 2026)                $26/month      |
+============================================================================+
```

---

## 13. Regulatory Strategy

**Phase 1 (Now): FDA General Wellness — no 510(k) required**

FDA guidance (updated January 6, 2026) covers non-invasive physiologic sensors estimating HRV and recovery metrics. Our interpretable metrics strengthen regulatory position — every output maps to a published clinical measurement.

| Allowed Language | Prohibited Language |
|-----------------|-------------------|
| "Your HR dropped 18 bpm in the first minute" | "Your cardiac recovery is abnormal" |
| "Your session effort was in your target zone" | "This suggests reduced autonomic function" |

SOAP notes framed as administrative documentation — outside Software as a Medical Device scope.

**Phase 2 (12–18 months): FDA 510(k)**

Predicates: Hexoskin (Nov 2025), CardioTag (2025), Apple Watch ECG, AliveCor KardiaMobile.

---

## 14. Risks and Mitigations

| Risk | Mitigation | Status |
|------|-----------|--------|
| **Vercel PoC doesn't demonstrate HIPAA** | Explicitly framed as proof of concept; DGX Spark production architecture documented with zero-egress design | Acknowledged |
| **Single-lead ECG limitation** | Scoped to rhythm/rate/HRV/recovery validated for single-lead; wellness framing avoids 12-lead parity claims | Mitigated |
| **LLM hallucination** | 4-layer safety (all implemented); deterministic alerts independent of LLMs; output grounded in structured data + RAG | Mitigated |
| **LF/HF on 30-second windows** | Time-domain metrics (RMSSD, SDNN) prioritized; LF/HF reported with confidence qualifier; 2-min windows for post-session analysis | Acknowledged |
| **BLE instability (Polar H10)** | SQI detects degradation; mock sensor fallback (`mock_sensor.py`); graceful degradation | Mitigated |
| **Ollama tunnel latency** | Acceptable for PoC demo; vLLM on DGX Spark eliminates for production | Accepted for PoC |
| **Ethical — algorithmic bias** | Post-deployment monitoring tracks alert accuracy stratified by demographics; SQI-conditioned confidence | Planned |
| **Ethical — automation complacency** | SOAP notes require clinician "Review and Approve"; raw signals preserved alongside AI summaries | Designed |

---

## 15. Ecosystem and Extensibility

### Three-Tier API (Implemented)

**Tier 1: Real-Time (MQTT + WebSocket)**
- Live wearable telemetry: EMQX broker → MQTT WebSocket → web dashboard
- Per-patient topic isolation: `pulseforgeai/{subject_id}/raw`, `pulseforgeai/{subject_id}/info`
- QoS 2 for critical alerts

**Tier 2: REST FastAPI (Implemented)**
```
POST   /upload                          Upload clinical PDFs for RAG
GET    /documents                       List uploaded documents
DELETE /documents/{filename}            Remove document from knowledge base
POST   /query                           Role-based clinical query
GET    /api/live/metrics                Poll live MQTT sensor state
GET    /api/session/{patient_id}/soap   Generate SOAP note
```

**Tier 3: FHIR R4 Export (Designed)**
- Observation + DiagnosticReport resources for Epic/Cerner/Meditech integration
- Planned for production deployment

### ChromaDB Collections (Implemented)

| Collection | Content | Status |
|------------|---------|--------|
| `medical_docs` | AHA/AACVPR guidelines, clinical papers (PubMedBERT embedded) | ✅ Implemented |
| `patient_cohorts` | ECG-FM 768-dim embeddings + clinical metadata from PhysioNet | ✅ Implemented |
| `live_patients` | Real-time MQTT wearable telemetry for RAG context | ✅ Implemented |

---

## 16. Scalability Design

### Single-Device Capacity (DGX Spark Production Target)

Each DGX Spark runs a fully independent pipeline for 6–8 concurrent patients with 2 concurrent LLM agents, on-demand SOAP generation, and <5s response latency. Without foundation model inference on the GPU critical path, the full Blackwell GPU is available for LLM serving.

### Horizontal Scaling Path

```
+=========================================================+
|              SCALING ARCHITECTURE                       |
+=========================================================+
|  SINGLE CLINIC (Demo/MVP — Current Stage)               |
|  Vercel PoC with Ollama tunneling                       |
|  1 patient demo, proving clinical workflow              |
+---------------------------------------------------------+
|  SINGLE CLINIC (Production Target)                      |
|  1× DGX Spark                                          |
|  6-8 concurrent patients                                |
|  On-premise MQTT + ChromaDB                             |
|  vLLM serving MedGemma-27B + Qwen3                     |
+---------------------------------------------------------+
|  MULTI-CLINIC (Phase 2, 6-12 months)                    |
|  1× DGX Spark per clinic                               |
|  Central aggregate analytics (de-identified)             |
+---------------------------------------------------------+
|  HEALTH SYSTEM (Phase 3, 12-18 months)                  |
|  FHIR R4 export for EHR integration                     |
|  Federated learning without pooling PHI                 |
+=========================================================+
```

### Workflow Impact

```
+============================================================================+
|   METRIC                        BASELINE (MANUAL)    WITH SYSTEM           |
|   ---------------------------------------------------------------------------
|   SOAP note generation time     10-15 min/patient    1.5-2 min (review)   |
|   SOAP reduction                 —                    80-87%               |
|   Concurrent patients monitored  3-4 effectively      6-8+ with AI assist |
|   Alert response latency         Minutes (scanning)   <1 second            |
|   Documentation completeness     Variable (recall)    Structured (sensor)  |
+============================================================================+
```

---

## 17. Demo Scenario

### Primary Demo (Vercel Web Application)

**Step 1: Knowledge Base Setup**
Upload AHA/AACVPR cardiac rehabilitation guidelines PDF. System chunks, embeds with PubMedBERT, and stores in ChromaDB.

**Step 2: Live Telemetry**
Polar H10 streams ECG/ACC via MQTT to EMQX broker. Web dashboard shows real-time HR, HRV RMSSD, and activity phase.

**Step 3: Clinician Query (Clinical Assistant)**
"How does this patient's HR recovery compare to similar post-CABG patients in the reference cohort?"
→ Clinical Assistant retrieves PhysioNet cohort matches, cites AHA guidelines, reports with SQI confidence.

**Step 4: Patient Interaction (Nurse Agent)**
Switch to Patient role. "I feel a bit tired today."
→ Nurse Agent responds with warm encouragement, references between-session activity from Google Fit baseline, never uses clinical terminology.

**Step 5: Emergency Safety**
Patient types "I have chest pain"
→ Hardcoded safety response fires immediately without LLM: "Please stop exercising immediately and alert the nearest staff member."

**Step 6: SOAP Note Generation**
Generate automated SOAP note from session data — structured, SQI-annotated, RAG-cited clinical documentation.

---

## 18. Vision

**6 months:** Pilot 2–3 rehab programs on DGX Spark. 70% documentation time reduction. <5% alert FP rate. FHIR R4 EHR integration.

**12–18 months:** FDA 510(k). Daily wellness reporting. Expand reference cohort with pilot site data — each new site enriches the clinical comparison database.

**2–3 years:** Pulmonary rehab, neurological rehab, post-surgical recovery. Federated multi-site analytics — anonymized population metrics across DGX Sparks without PHI sharing.

---

## 19. Repository Structure

```
PulseForgeAI/
├── backend/                           ← FastAPI + Vercel web application
│   ├── main.py                        ← FastAPI server (271 lines)
│   ├── agent_orchestrator.py          ← Role-based prompt assembly
│   ├── safety_engine.py               ← EnergySafeWindow deterministic safety
│   ├── mqtt_pipeline.py               ← MQTT pub/sub + alert QoS
│   ├── mqtt_subscriber.py             ← EMQX → ChromaDB live state sync
│   ├── ingest_literature.py           ← PubMedBERT RAG ingestion
│   ├── ingest_cohorts.py              ← PhysioNet cohort → ChromaDB
│   ├── utils.py                       ← Ollama inference utilities
│   ├── vercel.json                    ← Vercel deployment config
│   └── static/                        ← Web dashboard (HTML/CSS/JS)
│       ├── index.html                 ← Clinical reasoning UI
│       ├── app.js                     ← Chat, upload, MQTT, charting
│       └── style.css                  ← Glassmorphism design system
│
├── Application/
│   └── Polar_Livestream-analysis-Python/  ← PyQt5 desktop dashboard
│       ├── main.py                    ← Application entry point
│       ├── intake_state.json          ← Clinical + Google Fit schema
│       └── polar_ecg/
│           ├── workers/
│           │   ├── processing_worker.py  ← Dual-window signal processing
│           │   ├── ble_worker.py      ← Polar H10 BLE management
│           │   └── mqtt_worker.py     ← QThread MQTT publisher
│           ├── ui/
│           │   ├── dashboard.py       ← Real-time visualization
│           │   └── intake_form.py     ← Clinical intake form
│           └── utils/
│               ├── google_fit_fetcher.py  ← Google Fit API integration
│               ├── har_inference.py   ← HAR fusion model inference
│               ├── mock_sensor.py     ← Synthetic data generator
│               ├── data_exporter.py   ← Session data export
│               └── ring_buffer.py     ← Efficient circular buffer
│
├── Act_Recoginition/                  ← HAR fusion model training
│   ├── Data_Preparation/              ← PAMAP2 + PhysioNet preprocessing
│   ├── Train_Model/                   ← ResNet1D + HARNet10 SSL
│   ├── Fusion_Model/                  ← 1152-dim feature fusion
│   ├── Prediction_Model/              ← Inference demo
│   ├── fusion_model_proper.pth        ← Trained fusion weights (3 MB)
│   ├── harnet_physionet.pth           ← SSL pretrained weights (44 MB)
│   └── model.pth                      ← ResNet1D weights (1.8 MB)
│
├── ECG_Embedding/
│   └── ECG_frailty-db_feature_lookup.py  ← ECG-FM + signal metrics pipeline
│
├── docker-compose.yml                 ← DGX Spark multi-service deployment
├── system_prompts.md                  ← Production agent system prompts
└── master-plan_fin.md                 ← This document
```

---

## 20. Team

**Rumon** — Hardware / Clinical Workflow. Biomedical systems, cardiac monitoring hardware, BLE protocols. Polar H10 integration, intake schema, clinical workflow mapping, HAR feature extraction.

**Viggi** — DGX Spark / Infrastructure. NVIDIA GPU computing, container orchestration, edge deployment. vLLM deployment, MQTT infrastructure, ChromaDB, RAG ingestion, Vercel deployment.

**Shiva** — AI / ML / Signal Processing. ML, NLP, medical AI, LLM guardrails. ECG pipeline (processing_worker.py), agent system prompts, Lead Orchestrator, reference cohort matcher.

**Sansrit** — Frontend / Web Application. Web development, real-time visualization. Clinical reasoning dashboard, MQTT WebSocket integration, Chart.js telemetry visualization.

---

*Talk to Your Heart — On-Campus Cardiac Rehabilitation Intelligence*
*Proof of Concept: Vercel Web Demo | Production Target: NVIDIA DGX Spark*
*HIPAA-compliant by architecture | No PHI leaves the building*
