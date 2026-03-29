# Talk to Your Heart
## Multi-Agent Edge Cardiac Rehab Copilot

Talk to Your Heart is an edge-native cardiac rehabilitation copilot that converts Polar H10 wearable data into structured physiological features, similarity-grounded context, and role-specific AI outputs for patients and clinicians.

This version uses:
- **MedGemma-27B** for the **Duty Doctor Agent**
- **MedGemma-27B** for the **Clinical Assistant Agent**
- **Qwen3** for the **Nurse Agent**

The design follows a strict rule:

> **Deterministic pipelines handle sensing and safety.**
> **Retrieval systems handle memory and evidence.**
> **Agents handle communication and workflow.**

---

## 1. System intent

The system is designed for cardiac rehab and recovery monitoring using a Polar H10 chest strap, local processing, retrieval-backed reasoning, and separate patient-facing and clinician-facing AI personas.

### Primary goals
- Turn raw ECG, HR, and accelerometer streams into clinically useful summaries.
- Give patients supportive, understandable feedback.
- Give clinicians structured, evidence-backed reports and chat-based support.
- Keep safety-critical logic outside the LLM.

### Product framing
This should be presented as a **wellness / rehab support platform** with educational and workflow-assistive outputs, not an autonomous diagnostic system.

---

## 2. Architecture overview

The whiteboard implies a six-layer architecture:

1. **Sensing layer**
2. **Signal and feature layer**
3. **Messaging and storage layer**
4. **Retrieval and evidence layer**
5. **Agent layer**
6. **User interface layer**

### End-to-end flow
`Polar H10 -> BLE -> Python acquisition/service layer -> JSON payloads -> MQTT -> databases + retrieval -> agents -> patient UI / doctor chat / reports`

---

## 3. Sensing layer

The source device is the **Polar H10**, streaming the following physiological channels:

- **ECG** at 130 Hz
- **Accelerometer** at approximately 100 Hz on the whiteboard
- **Live heart rate**
- **RR intervals** when available from Polar services

### Why this matters
This gives the system both cardiac rhythm information and activity context. That combination is essential because heart rate alone is too delayed and too ambiguous for rehab interpretation.

---

## 4. Signal and feature layer

A Python service receives BLE data and performs real-time preprocessing, visualization, and feature extraction.

### ECG-derived features
The whiteboard and prior notes together support the following outputs:

- HR
- HRV, including RMSSD and frequency-domain or related HRV summaries
- P/QRS/ST/QT-width style interval and morphology features
- Signal quality index
- Band-limited ECG processing, roughly 0.5–40 Hz
- ECG foundation-model embeddings
- Time stamping and rolling session state [file:1][file:2]

### Accelerometer-derived features
The whiteboard explicitly calls out:

- Dominant frequency
- Band energy
- Activity class [file:1]

These map well to the earlier HAR design using SMA, FFT-based gait detection, and movement intensity scoring. [file:1][file:2]

### Output contract
The feature layer should emit a structured JSON payload, not raw waveform text, to downstream systems.

Example:
```json
{
  "patient_id": "pt_001",
  "timestamp": "2026-03-28T20:37:00Z",
  "ecg": {
    "hr_bpm": 102,
    "rr_ms_mean": 588,
    "rmssd_ms": 24.8,
    "qrs_ms": 94,
    "qt_ms": 372,
    "st_slope": -0.03,
    "signal_quality": 0.81
  },
  "activity": {
    "activity_class": "walking",
    "dominant_frequency_hz": 1.9,
    "band_energy": 0.44
  },
  "derived": {
    "vo2_proxy": 6.8,
    "hrr_zone_pct": 61,
    "ecg_embedding_ready": true
  },
  "alerts": []
}
```

---

## 5. Messaging and storage

The whiteboard strongly suggests **MQTT** as the event bus between signal-processing services, databases, and agents. [file:1]

### Recommended MQTT topics
- `ttyh/raw/ecg`
- `ttyh/raw/acc`
- `ttyh/features/live`
- `ttyh/features/window_5s`
- `ttyh/features/window_30s`
- `ttyh/alerts`
- `ttyh/agent/nurse/input`
- `ttyh/agent/duty_doctor/input`
- `ttyh/agent/clinical_assistant/input`
- `ttyh/reports/soap`
- `ttyh/reports/session`

### Datastores
The whiteboard shows two primary stores:

- **ChromaDB**
  - Patient vitals database
  - Embedding-backed lookup tables
  - Exercise vs stress retrieval
  - Beat morphology embedding lookup
  - Literature RAG index [file:1]

- **Patient Intake Database**
  - Subject profile
  - Age, weight
  - Medications
  - Risk factors
  - Exercise restrictions / clinician instructions [file:1][file:2]

### Design note
Use MQTT for transport and orchestration, but persist canonical patient state in a proper database layer. ChromaDB should not be the only source of truth for all patient facts.

---

## 6. Retrieval and evidence layer

This layer gives the agents structured memory and evidence instead of forcing them to reason from raw data alone.

### 6.1 Physiological retrieval
Use embedding-backed lookup for:

- Exercise vs stress matching
- Beat morphology similarity
- Population-grounded ECG 