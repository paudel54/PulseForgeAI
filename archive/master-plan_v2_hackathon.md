# Our Project

## Problem

Every year in the United States, hundreds of thousands of people survive a heart attack or cardiac procedure only to face a second, quieter crisis: the rehabilitation that could keep them alive is failing them. Cardiac rehabilitation is a Class Ia recommended therapy proven to reduce all-cause mortality by 13% and hospitalizations by 31%, yet only 24% of eligible Medicare beneficiaries ever attend a single session, and barely 27% of those who start will finish the full 36-session program. Fewer than 1% of U.S. hospitals meet the CMS Million Hearts target of 70% participation. The gap between what cardiac rehab can do and what it actually delivers represents one of the largest preventable losses of life in modern cardiology — the CMS Million Hearts initiative estimates that closing this gap would save 25,000 lives and prevent 180,000 hospitalizations every single year.

The crisis runs deeper than access alone. Inside clinics that operate cardiac rehabilitation programs, the care model is stretched thin. A single supervising clinician may oversee six to ten patients exercising simultaneously, each wearing a heart rate monitor, each responding differently to exertion, each recovering at a different rate. The clinician is expected to watch every screen, catch every subtle shift in heart rate recovery, notice when a patient's effort drifts outside their prescription, and document it all — often by hand, often after the session, often from memory. Wearable sensors generate rich physiologic data during these sessions — continuous ECG, beat-to-beat heart rate variability, movement intensity, recovery dynamics — but that data flows into fragmented displays with no unified interpretation, no intelligent alerting beyond simple threshold alarms, and no automated documentation.

At the same time, cloud-based AI tools introduce HIPAA compliance exposure, latency risk, and operational dependency on third-party infrastructure. Patient physiologic data is protected health information. Streaming raw ECG waveforms to external services during a live rehab session creates compliance exposure that most clinics cannot accept. For a system that needs to respond in real time to a patient whose heart rate is not recovering as expected, a cloud round-trip is architecturally wrong.

Among 366,000 eligible Medicare fee-for-service beneficiaries studied, hospital-level variation in enrollment spans 10-fold — a patient's zip code determines whether they receive a therapy that could save their life. Hispanic and non-Hispanic Black patients participate at roughly half the rate of White patients (13% vs. 26%). Dual-eligible Medicare/Medicaid beneficiaries participate at just 6.9% versus 26.7% for non-dual-eligible. CR participants show 48 fewer subsequent inpatient hospitalizations per 1,000 beneficiaries per year and $1,005 lower Medicare expenditures per beneficiary per year. Every CR session is associated with a 1.8% lower incidence of 1-year cardiac readmission. The U.S. cardiac rehab market stands at $984 million, projected to $1.39 billion by 2030, with the AI-driven platform segment growing at 21.1% CAGR toward $3.66 billion — yet the core participation problem remains unsolved.

What cardiac rehabilitation needs is an intelligent system that lives inside the clinic, processes patient data without sending it offsite, supports multiple patients and multiple clinical roles simultaneously, and turns raw physiologic streams into actionable care intelligence in real time — a system where better software directly translates into fewer deaths.

## Solution

We are building **Talk to Your Heart**, an on-campus cardiac rehabilitation intelligence platform powered by **NVIDIA DGX Spark**. The north star: **every supervised cardiac rehab session in the U.S. runs with an AI copilot that keeps patient data on campus, monitors every patient in real time, and generates clinical documentation automatically.** The system is purpose-built for supervised clinical sessions where multiple patients exercise simultaneously and clinicians need live support, structured monitoring, and efficient documentation.

The platform ingests live physiologic data from Polar H10 chest straps over BLE, performs real-time signal processing on DGX Spark using a dual-window analysis pipeline (5-second and 30-second windows), publishes structured patient state over MQTT to a local Mosquitto broker, integrates a rich patient intake profile that combines clinical history (surgery type, LVEF, comorbidities, medications, PHQ-2 depression screening) with **7-day Google Fit longitudinal baselines** (15-minute HR bucketing, daily steps, calories, heart points, body temperature trends, and sleep stage data) — providing between-session context even when the patient is not wearing the chest belt, matches patient profiles against a clinical reference cohort derived from the PhysioNet Wearable Exercise Frailty Dataset, augments clinical reasoning with RAG-retrieved AHA/AACVPR guidelines, and routes structured context into a two-agent AI workflow for patient education and clinician decision support. The entire pipeline — from raw ECG waveform to structured clinical summary — runs on campus. No patient data leaves the building. Zero data egress.

### Why Interpretable Clinical Metrics Over Foundation Model Embeddings

We made a deliberate architectural decision to center the system on **interpretable, deterministic clinical features** rather than opaque foundation model embeddings (CLEF, ECG-FM, etc.). The rationale is both clinical and engineering:

**Clinical interpretability.** Every metric our system produces — RMSSD, SDNN, LF/HF ratio, QRS width, HR recovery, MET estimate, signal quality index — has a direct mapping to published clinical literature and established cardiac rehab guidelines. When the Clinical Assistant Agent tells a clinician "RMSSD dropped 40% during recovery compared to the patient's baseline," the clinician understands what that means, can validate it against their own training, and can act on it with confidence. An embedding distance of 0.73 from a latent space provides no such clinical grounding. In a safety-critical clinical environment, the ability for a human expert to verify, interpret, and override any AI output is not optional — it is the foundation of clinical trust.

**Deterministic reproducibility.** Our signal processing pipeline produces identical outputs for identical inputs. This is essential for clinical documentation, audit trails, and regulatory positioning. Foundation model embeddings introduce stochastic inference, model version dependencies, and opaque representation spaces that complicate reproducibility and regulatory review.

**Latency and reliability.** Foundation model inference on continuous ECG streams adds 50–200ms per window per model. In a multi-patient real-time monitoring system, this latency compounds — three foundation models across eight patients creates queuing pressure that degrades the <5s response target for interactive agents. Our deterministic pipeline processes each 5-second window in <10ms on CPU, leaving GPU resources entirely available for LLM agent inference where they matter most.

**Clinical reference matching.** Instead of embedding similarity in a latent space, we match patients against the PhysioNet Wearable Exercise Frailty Dataset using interpretable clinical metadata — surgery type, age, gender, comorbidities, EFS frailty score, medication status — and physiologic feature profiles. This gives clinicians context grounded in published clinical data with known patient characteristics, not an opaque vector distance.

The system deploys two role-specific AI agents coordinated by a Lead Agent Orchestrator:

**Nurse Agent (Qwen3)** — the patient-facing communication layer. Translates complex physiologic state into warm, understandable language. Provides patient education and encouragement calibrated to actual effort. Supports configurable language (including Spanish-language mode to directly address documented Hispanic participation disparities). Operates under strict wellness-framing guardrails: never diagnoses, never recommends medication changes, routes emergency-keyword input to hardcoded safety responses without LLM involvement.

**Clinical Assistant Agent (MedGemma-27B)** — the clinician-facing interactive reasoning and documentation layer. Powers the Doctor Chat Interface for targeted questions: What changed in this patient's HRV? Is recovery slower than baseline? How does this patient compare to similar post-CABG patients in the reference cohort? Generate a session summary for charting. Produces structured session summaries and SOAP-note drafts grounded in measured physiologic data, patient intake context, reference cohort comparisons, and RAG-retrieved guidelines. Transforms the rehab session into a searchable, interpretable clinical record.

The safety architecture enforces a hard boundary: every critical clinical alert is generated by deterministic rule-based logic, never by an LLM. Four layers of guardrails prevent diagnostic language from reaching patients.

The utilitarian case is direct — if closing the rehab participation gap saves 25,000 lives annually, then every improvement in clinic throughput, monitoring quality, and documentation speed contributes to that number. Our system targets quantifiable impact: 80–87% reduction in session documentation time, 33%+ increase in concurrent monitoring capacity per clinician, and orders-of-magnitude improvement in alert response latency.

## Why DGX Spark

DGX Spark is not interchangeable compute. It is the only commercially available device that makes this system architecturally possible.

### Unified Memory Makes Multi-Agent Edge AI Viable

The system must run signal processing, multiple LLMs, vector retrieval, and a message broker simultaneously as concurrent real-time services:

```
+=====================================================+
|         DGX SPARK MEMORY ALLOCATION                 |
|         (128 GB Unified LPDDR5x @ 273 GB/s)        |
|                                                     |
|   Qwen2.5-72B-Instruct-AWQ (INT4)  ~37 GB          |
|   MedGemma-27B (INT4 quantized)     ~14 GB          |
|   Qwen3 (smaller variant)           ~ 8 GB          |
|   PubMedBERT embedding model        ~ 0.4 GB        |
|   ChromaDB indices + data           ~ 4 GB          |
|   Reference cohort feature store    ~ 1 GB          |
|   vLLM KV cache                     ~24 GB          |
|   MQTT broker + services            ~ 1 GB          |
|   Signal processing buffers         ~ 2 GB          |
|   OS + system overhead              ~ 8 GB          |
|   ----------------------------------------          |
|   TOTAL ESTIMATED                   ~99 GB          |
|   REMAINING HEADROOM                ~29 GB          |
+=====================================================+
```

Unified coherent memory means zero-copy handoff between CPU signal processing (NeuroKit2, SciPy on ARM Grace cores) and GPU model inference (vLLM on Blackwell). On discrete-GPU systems, this requires explicit PCIe transfers adding latency to every inference call. By eliminating foundation model inference from the critical path, we free GPU headroom entirely for LLM agent serving — the workload where GPU compute actually determines user experience quality.

### HIPAA Compliance Through Architecture

Cloud AI requires BAAs, encryption certificates, vendor audits, and ongoing compliance monitoring for every third-party service touching PHI. DGX Spark eliminates all three HIPAA risk vectors — data in transit, data at rest on third-party infrastructure, and vendor access — by design. Patient data never leaves the building. This is hardware-enforced data locality, not policy-layer compliance — a fundamentally stronger position that competitors cannot match without infrastructure redesign.

### Why Not Alternatives?

**Cloud (AWS/GCP/Azure):** Fails HIPAA architecture, adds 100-500ms network latency, creates cost dependency. **Consumer GPU (RTX 4090, 24GB):** Cannot fit even the 37GB primary LLM. **Professional GPU (A6000 48GB):** Fits one LLM but not clinical model + retrieval + KV cache simultaneously. **Apple Silicon (M4 Ultra, 192GB):** Has memory but ~20x less GPU compute than Blackwell — multi-agent serving infeasible at clinical latency.

DGX Spark is the only device combining 128 GB unified memory, Blackwell-class compute (1 PFLOP FP4), desktop form factor deployable in a clinical equipment closet, and the NVIDIA inference stack (vLLM, TensorRT-LLM) in a single package.

## Innovation

Talk to Your Heart represents the first convergence of four capabilities never combined for cardiac rehabilitation:

**1. Interpretable dual-window physiologic intelligence for real-time rehab monitoring.** Our pipeline extracts clinically grounded features at two temporal resolutions simultaneously: a 5-second window producing ECG signal quality, instantaneous HR, and the four most discriminative chest-belt HAR features (mean magnitude, variance/energy, spectral entropy, median frequency), and a 30-second window producing RMSSD, SDNN, LF/HF ratio, and DWT-based morphology widths (QRS, QT, ST deviation). Every feature maps directly to published cardiac rehabilitation literature and clinical decision frameworks. This is not a black-box score — it is a structured, verifiable, clinician-interpretable patient state representation that can be audited, challenged, and overridden.

**2. Clinical reference cohort matching + longitudinal baseline context.** Instead of opaque embedding similarity, we match incoming patients against a reference cohort of post-surgical cardiac patients with known clinical characteristics — age, gender, surgery type (CABG, valve replacement, etc.), EFS frailty score, comorbidities, medication status, 6MWT distance, TUG time, veloergometry measures, and gait/balance parameters. Additionally, the system integrates **Google Fit 7-day longitudinal baselines** — 15-minute-bucketed HR trends, daily steps, calories, heart points, body temperature, and sleep stage data — providing between-session physiologic context that captures what happened on the 4–5 days per week when the patient is not in the clinic. This combination of reference cohort comparison and longitudinal trending gives agents context grounded in both published clinical outcomes and the individual patient's daily reality.

**3. Multi-agent clinical AI with role separation on edge hardware.** Two specialized agents serving different clinical roles with different output requirements — patient education (Nurse) and clinician decision support with documentation (Clinical Assistant) — running concurrently on a single on-premise device. No existing cardiac rehab system deploys role-separated agents on edge.

**4. Signal-quality-aware AI interpretation.** Per-segment SQI scores (0.0–1.0) propagated through the entire pipeline into agent context. When the Clinical Assistant reviews a session, it knows whether an HRV drop occurred during clean signal (clinically meaningful) or motion artifact (likely spurious). This SQI-conditioned confidence propagation is a capability no competitor offers.

**5. Automated clinical documentation grounded in deterministic physiologic features + reference cohort + retrieved evidence.** Session summaries and SOAP-note drafts generated from the structured output of the deterministic signal pipeline, augmented by reference cohort comparisons, patient intake context, and RAG-retrieved AHA/AACVPR guidelines — producing documentation grounded in measured physiology rather than clinician recall. Existing documentation tools (Nuance DAX, Abridge, Amazon HealthScribe, Epic SmartPhrases) are speech-to-text or template-filling; ours is sensor-grounded.

## Architecture

### System Architecture Overview

```
+===========================================================================+
|                          NVIDIA DGX Spark                                 |
|                  (128 GB Unified LPDDR5x / GB10 Grace Blackwell)          |
|                                                                           |
|  +----------------+     +--------------------+    +--------------------+  |
|  |  Polar H10     | BLE |  Dual-Window       | pub|    MQTT Broker     |  |
|  |  Chest Strap   |---->|  Signal Processing |---->|  (Mosquitto Local) |  |
|  |  ECG  (130 Hz) |     |                    |    | patient/{id}/vitals|  |
|  |  ACC  (100 Hz) |     |  5s window:        |    | patient/{id}/alerts|  |
|  |  HR   (Live)   |     |   SQI, HR, HAR     |    +--------+-----------+  |
|  +----------------+     |  30s window:        |         sub |   sub       |
|                         |   HRV, Morphology   |             v     v       |
|  +----------------+     |  Energy Safe Window  |    +------------------+  |
|  | Google Fit API |     +--------+-------------+    | Lead Orchestrator|  |
|  | (7-day hist)   |             |                   | (Deterministic)  |  |
|  | HR, Steps,     |             |                   +---+----------+--+   |
|  | Sleep, Temp,   |    Intake State JSON                |          |      |
|  | Calories       |--------+                            |          |      |
|  +----------------+        |                            v          v      |
|                            v                        +------+  +--------+ |
|  +-------------------------------+-----------+      |Nurse |  |Clinical| |
|  |              ChromaDB (5 collections)      |     |Agent |  |Asst    | |
|  |  patient_vitals_db                         |     |Qwen3 |  |Med-    | |
|  |  reference_cohort_features                 |     |      |  |Gemma   | |
|  |  reference_cohort_metadata                 |     |      |  |27B     | |
|  |  rag_medical_literature                    |     +--+---+  +---+----+ |
|  |  patient_intake_db (clinical + Google Fit) |        |          |      |
|  +--------------------------------------------+        v          v      |
|                                                   Patient     Doctor     |
|  +-------------------------------+                Chat Bot    Chat UI    |
|  | Reference Cohort Matcher      |                + Education + SOAP     |
|  | - Clinical metadata similarity|                                       |
|  | - Feature profile comparison  |                                       |
|  +-------------------------------+                                       |
+===========================================================================+
```

### End-to-End Data Flow

```
+-------------+      +--------------------+     +------------+     +------------------+
|  Polar H10  | BLE  | Dual-Window Signal | pub |   MQTT     | sub | Lead Orchestrator|
|  Chest      |----->| Processing         |---->|  Broker    |---->| (Deterministic)  |
|  Strap      |      | (5s + 30s)         |     | (Mosquitto)|     |                  |
+-------------+      +--------------------+     +------------+     +-------+----------+
                              |                                            |
                      +-------v--------+                         +---------v----------+
                      | Energy Safe    |                         | Context Assembly    |
                      | Window Check   |                         | (7 sources)         |
                      | (Deterministic)|                         +---------+----------+
                      +-------+--------+                                   |
                              |                                    +-------+-------+
                       Alert if outside                            |               |
                       safe exercise zone                          v               v
                              |                               +------+      +--------+
                              v                               |Nurse |      |Clinical|
                      +----------------+                      |Agent |      |Asst    |
                      | Alert Rules    |                      +--+---+      +---+----+
                      | Engine         |                         |              |
                      | (NO LLM)       |                         v              v
                      +-------+--------+                   Patient Chat   Doctor Chat
                              |                            + Education    + SOAP Notes
                              v                                          + Summaries
                      +----------------+
                      | Clinician      |
                      | Alert Panel    |
                      +----------------+
```

### vLLM Deployment on DGX Spark

```bash
# Primary LLM: Qwen2.5-72B-Instruct-AWQ (patient + clinician agents)
docker run --gpus all -v /models:/models -p 8000:8000 \
  nvcr.io/nvidia/vllm:latest \
  --model /models/Qwen2.5-72B-Instruct-AWQ \
  --quantization awq --max-model-len 32768 \
  --gpu-memory-utilization 0.45 --enable-prefix-caching \
  --max-num-seqs 8 --tensor-parallel-size 1

# Clinical LLM: MedGemma-27B (clinician assistant + documentation)
docker run --gpus all -v /models:/models -p 8001:8001 \
  nvcr.io/nvidia/vllm:latest \
  --model /models/MedGemma-27B-IT \
  --quantization awq --max-model-len 8192 \
  --gpu-memory-utilization 0.15 --port 8001
```

### Dual-Window Signal Processing Pipeline

```python
import neurokit2 as nk
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import entropy as sp_entropy
from dataclasses import dataclass, asdict

@dataclass
class WindowResult5s:
    """5-second window: SQI + instant HR + ACC HAR features."""
    patient_id: str
    timestamp: float
    hr_bpm: float
    sqi: float
    # Top 4 most discriminative chest-belt HAR features
    mean_mag_mg: float       # signal magnitude (overall activity intensity)
    var_mag_mg2: float       # variance/energy (high=active, low=sedentary)
    spectral_entropy: float  # low=periodic walk/cycle, high=noise/rest
    median_freq_hz: float    # walking ~1-2 Hz, cycling higher, sitting ~0
    activity_class: str      # rest|warmup|exercise|cooldown|recovery
    energy_safe: bool
    alert_level: str

@dataclass
class HRVResult30s:
    """30-second window: HRV + ECG morphology widths."""
    patient_id: str
    timestamp: float
    hrv_rmssd_ms: float
    hrv_sdnn_ms: float
    hrv_pnn50: float
    hrv_lf_hf_ratio: float
    qrs_width_ms: float
    qt_interval_ms: float
    st_deviation_mv: float
    met_estimate: float

class ECGProcessor:
    def __init__(self, fs=130):
        self.fs = fs

    def preprocess(self, raw_ecg: np.ndarray) -> np.ndarray:
        b, a = butter(4, [0.5, 40], btype='bandpass', fs=self.fs)
        filtered = filtfilt(b, a, raw_ecg)
        b_notch, a_notch = iirnotch(60.0, 30.0, self.fs)
        return filtfilt(b_notch, a_notch, filtered)

    def detect_r_peaks(self, clean_ecg: np.ndarray) -> np.ndarray:
        _, info_pt = nk.ecg_peaks(clean_ecg, sampling_rate=self.fs,
                                   method="pantompkins1985")
        _, info_hm = nk.ecg_peaks(clean_ecg, sampling_rate=self.fs,
                                   method="hamilton2002")
        peaks_pt = set(info_pt["ECG_R_Peaks"])
        peaks_hm = set(info_hm["ECG_R_Peaks"])
        tolerance = int(0.050 * self.fs)  # 50ms consensus tolerance
        consensus = []
        for p in peaks_pt:
            if any(abs(p - h) <= tolerance for h in peaks_hm):
                consensus.append(p)
        return np.array(sorted(consensus))

    def compute_hrv_30s(self, r_peaks: np.ndarray) -> dict:
        """30-second window HRV: RMSSD, SDNN, pNN50, LF/HF."""
        rr = np.diff(r_peaks) / self.fs * 1000  # ms
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) < 5:
            return {"rmssd": 0, "sdnn": 0, "pnn50": 0, "lf_hf": 0}
        diff_rr = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diff_rr**2)))
        sdnn = float(np.std(rr, ddof=1))
        pnn50 = float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100)
        freqs, psd = welch(rr, fs=1000/np.mean(rr), nperseg=min(256, len(rr)))
        lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)])
        hf = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.40)])
        lf_hf = float(lf / hf) if hf > 0 else 0.0
        return {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50, "lf_hf": lf_hf}

    def delineate_morphology(self, clean_ecg: np.ndarray,
                              r_peaks: np.ndarray) -> dict:
        """DWT-based morphological delineation: QRS width, QT, ST deviation."""
        try:
            _, waves = nk.ecg_delineate(clean_ecg, r_peaks,
                                         sampling_rate=self.fs, method="dwt")
            qrs_onsets = waves.get("ECG_Q_Peaks", [])
            qrs_offsets = waves.get("ECG_S_Peaks", [])
            t_offsets = waves.get("ECG_T_Offsets", [])
            qrs_widths = []
            for on, off in zip(qrs_onsets, qrs_offsets):
                if on is not None and off is not None and not (
                    np.isnan(on) or np.isnan(off)):
                    qrs_widths.append((off - on) / self.fs * 1000)
            qrs_width = float(np.median(qrs_widths)) if qrs_widths else 0.0
            qt_intervals = []
            for on, t_off in zip(qrs_onsets, t_offsets):
                if on is not None and t_off is not None and not (
                    np.isnan(on) or np.isnan(t_off)):
                    qt_intervals.append((t_off - on) / self.fs * 1000)
            qt_interval = float(np.median(qt_intervals)) if qt_intervals else 0.0
            st_devs = []
            j_offset = int(0.080 * self.fs)
            for s_peak in qrs_offsets:
                if s_peak is not None and not np.isnan(s_peak):
                    st_idx = int(s_peak) + j_offset
                    if st_idx < len(clean_ecg):
                        st_devs.append(clean_ecg[st_idx])
            st_dev = float(np.median(st_devs)) if st_devs else 0.0
            return {"qrs_width_ms": qrs_width, "qt_interval_ms": qt_interval,
                    "st_deviation_mv": st_dev}
        except Exception:
            return {"qrs_width_ms": 0.0, "qt_interval_ms": 0.0,
                    "st_deviation_mv": 0.0}

    def compute_sqi(self, clean_ecg: np.ndarray, r_peaks: np.ndarray,
                    acc_magnitude: np.ndarray) -> float:
        """Three-metric SQI: template matching (0.4) + SNR (0.3) + motion (0.3)."""
        if len(r_peaks) < 3:
            return 0.0
        beats = []
        win = int(0.3 * self.fs)
        for p in r_peaks[1:-1]:
            if p - win >= 0 and p + win < len(clean_ecg):
                beats.append(clean_ecg[p-win:p+win])
        if not beats:
            return 0.0
        template = np.median(beats, axis=0)
        corrs = [np.corrcoef(b, template)[0,1] for b in beats]
        template_score = float(np.mean(corrs))
        signal_power = np.var(template)
        noise_power = np.mean([np.var(b - template) for b in beats])
        snr_score = min(signal_power / (noise_power + 1e-10), 10) / 10
        acc_len = min(len(clean_ecg), len(acc_magnitude))
        motion_score = 1.0 - min(abs(np.corrcoef(
            clean_ecg[:acc_len], acc_magnitude[:acc_len])[0,1]), 1.0)
        return float(np.clip(
            0.4 * template_score + 0.3 * snr_score + 0.3 * motion_score,
            0.0, 1.0))


class AccelerometerHAR:
    """Top 4 most discriminative chest-belt HAR features."""
    def __init__(self, fs=100):
        self.fs = fs

    def extract_features(self, acc_xyz: np.ndarray) -> dict:
        magnitude = np.sqrt(np.sum(acc_xyz**2, axis=1))
        fft_vals = np.abs(np.fft.rfft(magnitude - np.mean(magnitude)))
        freqs = np.fft.rfftfreq(len(magnitude), 1/self.fs)
        mask = (freqs > 0.5) & (freqs < 10)

        # 1. Mean magnitude (overall activity intensity, in mg)
        mean_mag = float(np.mean(magnitude))

        # 2. Variance / energy (high=active, low=sedentary)
        var_mag = float(np.var(magnitude))

        # 3. Spectral entropy (low=periodic walk/cycle, high=noise/rest)
        if np.any(mask) and np.sum(fft_vals[mask]) > 0:
            psd_norm = fft_vals[mask] / np.sum(fft_vals[mask])
            spec_entropy = float(sp_entropy(psd_norm + 1e-12))
        else:
            spec_entropy = 0.0

        # 4. Median frequency (walking ~1-2 Hz, cycling higher, sitting ~0)
        if np.any(mask) and np.sum(fft_vals[mask]) > 0:
            cumsum = np.cumsum(fft_vals[mask]**2)
            median_freq = float(freqs[mask][np.searchsorted(
                cumsum, cumsum[-1] / 2)])
        else:
            median_freq = 0.0

        return {
            "mean_mag_mg": mean_mag,
            "var_mag_mg2": var_mag,
            "spectral_entropy": spec_entropy,
            "median_freq_hz": median_freq,
            "magnitude": magnitude
        }

    def classify_activity(self, features: dict, hr_bpm: float) -> str:
        """Rule-based activity classification with Markov transition debouncing."""
        var = features["var_mag_mg2"]
        median_f = features["median_freq_hz"]
        if var < 50 and hr_bpm < 80: return "rest"
        elif var < 200 and hr_bpm < 100: return "warmup"
        elif median_f > 0.8 and hr_bpm > 100: return "exercise"
        elif var < 200 and hr_bpm > 90: return "cooldown"
        else: return "recovery"

    def estimate_mets(self, hr_bpm: float, var_mag: float,
                      resting_hr: float, age: int) -> float:
        """Swain 2000: %VO2R ≈ %HRR, weighted with accelerometer energy."""
        hr_max = 220 - age
        hrr_frac = np.clip((hr_bpm - resting_hr) / (hr_max - resting_hr + 1e-6), 0, 1)
        acc_intensity = min(var_mag / 5000, 1.0)
        combined = 0.7 * hrr_frac + 0.3 * acc_intensity
        return float(np.clip(1.0 + combined * 12.0, 1.0, 15.0))
```

### Energy Safe Window

```python
class EnergySafeWindow:
    def __init__(self, patient_intake: dict):
        self.age = patient_intake["age"]
        self.hr_max = 220 - self.age
        self.prescribed_low = patient_intake["prescribed_intensity_range"][0]
        self.prescribed_high = patient_intake["prescribed_intensity_range"][1]

    def check(self, hr_bpm: float, activity: str, sqi: float,
              hr_recovery_1min: float = None) -> tuple[bool, str]:
        if activity == "exercise" and hr_bpm > 0.90 * self.hr_max:
            return False, "critical"
        if activity == "exercise" and hr_bpm > self.prescribed_high * self.hr_max:
            return False, "warning"
        if activity == "recovery" and hr_recovery_1min is not None:
            if hr_recovery_1min < 12:
                return True, "warning"
        if sqi < 0.5:
            return True, "advisory"
        return True, "none"
```

### MQTT Architecture

```python
import paho.mqtt.client as mqtt
import json

class VitalsPublisher:
    def __init__(self, broker_host="localhost", port=1883):
        self.client = mqtt.Client(protocol=mqtt.MQTTv5)
        self.client.connect(broker_host, port)  # LOCAL Mosquitto on DGX Spark

    def publish_window_5s(self, result: WindowResult5s):
        payload = json.dumps(asdict(result))
        self.client.publish(f"patient/{result.patient_id}/vitals/5s", payload, qos=0)
        if result.alert_level in ("warning", "critical"):
            self.client.publish(f"patient/{result.patient_id}/alerts", payload, qos=2)

    def publish_hrv_30s(self, result: HRVResult30s):
        payload = json.dumps(asdict(result))
        self.client.publish(f"patient/{result.patient_id}/vitals/30s", payload, qos=0)
```

### PhysioNet Reference Cohort and Patient Context Matching

```python
import pandas as pd

class ReferenceCohortMatcher:
    """Match patients against PhysioNet Wearable Exercise Frailty Dataset
    using interpretable clinical metadata — not opaque embeddings.

    Reference cohort contains per-patient:
    - Demographics: age, gender, height, weight
    - Clinical: surgery_type, days_after_surgery, EFS_score, comorbidities,
      hr_altering_medications
    - Exercise test outcomes: 6MWT_distance, TUG_time, veloergometry measures,
      gait/balance parameters (Zebris)
    - Physiologic feature profiles: HRV baselines, HR recovery curves,
      exercise response patterns
    """

    def __init__(self, cohort_csv: str):
        self.cohort = pd.read_csv(cohort_csv)
        # Pre-compute normalized feature vectors for similarity
        self.feature_cols = ["age", "EFS_score", "6MWT_distance_m",
                             "TUG_time_s", "resting_hr", "peak_hr_exercise"]
        self.cohort_normalized = self._normalize(self.cohort[self.feature_cols])

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.mean()) / (df.std() + 1e-8)

    def find_similar_patients(self, patient_intake: dict,
                               current_vitals: dict, n=5) -> list[dict]:
        """Find n most similar patients from reference cohort using
        clinical metadata + physiologic feature similarity."""
        # Filter by surgery type first (categorical match)
        filtered = self.cohort
        if patient_intake.get("surgery_type"):
            match = filtered[filtered["surgery_type"] == patient_intake["surgery_type"]]
            if len(match) >= 3:  # only filter if sufficient matches
                filtered = match

        # Score by clinical feature similarity (Euclidean on normalized features)
        patient_vec = pd.Series({
            "age": patient_intake["age"],
            "EFS_score": patient_intake.get("efs_score", 3),
            "6MWT_distance_m": patient_intake.get("6mwt_distance", 300),
            "TUG_time_s": patient_intake.get("tug_time", 12),
            "resting_hr": current_vitals.get("resting_hr", 70),
            "peak_hr_exercise": current_vitals.get("hr_bpm", 100)
        })
        patient_norm = (patient_vec - self.cohort[self.feature_cols].mean()) / (
            self.cohort[self.feature_cols].std() + 1e-8)
        distances = np.sqrt(((self.cohort_normalized - patient_norm)**2).sum(axis=1))
        filtered_distances = distances.loc[filtered.index]
        top_n = filtered_distances.nsmallest(n)

        return [
            {
                "patient_id": self.cohort.loc[idx, "patient_id"],
                "age": self.cohort.loc[idx, "age"],
                "surgery_type": self.cohort.loc[idx, "surgery_type"],
                "efs_score": self.cohort.loc[idx, "EFS_score"],
                "6mwt_distance": self.cohort.loc[idx, "6MWT_distance_m"],
                "similarity_distance": float(dist),
                "outcome_summary": self._summarize_outcome(idx)
            }
            for idx, dist in top_n.items()
        ]

    def _summarize_outcome(self, idx: int) -> str:
        row = self.cohort.loc[idx]
        return (f"Post-{row['surgery_type']}, age {row['age']}, "
                f"EFS {row['EFS_score']}, 6MWT {row['6MWT_distance_m']}m, "
                f"{'frail' if row['EFS_score'] >= 5 else 'non-frail'}, "
                f"{'on HR meds' if row['hr_altering_meds'] else 'no HR meds'}")
```

### Google Fit Longitudinal Baseline Integration

The patient intake state JSON combines structured clinical intake with a 7-day Google Fit historical baseline, providing between-session physiologic context even when the chest belt is not worn. This is critical because cardiac rehab sessions occur 2–3 times per week — the system needs to understand what happened during the other 4–5 days.

```python
import json
from datetime import datetime

class IntakeStateProcessor:
    """Processes the combined intake + Google Fit longitudinal state JSON.

    Intake fields:
      Clinical: subject_id, age, sex, height_cm, weight_kg, event (Post-MI/CABG/etc),
        event_date, lvef, comorbidities (dia, copd, hyp, pad, ren), beta_blocker,
        tobacco, activity_level, chest_pain, dyspnea, phq2
      Prescribed: hr_target_low, hr_target_high

    Google Fit historical_baseline (7 days, 15-min bucketed):
      Per day: steps, calories, heart_points, avg_bpm, hr_array[{ts, val}],
        body_temp, temp_array[{ts, val}], sleep_hours, sleep_stages{light, deep, rem, awake}
    """

    def __init__(self, intake_json_path: str):
        with open(intake_json_path) as f:
            self.state = json.load(f)

    @property
    def clinical_profile(self) -> dict:
        s = self.state
        return {
            "subject_id": s["subject_id"], "age": s["age"], "sex": s["sex"],
            "height_cm": s["height_cm"], "weight_kg": s["weight_kg"],
            "bmi": round(s["weight_kg"] / (s["height_cm"]/100)**2, 1),
            "event": s["event"], "event_date": s["event_date"],
            "lvef": s["lvef"],
            "comorbidities": [c for c in ["dia","copd","hyp","pad","ren"] if s.get(f"comorb_{c}")],
            "beta_blocker": s["beta_blocker"],
            "tobacco": s["tobacco"],
            "activity_level": s["activity_level"],
            "chest_pain": s["chest_pain"],
            "dyspnea": s["dyspnea"],
            "phq2": s["phq2"],  # depression screening score
            "hr_target_low": s["hr_target_low"],
            "hr_target_high": s["hr_target_high"],
            "prescribed_intensity_range": [
                s["hr_target_low"] / (220 - s["age"]),
                s["hr_target_high"] / (220 - s["age"])]
        }

    def compute_longitudinal_baselines(self) -> dict:
        """Extract between-session trends from Google Fit 7-day history."""
        days = self.state.get("historical_baseline", {}).get("days", [])
        if not days:
            return {"available": False}
        # Aggregate across days with data
        daily_avg_hr = [d["avg_bpm"] for d in days if d.get("avg_bpm")]
        daily_steps = [d["steps"] for d in days if d.get("steps", 0) > 0]
        daily_calories = [d["calories"] for d in days if d.get("calories", 0) > 0]
        daily_heart_pts = [d["heart_points"] for d in days if d.get("heart_points")]
        daily_sleep = [d["sleep_hours"] for d in days if d.get("sleep_hours", 0) > 0]
        # Resting HR estimate: minimum of 15-min bucketed HR values
        all_hr_vals = []
        for d in days:
            for sample in d.get("hr_array", []):
                if sample.get("val") and sample["val"] > 30:
                    all_hr_vals.append(sample["val"])
        resting_hr = float(np.percentile(all_hr_vals, 5)) if all_hr_vals else None
        peak_hr = float(np.max(all_hr_vals)) if all_hr_vals else None
        return {
            "available": True,
            "timeframe_days": len(days),
            "resting_hr_est": resting_hr,
            "peak_hr_7d": peak_hr,
            "mean_daily_avg_hr": float(np.mean(daily_avg_hr)) if daily_avg_hr else None,
            "mean_daily_steps": float(np.mean(daily_steps)) if daily_steps else 0,
            "mean_daily_calories": float(np.mean(daily_calories)) if daily_calories else 0,
            "mean_heart_points": float(np.mean(daily_heart_pts)) if daily_heart_pts else 0,
            "mean_sleep_hours": float(np.mean(daily_sleep)) if daily_sleep else None,
            "hr_trend": "stable" if daily_avg_hr and np.std(daily_avg_hr) < 5
                        else "variable" if daily_avg_hr else "unknown",
            "activity_trend": "active" if daily_steps and np.mean(daily_steps) > 5000
                              else "sedentary" if daily_steps else "unknown",
            "days_since_event": (datetime.now() -
                datetime.strptime(self.state["event_date"], "%Y-%m-%d")).days
        }

    def get_full_context(self) -> dict:
        """Combined clinical + longitudinal context for agent dispatch."""
        return {
            "clinical": self.clinical_profile,
            "longitudinal": self.compute_longitudinal_baselines(),
            "risk_flags": self._compute_risk_flags()
        }

    def _compute_risk_flags(self) -> list[str]:
        """Flag clinically relevant risk factors from intake + baseline."""
        flags = []
        s = self.state
        if s.get("lvef") and s["lvef"] < 40:
            flags.append("reduced_ejection_fraction")
        if s.get("phq2") and s["phq2"] >= 3:
            flags.append("positive_depression_screen")
        if s.get("comorb_dia"):
            flags.append("diabetes")
        if s.get("comorb_hyp"):
            flags.append("hypertension")
        if s.get("beta_blocker") == "Yes":
            flags.append("hr_altering_medication_beta_blocker")
        if s.get("tobacco") in ("Current", "Former"):
            flags.append(f"tobacco_{s['tobacco'].lower()}")
        if s.get("activity_level", 0) <= 1:
            flags.append("very_low_baseline_activity")
        baseline = self.compute_longitudinal_baselines()
        if baseline.get("mean_sleep_hours") and baseline["mean_sleep_hours"] < 5:
            flags.append("poor_sleep_baseline")
        if baseline.get("resting_hr_est") and baseline["resting_hr_est"] > 90:
            flags.append("elevated_resting_hr_baseline")
        return flags
```

### ChromaDB Collections + RAG Ingestion

```python
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

chroma = chromadb.PersistentClient(path="/data/chromadb")

# 5 collections (simplified from 6 — no embedding tables needed)
patient_vitals = chroma.get_or_create_collection("patient_vitals_db")
ref_cohort_features = chroma.get_or_create_collection("reference_cohort_features")
ref_cohort_metadata = chroma.get_or_create_collection("reference_cohort_metadata")
rag_literature = chroma.get_or_create_collection("rag_medical_literature")
patient_intake = chroma.get_or_create_collection("patient_intake_db")

embed_model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

def ingest_guidelines(file_path: str, source: str):
    with open(file_path) as f:
        chunks = splitter.split_text(f.read())
    embeddings = embed_model.encode(chunks).tolist()
    rag_literature.add(
        documents=chunks, embeddings=embeddings,
        ids=[f"{source}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": source} for _ in range(len(chunks))])
```

### Agent System Prompts

```python
NURSE_AGENT_PROMPT = """You are the Talk to Your Heart Wellness Companion — a warm,
supportive assistant helping patients during cardiac rehabilitation.

RULES:
1. You are a WELLNESS companion, NOT a medical provider.
2. NEVER use: "abnormal", "disease", "diagnosis", "arrhythmia", "fibrillation",
   "condition indicates", "you have", "prescribe", "take medication", "stop taking".
3. Frame ALL observations as trends: YES: "Your heart rate dropped 18 bpm in the
   first minute" / NO: "Your cardiac recovery is abnormal"
4. ALWAYS suggest consulting the care team for medical concerns.
5. Use simple, warm language. No jargon. No abbreviations.
6. If patient mentions chest pain, breathing difficulty, dizziness, or faintness:
   respond ONLY: "Please stop exercising immediately and alert the nearest staff
   member. Your safety is the priority."
7. Support configured language (English/Spanish) based on patient preference.
Temperature: 0.5"""

CLINICAL_ASSISTANT_PROMPT = """You are the Talk to Your Heart Clinical Assistant —
an interactive AI supporting clinicians with patient-specific questions and
automated clinical documentation during cardiac rehabilitation sessions.

CONTEXT PROVIDED PER QUERY:
- Current vitals: 5s window (HR, SQI, HAR features) + 30s window (HRV, morphology)
- Patient intake: age, sex, event type, LVEF, comorbidities, beta-blocker status,
  tobacco, activity level, chest pain/dyspnea history, PHQ-2 depression screen
- Google Fit 7-day longitudinal baseline: resting HR estimate, daily avg HR trend,
  steps, calories, heart points, sleep hours/stages, body temperature — providing
  between-session context even when chest belt is not worn
- Computed risk flags (reduced EF, positive depression screen, diabetes, beta-blocker,
  elevated resting HR, poor sleep, low baseline activity)
- Reference cohort matches (similar patients from PhysioNet Wearable Exercise Frailty
  Dataset with known clinical characteristics and outcomes)
- Prior session history from database
- RAG-retrieved AHA/AACVPR guideline excerpts

RULES:
1. Be specific and data-grounded. Cite actual values from the patient state.
2. Use longitudinal baseline to contextualize in-session observations: "Patient's
   resting HR from Google Fit 7-day baseline is 72 bpm; current resting HR of 85 bpm
   is elevated, possibly related to [risk flag: beta-blocker status change / poor sleep]."
3. Flag risk factors from the computed risk flags in your assessments.
4. Flag confidence based on SQI scores — mark any metric where SQI < 0.7 with
   "[low confidence - signal quality reduced]".
5. Reference similar patients from the reference cohort to contextualize findings.
6. For SOAP notes: Subjective (patient-reported + PHQ-2 + Google Fit activity/sleep),
   Objective (sensor-measured metrics with SQI % + longitudinal baselines),
   Assessment (exercise response + reference cohort comparison + risk flags),
   Plan (suggestions for clinician review, NOT directives).
7. You may use clinical terminology — audience is healthcare professionals.
8. Cite RAG-retrieved guidelines when relevant.
Temperature: 0.3"""
```

### Lead Agent Orchestrator

```python
import asyncio

class LeadOrchestrator:
    def __init__(self, nurse_url, assistant_url, chroma, cohort_matcher):
        self.nurse_url = nurse_url       # Qwen3 endpoint
        self.assistant_url = assistant_url  # MedGemma-27B endpoint
        self.cohort_matcher = cohort_matcher
        self.patient_states = {}  # latest 5s + 30s windows per patient

    async def assemble_context(self, patient_id: str) -> dict:
        """8-source context assembly for agent dispatch."""
        v = self.patient_states.get(patient_id, {})
        intake_data = patient_intake.get(where={"patient_id": patient_id})
        intake_processor = IntakeStateProcessor(f"/data/patients/{patient_id}/intake_state.json")
        full_context = intake_processor.get_full_context()
        return {
            "current_vitals_5s": v.get("window_5s", {}),
            "current_vitals_30s": v.get("hrv_30s", {}),
            "patient_intake": full_context["clinical"],
            "longitudinal_baseline": full_context["longitudinal"],  # Google Fit 7-day
            "risk_flags": full_context["risk_flags"],
            "reference_cohort_matches": self.cohort_matcher.find_similar_patients(
                full_context["clinical"], v.get("window_5s", {})),
            "session_history": patient_vitals.query(
                query_texts=[patient_id], where={"patient_id": patient_id},
                n_results=5),
            "rag_guidelines": rag_literature.query(
                query_texts=[f"cardiac rehab HR {v.get('window_5s', {}).get('hr_bpm', 0)} "
                             f"recovery HRV exercise response"], n_results=5),
        }

    async def route(self, event_type: str, patient_id: str, data: dict = {}):
        context = await self.assemble_context(patient_id)
        if event_type == "patient_chat":
            if self._emergency_keywords(data.get("message", "")):
                return {"response": "Please stop exercising immediately and "
                        "alert the nearest staff member. Your safety is the priority.",
                        "alert": "critical"}
            response = await self._call_agent(self.nurse_url,
                NURSE_AGENT_PROMPT, context, data["message"])
            if self._diagnostic_language(response):
                return {"response": "I'm here to support your rehab session. "
                        "For medical questions, please speak with your care team."}
            return {"response": response}
        elif event_type == "clinician_chat":
            return await self._call_agent(self.assistant_url,
                CLINICAL_ASSISTANT_PROMPT, context, data["message"])
        elif event_type == "generate_soap":
            return await self._call_agent(self.assistant_url,
                CLINICAL_ASSISTANT_PROMPT, context,
                "Generate a complete SOAP note for this session including "
                "reference cohort comparison and SQI confidence annotations.")

    @staticmethod
    def _emergency_keywords(text: str) -> bool:
        return any(kw in text.lower() for kw in [
            "chest pain", "can't breathe", "cannot breathe", "dizzy",
            "passing out", "faint", "heart racing", "nauseous", "blacking out"])

    @staticmethod
    def _diagnostic_language(text: str) -> bool:
        return any(t in text.lower() for t in [
            "diagnose", "diagnosis", "abnormal", "disease", "you have",
            "arrhythmia", "fibrillation", "prescribe", "take medication"])
```

### FastAPI Backend

```python
from fastapi import FastAPI, WebSocket
app = FastAPI(title="Talk to Your Heart — DGX Spark Backend")

@app.websocket("/ws/ecg/{patient_id}")
async def ecg_stream(websocket: WebSocket, patient_id: str):
    await websocket.accept()
    ecg_proc = ECGProcessor(fs=130)
    har = AccelerometerHAR(fs=100)
    publisher = VitalsPublisher()
    intake = get_patient_intake(patient_id)
    safety = EnergySafeWindow(intake)
    buf_5s = {"ecg": [], "acc": []}
    buf_30s = {"ecg": [], "acc": []}
    while True:
        data = await websocket.receive_json()
        ecg_chunk = np.array(data["ecg"])
        acc_chunk = np.array(data["acc"]).reshape(-1, 3)
        buf_5s["ecg"].extend(ecg_chunk); buf_5s["acc"].extend(acc_chunk)
        buf_30s["ecg"].extend(ecg_chunk); buf_30s["acc"].extend(acc_chunk)
        # 5-second window processing
        if len(buf_5s["ecg"]) >= 130 * 5:
            ecg_5 = np.array(buf_5s["ecg"][:130*5])
            acc_5 = np.array(buf_5s["acc"][:100*5]).reshape(-1, 3)
            clean = ecg_proc.preprocess(ecg_5)
            r_peaks = ecg_proc.detect_r_peaks(clean)
            hr = 60000 / np.mean(np.diff(r_peaks)/130*1000) if len(r_peaks) > 1 else 0
            har_feat = har.extract_features(acc_5)
            sqi = ecg_proc.compute_sqi(clean, r_peaks, har_feat["magnitude"])
            activity = har.classify_activity(har_feat, hr)
            safe, alert = safety.check(hr, activity, sqi)
            result_5s = WindowResult5s(
                patient_id=patient_id, timestamp=data["timestamp"],
                hr_bpm=hr, sqi=sqi,
                mean_mag_mg=har_feat["mean_mag_mg"],
                var_mag_mg2=har_feat["var_mag_mg2"],
                spectral_entropy=har_feat["spectral_entropy"],
                median_freq_hz=har_feat["median_freq_hz"],
                activity_class=activity, energy_safe=safe, alert_level=alert)
            publisher.publish_window_5s(result_5s)
            await websocket.send_json(asdict(result_5s))
            buf_5s = {"ecg": buf_5s["ecg"][130*5:], "acc": buf_5s["acc"][100*5:]}
        # 30-second window processing
        if len(buf_30s["ecg"]) >= 130 * 30:
            ecg_30 = np.array(buf_30s["ecg"][:130*30])
            acc_30 = np.array(buf_30s["acc"][:100*30]).reshape(-1, 3)
            clean_30 = ecg_proc.preprocess(ecg_30)
            r_peaks_30 = ecg_proc.detect_r_peaks(clean_30)
            hrv = ecg_proc.compute_hrv_30s(r_peaks_30)
            morph = ecg_proc.delineate_morphology(clean_30, r_peaks_30)
            hr_30 = 60000/np.mean(np.diff(r_peaks_30)/130*1000) if len(r_peaks_30)>1 else 0
            har_30 = har.extract_features(acc_30)
            mets = har.estimate_mets(hr_30, har_30["var_mag_mg2"],
                                     intake.get("resting_hr", 70), intake["age"])
            result_30s = HRVResult30s(
                patient_id=patient_id, timestamp=data["timestamp"],
                hrv_rmssd_ms=hrv["rmssd"], hrv_sdnn_ms=hrv["sdnn"],
                hrv_pnn50=hrv["pnn50"], hrv_lf_hf_ratio=hrv["lf_hf"],
                qrs_width_ms=morph["qrs_width_ms"],
                qt_interval_ms=morph["qt_interval_ms"],
                st_deviation_mv=morph["st_deviation_mv"], met_estimate=mets)
            publisher.publish_hrv_30s(result_30s)
            buf_30s = {"ecg": buf_30s["ecg"][130*30:], "acc": buf_30s["acc"][100*30:]}

@app.post("/api/chat/patient/{patient_id}")
async def patient_chat(patient_id: str, message: dict):
    return await orchestrator.route("patient_chat", patient_id, message)

@app.post("/api/chat/clinician/{patient_id}")
async def clinician_chat(patient_id: str, message: dict):
    return await orchestrator.route("clinician_chat", patient_id, message)

@app.get("/api/session/{patient_id}/soap")
async def generate_soap(patient_id: str):
    return await orchestrator.route("generate_soap", patient_id)

@app.get("/api/dashboard/active")
async def active_patients():
    return orchestrator.get_all_patient_states()

@app.get("/api/export/fhir/{patient_id}/{session_id}")
async def export_fhir(patient_id: str, session_id: str):
    """FHIR R4 Observation for Epic/Cerner/Meditech integration."""
    session = patient_vitals.get(where={"patient_id": patient_id, "session_id": session_id})
    return {"resourceType": "Observation", "status": "final",
            "subject": {"reference": f"Patient/{patient_id}"},
            "component": [
                {"code": {"text": "Peak HR"}, "valueQuantity": {"value": session.get("peak_hr"), "unit": "bpm"}},
                {"code": {"text": "RMSSD"}, "valueQuantity": {"value": session.get("rmssd"), "unit": "ms"}},
                {"code": {"text": "METs"}, "valueQuantity": {"value": session.get("peak_mets"), "unit": "MET"}},
                {"code": {"text": "SQI"}, "valueQuantity": {"value": session.get("mean_sqi"), "unit": "%"}}]}
```

### Flutter BLE Integration

```dart
import 'package:polar/polar.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';

class PolarH10Service {
  final polar = Polar();
  late WebSocketChannel _wsChannel;

  PolarH10Service({required String patientId}) {
    _wsChannel = WebSocketChannel.connect(
      Uri.parse('ws://dgx-spark-local:8000/ws/ecg/$patientId'));
  }

  Future<void> startStreaming(String deviceId) async {
    await polar.connectToDevice(deviceId);
    polar.startEcgStreaming(deviceId).listen((ecgData) {
      _wsChannel.sink.add(jsonEncode({
        'ecg': ecgData.samples.map((s) => s.voltage).toList(),
        'timestamp': DateTime.now().millisecondsSinceEpoch / 1000,
      }));
    });
    polar.startAccStreaming(deviceId).listen((accData) {
      _wsChannel.sink.add(jsonEncode({
        'acc': accData.samples.map((s) => [s.x, s.y, s.z]).toList(),
        'timestamp': DateTime.now().millisecondsSinceEpoch / 1000,
      }));
    });
    _wsChannel.stream.listen((msg) => updateDashboard(jsonDecode(msg)));
  }
}
```

## Features

- Real-time multi-patient cardiac rehab monitoring on DGX Spark — zero patient data egress
- HIPAA compliance through hardware-enforced data locality, not cloud BAAs
- Dual-window signal processing: 5-second (SQI, HR, HAR) + 30-second (HRV, morphology) concurrent analysis
- Pan-Tompkins + Hamilton consensus QRS detection (50ms tolerance) with DWT morphology delineation
- Four most discriminative chest-belt HAR features: mean magnitude, variance/energy, spectral entropy, median frequency
- Three-metric SQI (template matching 0.4 + SNR 0.3 + motion correlation 0.3) propagated as confidence through all agent context
- HRV analysis: RMSSD, SDNN, pNN50, LF/HF via Welch periodogram over 30-second windows
- MET estimation via Swain 2000 %VO2R ≈ %HRR method weighted with accelerometer energy
- Energy Safe Window: deterministic safety check before every MQTT publish
- PhysioNet Wearable Exercise Frailty Dataset reference cohort matching by clinical metadata (surgery type, age, EFS score, comorbidities, 6MWT, TUG) — interpretable similarity, not opaque embeddings
- ChromaDB: 5 collections (vitals, reference cohort features, reference cohort metadata, RAG literature, patient intake)
- RAG grounded in AHA/AACVPR 2024 guidelines — 512-token chunks, PubMedBERT embeddings, top-5 cosine similarity
- MQTT (local Mosquitto): per-patient topic isolation, QoS 2 for alerts, dual-window topic structure
- Qwen3 Nurse Agent: compassionate patient education with wellness guardrails, configurable Spanish-language mode
- MedGemma-27B Clinical Assistant: Doctor Chat Interface + SOAP note generation with reference cohort context
- Lead Orchestrator: deterministic routing, 8-source context assembly, emergency keyword bypass
- Google Fit 7-day longitudinal baseline integration: resting HR estimate, daily avg HR trend, steps, calories, heart points, sleep hours/stages, body temperature — between-session context without chest belt
- Structured patient intake with clinical risk flag computation: LVEF, comorbidities, beta-blocker status, PHQ-2 depression screening, tobacco, activity level
- 4-layer safety: deterministic alerts → emergency keyword classifier → output validator → wellness system prompt
- SOAP notes as admin documentation with SQI % confidence annotations and reference cohort comparisons
- All metrics directly mapped to published clinical literature — fully interpretable, auditable, overridable
- Concurrent dual-model serving in 128 GB unified memory with 29 GB headroom
- Flutter dashboards: patient chat + clinician multi-patient view + Doctor Chat + PDF reports
- WCAG 2.1 AA compliant UI: 4.5:1 contrast, ≥16px fonts, ≥44px touch targets for mean patient age 63
- Three-tier API: WebSocket+MQTT (real-time), REST FastAPI (queries), FHIR R4 export
- Immutable clinical audit trail for HIPAA 45 CFR § 164.312(b) compliance
- Prometheus + Grafana observability: agent latency, GPU memory, MQTT rates
- Two-phase regulatory: FDA 2026 general wellness now, 510(k) later
- Reimbursement: CPT 93798 + RPM 99454/99457/99458 = $99+/month/patient

## User Experience

### Accessibility Design

Designed for cardiac rehab's actual patient population: mean age 63, frequently with comorbid vision impairment and limited health literacy. WCAG 2.1 AA: 4.5:1 contrast, ≥16px body font (24px vitals), ≥44px touch targets, screen reader compatible. English and Spanish at launch — directly addressing the 50% Hispanic participation gap.

### Patient-Facing

The patient wears a Polar H10 and interacts with a tablet. **Warm-up:** "Good morning, Maria. Your session is underway and everything looks steady." **Exercise:** "You've been at it for 12 minutes and you're right in your target zone." **Approaching limit:** "Your heart rate has climbed a bit higher than usual. You might ease back slightly." **Recovery:** "Nice cooldown — your heart rate is settling toward your resting level." The patient never sees HRV numbers, morphology metrics, or clinical terminology. The equity story is central: Maria Santos interacts in Spanish, the system adapts seamlessly, and a patient population historically excluded from full participation gets the same AI-supported care.

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

Doctor Chat Interface: "How does James compare to similar post-CABG patients?" → Clinical Assistant retrieves reference cohort matches, compares HR recovery and HRV profiles, responds with data-grounded context. SOAP notes auto-generated with SQI confidence annotations and reference cohort comparisons — 2-minute clinician review replaces 10–15 minutes manual documentation. Six patients saves 48–78 minutes per session.

### Targeted Patient Outcomes

CR completion rate increase from 27% baseline toward 50%+. Documentation time reduction of 80–87%. Monitoring capacity increase of 33–100%. Alert false positive rate below 5%. Clinician satisfaction SUS score >80. Health equity impact tracked by Hispanic and non-English enrollment rates.

## Scalability Design

### Single-Device Capacity

Each DGX Spark runs a fully independent pipeline for 8 concurrent patients with 2 concurrent LLM agents, on-demand SOAP generation, and <5s Nurse response latency. Without foundation model inference on the GPU critical path, the full Blackwell GPU is available for LLM serving — improving response latency and concurrent request handling compared to architectures that share GPU between signal models and language models.

### Horizontal Scaling

Multiple DGX Sparks operate independently — one per clinic. Central MQTT broker federation aggregates anonymized metrics without PHI sharing. Plugin architecture: new agents subscribe via documented interface with semantic versioning. Billing Code Agent, Research Data Extractor, or site-specific agents add without core pipeline changes.

### Workflow Impact

```
+============================================================================+
|   METRIC                        BASELINE (MANUAL)    WITH SYSTEM           |
|   ---------------------------------------------------------------------------
|   SOAP note generation time     10-15 min/patient    1.5-2 min (review)   |
|   SOAP reduction                 —                    80-87%               |
|   Concurrent patients monitored  3-4 effectively      8+ with AI assist   |
|   Monitoring capacity increase   —                    33-100%+             |
|   Alert response latency         Minutes (scanning)   <1 second            |
|   Documentation completeness     Variable (recall)    Structured (sensor)  |
+============================================================================+
```

## Ecosystem Thinking

### Three-Tier API

**Tier 1: Real-Time (WebSocket + MQTT).** Dual-window vitals via WebSocket. MQTT pub/sub with QoS 0 vitals, QoS 2 alerts. **Tier 2: REST FastAPI.** Patient chat, clinician queries, dashboard state, SOAP generation. Schema-versioned JSON. **Tier 3: FHIR R4 Export.** Observation + DiagnosticReport for Epic/Cerner/Meditech.

### Monitoring, Audit, Governance

Prometheus metrics (agent latency p50/p95/p99, GPU memory, MQTT rates). Grafana dashboards. Immutable audit trail per HIPAA 45 CFR § 164.312(b). Configurable data retention policies. Patient_id-scoped ChromaDB access. Cascading deletion.

## Market Opportunity

**TAM:** $3.66B AI-driven cardiac platforms by 2030. **SAM:** $1.39B U.S. cardiac rehab programs. **SOM (Year 1–2):** $4.2M — 35 clinics at $120K ACV. Revenue: hardware deployment ($40–60K) + annual license ($60–80K) + RPM billing ($99+/month/patient, ~$118,800/year per 100). Go-to-market: direct sales to top 200 hospitals by CR volume. CMS permanently enabled virtual CR supervision in CY 2026.

## Competitive Landscape

```
+============================================================================+
|                    Continuous  Interpretable  Multi-Agent  Auto    On-Prem |
|   Competitor        ECG        Clinical AI    Role-Sep     SOAP    Edge    |
|   --------------------------------------------------------------------------
|   Fourth Frontier    YES        no             no          no      no      |
|   Recora             no         no             no          no      no      |
|   Biofourmis*        YES        proprietary    no          no      no      |
|   Movn Health        no         no             no          no      no      |
|   --------------------------------------------------------------------------
|   Talk to Your Heart YES        Interpretable  2 agents    YES     DGX     |
|                                 + Ref Cohort                       Spark   |
+============================================================================+
* Biofourmis (now General Informatics): cloud-dependent, no role separation,
  no SOAP, no edge. HIPAA-as-architecture cannot be retrofitted onto cloud-native.
```

Defensibility: integration complexity across interpretable signal pipeline + reference cohort matching + role-separated agents + edge deployment. 12–18 months to replicate. Privacy-preserving architecture is an offensive advantage, not just compliance.

## Regulatory and Reimbursement Strategy

**Phase 1 (Now): General Wellness.** FDA January 2026 guidance covers non-invasive physiologic sensors estimating HRV and recovery. Allowed: "Your HR dropped 18 bpm." Prohibited: "Your cardiac recovery is abnormal." SOAP notes as administrative documentation outside SaMD. Interpretable metrics strengthen regulatory position — every output maps to a published clinical measurement with known validity.

**Phase 2 (12–18 months): FDA 510(k).** Predicates: Hexoskin (Nov 2025), CardioTag (2025), Apple Watch ECG, AliveCor KardiaMobile.

```
+============================================================================+
|   CODE    DESCRIPTION                                      REIMBURSEMENT   |
|   93798   Outpatient CR with continuous ECG                 Per session     |
|   99453   Initial RPM setup                                 $22 one-time   |
|   99454   Device supply + data (16+ days)                   $47/month      |
|   99457   RPM management, first 20 min                      $52/month      |
|   99458   Each additional 20 min                            $41/month      |
|   99445   Device supply, 2-15 days (NEW 2026)               $52/month      |
|   99470   First 10 min management (NEW 2026)                $26/month      |
+============================================================================+
```

## Execution Plan

### Core vs. Stretch

```
+============================================================================+
|   CORE (Must ship):                                                        |
|   - BLE → dual-window signal processing → MQTT → single agent → dashboard |
|   - Deterministic alert engine + Energy Safe Window                        |
|   - Clinical Assistant Agent with RAG + reference cohort matching          |
|   - Clinician dashboard + Doctor Chat + SOAP generation                    |
|                                                                            |
|   STRETCH:                                                                 |
|   - Nurse Agent (patient-facing chat)                                      |
|   - ECG waveform visualizer in Flutter                                     |
|   - FHIR R4 export endpoint                                               |
|   - Cross-patient trend comparison (toy federated analytics)               |
+============================================================================+
```

### Phase 1: Foundation (Hours 0–6) — Viggi + Sansrit
DGX Spark: Ubuntu, Mosquitto MQTT, ChromaDB, vLLM with Qwen2.5-72B-AWQ. Flutter: polar Dart BLE → WebSocket → FastAPI → MQTT.
**Hour 6 Go/No-Go:** Live data streaming through DGX Spark. If BLE unstable → mock sensor (ecg_simulate at 130 Hz).

### Phase 2: Signal Intelligence (Hours 6–12) — Shiva + Rumon
Dual-window pipeline: 5s (SQI, HR, HAR features) + 30s (HRV, morphology). Energy Safe Window. Deterministic alerts. Reference cohort ingestion from PhysioNet dataset.
**Workload redistribution:** Rumon handles accelerometer HAR feature extraction. Shiva focuses on ECG pipeline + SQI.
**Hour 12 Go/No-Go:** Structured JSON pipeline operational. If morphology delineation unreliable → drop QT/ST, keep HR/HRV/SQI/HAR.

### Phase 3: RAG + Agent (Hours 12–18) — Shiva + Viggi
PubMedBERT RAG into ChromaDB. Clinical Assistant Agent deployment with reference cohort matching + 8-source context assembly + SOAP template. 4-layer safety guardrails.
**Viggi absorbs RAG ingestion so Shiva focuses on agent engineering.**
**Hour 18 Go/No-Go:** Clinical Assistant operational with RAG + cohort matching + SOAP. If stable → add Nurse Agent as stretch.

### Phase 4: Integration + Demo (Hours 18–24) — Sansrit + Rumon + All
Clinician dashboard, Doctor Chat, SOAP display. Multi-patient testing (ecg_simulate). Safety testing. Demo build.
**Rumon supports Sansrit on UI. ECG waveform visualizer = stretch.**

### Fallback Cascade

```
+============================================================================+
|   FAILURE MODE                FALLBACK                         CUT POINT   |
|   ---------------------------------------------------------------------------
|   BLE unstable                Mock sensor (ecg_simulate)       Hour 6      |
|   Memory pressure             Reduce KV cache → swap to 32B   Hour 12     |
|   Morphology delineation      Drop QT/ST, keep HR/HRV/SQI     Hour 12     |
|   Reference cohort matching   Static top-5 matches per profile Hour 18     |
|   Multi-agent (Nurse) unstable Single agent (Clinical Asst)    Hour 18     |
|   Flutter UI overscoped       Drop ECG visualizer, keep cards  Hour 18     |
|   SOAP generation slow        Pre-compute, show cached example Hour 22     |
+============================================================================+
```

## Validation and Demo

**Signal:** ecg_simulate at 60–180 bpm. Verify HR/HRV vs reference. SQI: clean >0.9, noised <0.5.

**Alerts:** Synthetic tachycardia (CRITICAL), delayed recovery (WARNING), SQI degradation (ADVISORY). Fire without LLM.

**Agent Safety:** Emergency keywords → hardcoded response <100ms. Output validator blocks diagnostic language.

**Reference Cohort:** Query with test patient profiles → verify clinically sensible matches (post-CABG returns post-CABG, frail returns frail).

**Concurrency:** 4–6 simulated patients. Targets: >130 samples/sec/stream, Clinical Assistant <5s, SOAP <30s.

### Demo Scenario

**Patient A (Maria, 62, post-MI, Spanish-speaking):** Normal session. Nurse Agent in Spanish provides encouragement. Clean SOAP with reference cohort context. *The equity story is the emotional center: a Spanish-speaking patient gets the same AI-supported care.*

**Patient B (James, 55, post-CABG, diabetic):** HR approaches max → WARNING → dashboard highlights → Clinical Assistant flags with intake context + reference cohort comparison showing similar patients' expected recovery.

**Patient C (Ruth, 70, HF):** Motion artifact → SQI drops → ADVISORY → agent notes reduced confidence → staff checks sensor → recovers. Signal quality awareness.

**Clinician asks:** "How does James compare to similar post-CABG patients?" → Clinical Assistant retrieves reference cohort, compares, answers with data.

**Stretch demo:** Show anonymized cross-patient trend comparison — toy federated analytics across the three demo patients.

## Risks and Mitigations

**Memory pressure (99 GB / 128 GB):** Reduce KV cache → swap to 32B model → single-agent mode.

**Single-lead limitation:** Scoped to rhythm/rate/HRV/recovery (validated). Phase 1 wellness framing avoids claims requiring 12-lead parity. Interpretable metrics mean every output is clinically verifiable.

**LLM hallucination:** 4-layer safety. Deterministic alerts independent of LLMs. Output grounded in structured data + RAG + reference cohort. Temperature 0.3.

**BLE instability:** SQI detects degradation. Mock sensor fallback at Hour 6. Graceful degradation.

**Regulatory risk:** Output validator + wellness prompt + FDA 2026 review. Interpretable metrics strengthen regulatory position — no opaque model outputs to explain.

**Staff resistance:** 80–87% SOAP reduction. 48–78 min saved/session. System subtracts work.

**Ethical — algorithmic bias:** Post-deployment monitoring tracks alert accuracy stratified by demographics. SQI-conditioned confidence ensures agents flag uncertainty.

**Ethical — automation complacency:** System is additive. SOAP notes require "Review and Approve." Raw signals preserved alongside AI summaries.

## Team Plan

**Rumon — Hardware / PMF.** Biomedical systems, cardiac monitoring hardware, BLE protocols. Polar H10 setup, intake schema, clinical workflow mapping, accelerometer HAR extraction (from Shiva Phase 2), UI support for Sansrit Phase 4. Phases 1, 2, 4.

**Viggi — DGX Spark.** NVIDIA GPU computing, container orchestration, edge deployment. vLLM (72B + 27B), Mosquitto MQTT, ChromaDB, memory optimization, RAG ingestion (from Shiva Phase 3), Prometheus/Grafana. Phases 1, 3, 4.

**Shiva — AI / ML / RAG.** ML, NLP, medical AI, LLM guardrails. ECG pipeline (processing_worker.py — already implemented: dual-window analysis, Lomb-Scargle HRV, DWT morphology, three-metric SQI), agent prompts + 4-layer guardrails, Lead Orchestrator, reference cohort matcher, SOAP template. Phases 2, 3.

**Sansrit — Flutter / Frontend.** Flutter/Dart, real-time visualization, BLE integration. polar Dart, WebSocket bridge, patient chat, clinician dashboard (WCAG AA), Doctor Chat, SOAP display, PDF reports. Phases 1, 4. *ECG visualizer = stretch. Rumon supports Hours 18–24.*

### Already Implemented

- `processing_worker.py` (+418 lines): Dual-window analysis, Lomb-Scargle HRV, DWT morphology, three-metric SQI
- `mqtt_worker.py`: QThread MQTT publisher, paho-mqtt v2, thread-safe queue
- HAR fusion model: SSL HARNet10 + ResNet1D, 1152-dim features, subject-wise splits
- `ecg_fm_inference.py`: ECG-FM inference + `Metric_lookup.py` segment-level parquet generation
- Polar H10 BLE dashboard with live streaming
- Google Fit historical baseline integration (15-min HR/temp bucketing, sleep stage segmentation)
- Role-based web UI with doctor/patient/report tabs
- Intake state JSON schema combining clinical profile + Google Fit 7-day longitudinal data

**Remaining:** Agent orchestration (LeadOrchestrator + 2 agent personas), ChromaDB RAG ingestion, vLLM deployment on DGX Spark, reference cohort integration, Flutter clinician dashboard + Doctor Chat.

## Vision

**6 months:** Pilot 2–3 rehab programs. 70% documentation time reduction. <5% alert FP rate. Movesense MD. FHIR R4 EHR integration.

**12–18 months:** FDA 510(k). Daily wellness reporting — HRR during daily activities, MET tracking, 6MWT estimation (Cole et al., NEJM 1999). Expand reference cohort with pilot site data — each new site enriches the clinical comparison database.

**2–3 years:** Pulmonary rehab, neurological rehab, post-surgical recovery. Federated multi-site analytics — anonymized population metrics (HRR percentiles by surgery type, recovery benchmarks by EFS score) across DGX Sparks without PHI sharing. Privacy-preserving learning counters cloud data network effects.

DGX Spark as an intelligent clinical operations core. If this system helps one clinic complete rehab for ten more patients per year, that is ten fewer cardiac deaths. Multiply across thousands of programs, and the numbers define whether we took this crisis seriously. Talk to Your Heart is the integration that has been missing.