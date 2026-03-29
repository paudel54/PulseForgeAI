# Talk to Your Heart — End-to-End Pipeline Blueprint

## Model Decision: The Three-Tier Strategy

Before detailing the pipeline, here's the model recommendation with a clear decision framework.

### Primary LLM: Pick ONE based on your hackathon risk tolerance

| Strategy | Model | INT4 Memory | Decode Speed | MedQA | Risk Level |
|----------|-------|-------------|-------------|-------|-----------|
| **Safe bet** | `nvidia/Qwen3-32B-FP4` | ~18GB | ~15-20 tok/s | Competitive | LOW — NVIDIA pre-validated on Spark |
| **Best medical** | MedGemma-27B-text v1.5 | ~14GB | ~18-22 tok/s | **87.7%** | MEDIUM — needs Gemma tokenizer setup |
| **Maximum reasoning** | HuatuoGPT-o1-72B (AWQ) | ~40GB | ~4-8 tok/s | Top-tier | HIGH — slow decode, large KV cache |
| **Fastest fallback** | m1-7B-23K | ~4GB | ~40-50 tok/s | 60.3% | LOWEST — snappy but less knowledgeable |

**Recommended approach**: Start with `nvidia/Qwen3-32B-FP4` (pre-quantized, validated on DGX Spark, native `<think>` reasoning blocks). If you get it running in hours 0-4, try swapping to MedGemma-27B or HuatuoGPT-o1-72B during polish time. Keep m1-7B-23K as an emergency fallback that will always be fast enough for live demos.

**Why not HuatuoGPT-o1-72B as primary?** At ~40GB INT4, it leaves only ~78GB for everything else. More critically, 4-8 tok/s decode means a 200-token narrative takes 25-50 seconds — too slow for a live demo's "talk to your heart and it talks back" moment. Use it only if you pre-generate session summaries (not real-time).

**Why Qwen3-32B over m1-32B-1K?** Qwen3's native `<think>` / `</think>` blocks give you test-time scaling without any custom inference hacking. m1-32B-1K requires manual "Wait" token injection that vLLM doesn't natively support. Qwen3-32B with a medical system prompt + RAG gets you 90% of m1's benefit with zero custom inference code.

---

## The Complete Pipeline: 7 Stages from Heartbeat to Insight

```
STAGE 1          STAGE 2           STAGE 3          STAGE 4
Polar H10  →  Ring Buffer  →  Signal Processing → CLEF Embedding
(BLE 130Hz)   (5s sliding)    (NeuroKit2+SQI)    (500Hz, 10s)
                                     |                  |
                                     v                  v
                              STAGE 5: Feature     STAGE 6: ChromaDB
                              Aggregation JSON     k-NN Retrieval
                                     |                  |
                                     +--------+---------+
                                              |
                                              v
                                    STAGE 7: LLM Narrative
                                    (Qwen3-32B + RAG)
                                              |
                                    +---------+---------+
                                    |                   |
                              Patient Coach       Clinician Scribe
                              (WebSocket→Flutter)  (SOAP + PDF)
```

---

### STAGE 1: Polar H10 BLE Data Acquisition

**What happens**: The Polar H10 chest strap streams two independent data channels over Bluetooth Low Energy to the DGX Spark backend.

**Data streams**:
- **Raw ECG**: 130 Hz, 14-bit resolution, ~73 samples per BLE packet (~560ms intervals). Single-lead approximating V4-V5 chest position.
- **RR Intervals**: 1ms resolution timestamps between consecutive R-peaks. Computed on-device by Polar's firmware. This is your primary HRV source (bypasses the 130Hz jitter problem).
- **Accelerometer** (optional): 25-200 Hz configurable, 3-axis, 16-bit. For activity phase detection.

**Implementation**:

```python
# polar_client.py
from polar_python import PolarDevice, MeasurementType
import asyncio

class PolarH10Client:
    def __init__(self, ecg_queue, rr_queue, acc_queue=None):
        self.ecg_queue = ecg_queue
        self.rr_queue = rr_queue
        self.acc_queue = acc_queue

    async def connect_and_stream(self, device_id: str):
        device = PolarDevice(device_id)
        await device.connect()

        # Start ECG stream (130Hz)
        await device.start_measurement(
            MeasurementType.ECG,
            callback=self._on_ecg_data
        )
        # Start RR interval stream
        await device.start_measurement(
            MeasurementType.RR_INTERVAL,
            callback=self._on_rr_data
        )
        # Optional: accelerometer for activity detection
        if self.acc_queue:
            await device.start_measurement(
                MeasurementType.ACC,
                settings={"sample_rate": 50, "range": 8},
                callback=self._on_acc_data
            )

    async def _on_ecg_data(self, data):
        # data.samples = list of int (µV values)
        # data.timestamp = nanosecond timestamp
        await self.ecg_queue.put({
            "samples": data.samples,
            "timestamp": data.timestamp,
            "type": "ecg"
        })

    async def _on_rr_data(self, data):
        await self.rr_queue.put({
            "rr_intervals_ms": data.rr_intervals,
            "timestamp": data.timestamp,
            "type": "rr"
        })
```

**Output**: Raw ECG samples and RR intervals flowing into bounded asyncio queues.

**Failure modes to handle**: BLE disconnection (auto-reconnect with exponential backoff), electrode contact loss (HR drops to 0 — detect and flag), packet loss (timestamp gaps > 600ms).

---

### STAGE 2: Ring Buffer with Sliding Windows

**What happens**: Raw ECG samples accumulate in a circular buffer. Every 1 second, a 5-second window (650 samples at 130Hz) is extracted for signal processing. Every 10 seconds, a 10-second window (1,300 samples) is extracted and resampled for CLEF embedding.

**Two buffer outputs**:
- **5-second windows** (1s stride) → Stage 3 (signal processing, SQI, R-peaks)
- **10-second windows** (5s stride) → Stage 4 (CLEF embedding inference)

```python
# ring_buffer.py
import numpy as np
from collections import deque

class ECGRingBuffer:
    def __init__(self, max_seconds=30, sample_rate=130):
        self.sr = sample_rate
        self.buffer = deque(maxlen=max_seconds * sample_rate)
        self.timestamps = deque(maxlen=max_seconds * sample_rate)

    def append(self, samples, timestamp):
        for i, s in enumerate(samples):
            self.buffer.append(s)
            self.timestamps.append(timestamp + i * (1e9 / self.sr))

    def get_window(self, seconds=5):
        n = seconds * self.sr
        if len(self.buffer) < n:
            return None
        return np.array(list(self.buffer)[-n:], dtype=np.float64)

    def get_window_for_clef(self, seconds=10):
        """Returns 10s window resampled to 500Hz (5000 samples)"""
        raw = self.get_window(seconds)
        if raw is None:
            return None
        from scipy.signal import resample
        return resample(raw, 5000)  # 130Hz→500Hz
```

**Output**: Numpy arrays of ECG samples, ready for processing.

---

### STAGE 3: Signal Processing & Quality Gating

**What happens**: Each 5-second window goes through a deterministic signal processing pipeline. This stage produces ALL the hard numbers — HR, HRV, SQI, R-peaks, QRS morphology. No AI involved. These are the safety-critical metrics.

**Sub-steps**:

**3a. Signal Quality Index (SQI)** — computed FIRST, gates everything else:

```python
# sqi.py
import neurokit2 as nk
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import welch

def compute_sqi(ecg_window, sr=130):
    """4-component SQI → float 0.0-1.0"""

    # pSQI: QRS power ratio (5-15Hz / 5-40Hz)
    f, psd = welch(ecg_window, fs=sr, nperseg=min(256, len(ecg_window)))
    qrs_power = np.trapz(psd[(f >= 5) & (f <= 15)], f[(f >= 5) & (f <= 15)])
    total_power = np.trapz(psd[(f >= 5) & (f <= 40)], f[(f >= 5) & (f <= 40)])
    p_sqi = qrs_power / (total_power + 1e-10)
    p_sqi_score = 1.0 if 0.4 <= p_sqi <= 0.8 else max(0, 1 - abs(p_sqi - 0.6) * 2)

    # kSQI: kurtosis (sharp R-peaks → high kurtosis)
    k = kurtosis(ecg_window, fisher=True)
    k_sqi = min(1.0, max(0, (k - 2) / 8))  # normalize: 2→0, 10→1

    # basSQI: baseline wander (power 0-1Hz / 0-40Hz)
    baseline_power = np.trapz(psd[(f >= 0) & (f <= 1)], f[(f >= 0) & (f <= 1)])
    total_40 = np.trapz(psd[(f >= 0) & (f <= 40)], f[(f >= 0) & (f <= 40)])
    bas_sqi = 1 - (baseline_power / (total_40 + 1e-10))

    # qSQI: dual-detector agreement
    try:
        peaks_nk = nk.ecg_findpeaks(ecg_window, sampling_rate=sr, method="neurokit")["ECG_R_Peaks"]
        peaks_ham = nk.ecg_findpeaks(ecg_window, sampling_rate=sr, method="hamilton2002")["ECG_R_Peaks"]
        if len(peaks_nk) > 0 and len(peaks_ham) > 0:
            matches = sum(1 for p in peaks_nk if any(abs(p - h) < sr * 0.15 for h in peaks_ham))
            q_sqi = matches / max(len(peaks_nk), len(peaks_ham))
        else:
            q_sqi = 0.0
    except:
        q_sqi = 0.5

    # Weighted combination
    combined = 0.3 * q_sqi + 0.3 * p_sqi_score + 0.2 * k_sqi + 0.2 * bas_sqi
    return round(combined, 3), {"q": q_sqi, "p": p_sqi_score, "k": k_sqi, "bas": bas_sqi}
```

**3b. R-peak detection & HR**:

```python
# ecg_processing.py
import neurokit2 as nk
import heartpy as hp

def process_ecg_window(ecg_window, sr=130, sqi_score=1.0):
    """Full signal processing on a 5-second window"""
    result = {}

    # R-peak detection (NeuroKit2 default method)
    try:
        signals, info = nk.ecg_process(ecg_window, sampling_rate=sr)
        r_peaks = info["ECG_R_Peaks"]
        result["r_peaks"] = r_peaks.tolist()
        result["hr_bpm"] = round(signals["ECG_Rate"].mean(), 1)
    except:
        result["r_peaks"] = []
        result["hr_bpm"] = None

    # High-precision R-peak timing via HeartPy (for HRV)
    if sqi_score > 0.5:
        try:
            wd, m = hp.process(ecg_window, sample_rate=sr,
                              high_precision=True, high_precision_fs=1000)
            result["rr_intervals_hp"] = wd["RR_list_cor"]  # artifact-corrected
        except:
            pass

    # QRS morphology (only when SQI > 0.6)
    if sqi_score > 0.6:
        try:
            _, waves = nk.ecg_delineate(ecg_window, r_peaks,
                                         sampling_rate=sr, method="dwt")
            # Approximate QRS duration
            qrs_onsets = waves.get("ECG_Q_Peaks", [])
            qrs_offsets = waves.get("ECG_S_Peaks", [])
            if qrs_onsets and qrs_offsets:
                durations = [(s - q) / sr * 1000
                            for q, s in zip(qrs_onsets, qrs_offsets)
                            if q is not None and s is not None]
                result["qrs_ms"] = round(np.median(durations), 1) if durations else None
        except:
            pass

    return result
```

**3c. HRV from RR intervals** (NOT from raw ECG — uses the dedicated RR stream):

```python
# hrv.py
import neurokit2 as nk
import numpy as np

class HRVComputer:
    def __init__(self, window_minutes=5):
        self.rr_buffer = []  # milliseconds
        self.window_ms = window_minutes * 60 * 1000

    def add_rr(self, rr_intervals_ms):
        self.rr_buffer.extend(rr_intervals_ms)
        # Trim to window
        total = sum(self.rr_buffer)
        while total > self.window_ms and len(self.rr_buffer) > 10:
            total -= self.rr_buffer.pop(0)

    def compute(self):
        if len(self.rr_buffer) < 30:  # need minimum beats
            return {"status": "insufficient_data"}

        rr_array = np.array(self.rr_buffer)

        # Time-domain (always reliable from RR intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_array) ** 2))
        sdnn = np.std(rr_array, ddof=1)
        pnn50 = np.sum(np.abs(np.diff(rr_array)) > 50) / len(np.diff(rr_array)) * 100

        return {
            "rmssd_ms": round(rmssd, 1),
            "sdnn_ms": round(sdnn, 1),
            "pnn50_pct": round(pnn50, 1),
            "mean_rr_ms": round(np.mean(rr_array), 1),
            "rr_count": len(rr_array),
            "status": "ok"
        }
```

**Stage 3 Output** — a structured JSON every 1 second:

```json
{
    "timestamp": "2026-03-27T14:32:15.000Z",
    "sqi": 0.82,
    "sqi_label": "good",
    "hr_bpm": 98,
    "r_peaks": [45, 128, 211, 296, 382, 468, 553],
    "qrs_ms": 92,
    "hrv": {
        "rmssd_ms": 28.4,
        "sdnn_ms": 45.2,
        "pnn50_pct": 12.3,
        "status": "ok"
    },
    "activity_phase": "walking",
    "ecg_window_raw": [/* 650 samples for UI display */]
}
```

---

### STAGE 4: CLEF Embedding (Every 10 Seconds)

**What happens**: A 10-second ECG window (resampled to 500Hz = 5,000 samples) is passed through the frozen CLEF-Medium encoder. The output is a dense vector that encodes the cardiac morphology in a clinically meaningful space — similar patients produce similar embeddings.

```python
# clef_encoder.py
import torch
import numpy as np
from scipy.signal import resample, butter, sosfilt

class CLEFEncoder:
    def __init__(self, checkpoint_path="clef_medium.ckpt", device="cuda"):
        self.device = device
        # Load CLEF model (ResNeXt1D architecture)
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        self.model.to(device)

    def _load_model(self, path):
        # Follow clef_quickstart.ipynb loading pattern
        from models.resnet1d import ResNeXt1D  # from CLEF repo
        checkpoint = torch.load(path, map_location="cpu")
        model = ResNeXt1D(**checkpoint["hyper_parameters"])
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def preprocess(self, ecg_130hz, target_sr=500, target_len=5000):
        """130Hz raw → 500Hz normalized 10s window"""
        # Resample
        ecg_500 = resample(ecg_130hz, target_len)

        # Bandpass 0.5-40Hz
        sos = butter(4, [0.5, 40], btype='band', fs=target_sr, output='sos')
        ecg_filtered = sosfilt(sos, ecg_500)

        # Z-score normalize
        ecg_norm = (ecg_filtered - np.mean(ecg_filtered)) / (np.std(ecg_filtered) + 1e-8)

        return ecg_norm

    @torch.no_grad()
    def encode(self, ecg_130hz_10s):
        """Raw 130Hz 10s window → embedding vector"""
        processed = self.preprocess(ecg_130hz_10s)
        tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)  # → (1, h)
        # L2 normalize for cosine similarity
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().flatten()
```

**What the embedding captures**: Because CLEF was trained with SCORE2 cardiovascular risk score weighting, nearby embeddings come from patients with similar cardiovascular risk profiles. Two ECGs that "look similar" to a cardiologist will have high cosine similarity. This is more powerful than hand-crafted features because it captures subtle morphological patterns that rules-based systems miss.

**Stage 4 Output**: A single float vector (512-2048 dimensions depending on CLEF checkpoint), produced every 10 seconds, taking ~3-5ms on GPU.

---

### STAGE 5: Feature Aggregation

**What happens**: All outputs from Stages 3 and 4 are merged into a single structured payload that represents "everything we know about this patient right now." This is the bridge between the signal world and the language world.

```python
# feature_aggregator.py

def aggregate_features(
    signal_result,      # from Stage 3
    clef_embedding,     # from Stage 4
    retrieval_result,   # from Stage 6
    session_context,    # running session state
    patient_profile     # demographics, meds, history
):
    return {
        "current_window": {
            "timestamp": signal_result["timestamp"],
            "sqi": signal_result["sqi"],
            "hr_bpm": signal_result["hr_bpm"],
            "qrs_ms": signal_result.get("qrs_ms"),
            "activity_phase": signal_result.get("activity_phase", "unknown"),
        },
        "hrv": signal_result["hrv"],
        "foundation_model": {
            "beat_classification": retrieval_result["beat_class"],
            "beat_similarity": retrieval_result["beat_confidence"],
            "exercise_phase_match": retrieval_result["exercise_phase"],
            "exercise_similarity": retrieval_result["exercise_confidence"],
            "anomaly_score": retrieval_result["anomaly_score"],
            "nearest_frailty_profile": retrieval_result["nearest_frailty"],
        },
        "session": {
            "duration_min": session_context["elapsed_minutes"],
            "peak_hr": session_context["peak_hr"],
            "resting_hr": session_context["resting_hr"],
            "hr_recovery_1min": session_context.get("hrr_1min"),
            "phase": session_context["current_phase"],  # rest/exercise/recovery
            "hr_trend_5min": session_context["hr_trend"],
        },
        "patient": {
            "age": patient_profile.get("age"),
            "nyha_class": patient_profile.get("nyha_class"),
            "beta_blocker": patient_profile.get("beta_blocker", False),
            "surgery_type": patient_profile.get("surgery_type"),
            "days_post_surgery": patient_profile.get("days_post_surgery"),
        }
    }
```

---

### STAGE 6: ChromaDB Similarity Retrieval

**What happens**: The CLEF embedding is searched against three pre-computed reference banks. The system finds the most similar labeled ECG patterns from clinical datasets, giving the LLM grounded population context.

**Three reference collections** (pre-computed during hours 4-12 of the hackathon):

| Collection | Source | Embeddings | What It Tells You |
|-----------|--------|-----------|-------------------|
| `mitbih_beats` | MIT-BIH | ~75,000 | "This beat looks like a PVC / Normal / BBB" |
| `epfl_exercise` | EPFL | ~100 | "This pattern matches near-VO2max intensity" |
| `frailty_rehab` | Wearable Frailty | ~3,000 | "This resembles Patient 47, EFS 6, during 6MWT" |

```python
# retrieval.py
import chromadb
import numpy as np

class SimilarityEngine:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.beat_collection = self.client.get_collection("mitbih_beats")
        self.exercise_collection = self.client.get_collection("epfl_exercise")
        self.frailty_collection = self.client.get_collection("frailty_rehab")

        # Pre-compute normal centroid for anomaly scoring
        normal_embeddings = self.beat_collection.get(
            where={"aami_class": "N"}, include=["embeddings"], limit=1000
        )
        self.normal_centroid = np.mean(normal_embeddings["embeddings"], axis=0)

    def query(self, embedding, k=5):
        result = {}

        # Beat classification (k-NN vote from MIT-BIH)
        beat_results = self.beat_collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
            include=["metadatas", "distances"]
        )
        classes = [m["aami_class"] for m in beat_results["metadatas"][0]]
        from collections import Counter
        vote = Counter(classes).most_common(1)[0]
        result["beat_class"] = vote[0]
        result["beat_confidence"] = round(vote[1] / k, 2)
        result["beat_neighbors"] = [
            {"class": m["aami_class"], "record": m["record"],
             "distance": round(d, 4)}
            for m, d in zip(beat_results["metadatas"][0],
                           beat_results["distances"][0])
        ]

        # Exercise phase matching (from EPFL)
        ex_results = self.exercise_collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=3,
            include=["metadatas", "distances"]
        )
        result["exercise_phase"] = ex_results["metadatas"][0][0].get("segment", "unknown")
        result["exercise_confidence"] = round(1 - ex_results["distances"][0][0], 3)

        # Frailty population context
        frailty_results = self.frailty_collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=3,
            include=["metadatas", "distances"]
        )
        nearest = frailty_results["metadatas"][0][0]
        result["nearest_frailty"] = {
            "efs_score": nearest.get("efs_score"),
            "exercise_type": nearest.get("exercise_type"),
            "days_post_surgery": nearest.get("days_post_surgery"),
            "similarity": round(1 - frailty_results["distances"][0][0], 3)
        }

        # Anomaly score (cosine distance from normal centroid)
        cosine_dist = 1 - np.dot(embedding, self.normal_centroid) / (
            np.linalg.norm(embedding) * np.linalg.norm(self.normal_centroid) + 1e-8
        )
        result["anomaly_score"] = round(cosine_dist, 4)

        return result
```

**What "92% match to record 207-PVC" means concretely**: The CLEF embedding of the current 10-second ECG window has cosine similarity of 0.92 with a pre-computed embedding from MIT-BIH record 207, which was annotated as a premature ventricular contraction by two independent cardiologists. The LLM doesn't interpret the ECG — it receives this structured retrieval result and narrates it.

---

### STAGE 7: LLM Narrative Generation (Dual Persona)

**What happens**: The aggregated features + retrieval results are formatted into a structured prompt and sent to the local LLM. Two personas generate different outputs from the same data.

**7a. The System Prompt** (loaded once at startup):

```python
SYSTEM_PROMPT = """You are a cardiac rehabilitation AI assistant running locally
on-device. You analyze structured ECG metrics and foundation model retrieval
results from a Polar H10 chest strap. You have two modes:

MODE: PATIENT_COACH
- Speak in warm, encouraging language. No jargon.
- Celebrate specific achievements ("Great job on those stairs!")
- Frame metrics as progress indicators, not diagnoses
- If anything concerning: "You might want to mention this to your care team"
- NEVER diagnose. NEVER say "you have [condition]"

MODE: CLINICIAN_SCRIBE
- Use precise medical terminology
- Frame everything as "observations for clinician review"
- Generate structured SOAP notes when requested
- Include confidence levels and data quality caveats
- Reference specific retrieval matches (e.g., "92% similarity to MIT-BIH record 207")

CRITICAL SAFETY RULES:
- All ECG interpretations are OBSERVATIONS, not diagnoses
- Always note signal quality limitations
- Similarity retrieval is population context, not classification
- ST-segment analysis from single-lead 130Hz is informational only
- This system is NOT an FDA-cleared medical device"""
```

**7b. The Per-Window Prompt** (every 30 seconds or on clinical events):

```python
def build_llm_prompt(features, mode="PATIENT_COACH", rag_context=""):
    return f"""## Current Cardiac Analysis

**Signal Quality:** {features["current_window"]["sqi"]} ({"good" if features["current_window"]["sqi"] > 0.7 else "moderate" if features["current_window"]["sqi"] > 0.5 else "poor"})
**Heart Rate:** {features["current_window"]["hr_bpm"]} bpm
**Activity Phase:** {features["current_window"]["activity_phase"]}

**HRV (from RR intervals):**
- RMSSD: {features["hrv"].get("rmssd_ms", "N/A")} ms
- SDNN: {features["hrv"].get("sdnn_ms", "N/A")} ms

**Foundation Model Analysis (CLEF → ChromaDB):**
- Beat classification (k=5 neighbors): {features["foundation_model"]["beat_classification"]} (confidence: {features["foundation_model"]["beat_similarity"]})
- Exercise intensity match: {features["foundation_model"]["exercise_phase_match"]} (similarity: {features["foundation_model"]["exercise_similarity"]})
- Anomaly score: {features["foundation_model"]["anomaly_score"]} (threshold: 0.3)
- Nearest rehab profile: EFS={features["foundation_model"]["nearest_frailty_profile"].get("efs_score")}, {features["foundation_model"]["nearest_frailty_profile"].get("exercise_type")}, {features["foundation_model"]["nearest_frailty_profile"].get("days_post_surgery")} days post-surgery

**Session Context:**
- Duration: {features["session"]["duration_min"]} minutes
- Peak HR: {features["session"]["peak_hr"]} | Resting HR: {features["session"]["resting_hr"]}
- HR Recovery (1 min): {features["session"].get("hr_recovery_1min", "N/A")} bpm
- Phase: {features["session"]["phase"]}

**Patient Profile:**
- Age: {features["patient"].get("age")} | NYHA: {features["patient"].get("nyha_class")}
- Beta-blocker: {features["patient"].get("beta_blocker")}

**Relevant Guidelines:**
{rag_context}

---
MODE: {mode}
Provide a concise response (3-4 sentences for coach, structured SOAP for scribe)."""
```

**7c. RAG Retrieval** (fetches relevant AHA/AACVPR guidelines):

```python
# rag.py
from sentence_transformers import SentenceTransformer

class GuidelineRAG:
    def __init__(self, chroma_client):
        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.collection = chroma_client.get_collection("aha_guidelines")

    def retrieve(self, query, k=3):
        query_emb = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_emb], n_results=k,
            include=["documents", "metadatas"]
        )
        context_parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context_parts.append(
                f"[{meta['source']}, {meta['section']}]: {doc[:300]}..."
            )
        return "\n".join(context_parts)
```

**7d. vLLM Streaming Call**:

```python
# vllm_client.py
import httpx

async def stream_narrative(prompt, system_prompt, max_tokens=512):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "nvidia/Qwen3-32B-FP4",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,  # low for clinical consistency
                "stream": True
            },
            timeout=60.0
        )
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                token = chunk["choices"][0]["delta"].get("content", "")
                if token:
                    yield token
```

---

## What You Actually Get: Concrete Outputs

### For the Patient (via Flutter app):

1. **Live ECG trace** — 5-second scrolling waveform with R-peak markers, color-coded by SQI (green/yellow/red)
2. **Current HR gauge** — large number with trend arrow (↑↓→)
3. **Activity badge** — "Resting" / "Walking" / "Exercising" / "Recovering"
4. **Coach messages** — streaming text like:
   > "Nice work on that walk! Your heart rate came down 22 bpm in the first minute of rest — that's a strong recovery sign. Your heart rhythm has been steady throughout, matching healthy exercise patterns in our reference population."
5. **Session summary** — generated at end of exercise with HR curve, recovery metrics, and progress narrative

### For the Clinician (SOAP note + PDF):

```
SUBJECTIVE:
Patient reports feeling "good" during today's session. Borg RPE 12/20.
No chest pain, dyspnea, or dizziness reported.

OBJECTIVE:
- Monitoring: Polar H10 single-lead ECG, continuous 45 minutes
- Signal Quality: Mean SQI 0.78 (good), 3 windows <0.5 (flagged)
- Peak HR: 118 bpm (74% age-predicted max, 68% HRR)
- Resting HR: 72 bpm | Recovery HR (1 min): 96 bpm (HRR₁ = 22 bpm)
- HRV: RMSSD 32.4 ms (rest), 8.2 ms (peak), 24.1 ms (recovery +3 min)
- ECG Morphology: QRS 88ms, no ST changes detected at this lead/resolution
- Foundation Model: 94% of windows classified as Normal sinus rhythm
  (k=5 MIT-BIH). 2 windows showed PVC-like morphology (3.4% burden).
  Exercise phase matched "moderate intensity, sub-VT2" in EPFL reference.
- Population Context: Recovery trajectory comparable to EFS 4-5 patients
  at 21 days post-CABG in Wearable Frailty reference cohort.

ASSESSMENT:
Appropriate chronotropic response. HR recovery within normal limits
(HRR₁ > 12 bpm). Low PVC burden, consistent with benign exercise-related
ectopy. Autonomic recovery (RMSSD rebound) appropriate.
Functional capacity trending positively relative to reference population.

PLAN:
Continue current exercise prescription. Consider increasing intensity
to 75-80% HRR at next session if tolerated. Monitor PVC burden trend.
Follow up on HRV weekly trend for autonomic adaptation markers.
```

### The PDF "Opportunistic Stress Test Report":
- Patient demographics + session metadata
- HR curve overlaid on age/sex percentile bands (from 992 treadmill tests)
- HR recovery waterfall chart (color-coded: red <12, yellow 12-20, green >20 bpm)
- ECG embedding trajectory on UMAP (rest → exercise → recovery path)
- Morphological stability trend (% normal beats per 5-minute epoch)
- VO2max estimate via Uth-Sørensen formula with percentile
- Foundation model retrieval summary with confidence scores
- Disclaimer: "Not a diagnostic device. For wellness and educational purposes only."

---

## The Orchestrator: Tying It All Together

```python
# orchestrator.py
import asyncio

class PipelineOrchestrator:
    def __init__(self):
        self.ecg_queue = asyncio.Queue(maxsize=1000)
        self.rr_queue = asyncio.Queue(maxsize=500)
        self.feature_queue = asyncio.Queue(maxsize=50)
        self.narrative_queue = asyncio.Queue(maxsize=10)

        self.polar = PolarH10Client(self.ecg_queue, self.rr_queue)
        self.buffer = ECGRingBuffer()
        self.clef = CLEFEncoder("clef_medium.ckpt")
        self.retrieval = SimilarityEngine("./chroma_db")
        self.hrv_computer = HRVComputer()
        self.rag = GuidelineRAG(self.retrieval.client)
        self.session = SessionState()

    async def ble_collector(self):
        """Task 1: BLE → Ring Buffer"""
        await self.polar.connect_and_stream("POLAR_H10_DEVICE_ID")
        while True:
            data = await self.ecg_queue.get()
            self.buffer.append(data["samples"], data["timestamp"])

    async def rr_collector(self):
        """Task 1b: RR intervals → HRV computer"""
        while True:
            data = await self.rr_queue.get()
            self.hrv_computer.add_rr(data["rr_intervals_ms"])

    async def signal_processor(self):
        """Task 2: Ring Buffer → Features (every 1 second)"""
        while True:
            await asyncio.sleep(1.0)
            window = self.buffer.get_window(5)
            if window is None:
                continue

            # Signal quality gate
            sqi, sqi_detail = compute_sqi(window, sr=130)

            # ECG processing
            ecg_result = process_ecg_window(window, sr=130, sqi_score=sqi)
            ecg_result["sqi"] = sqi
            ecg_result["hrv"] = self.hrv_computer.compute()

            # Update session state
            self.session.update(ecg_result)

            # Send to feature queue for display
            await self.feature_queue.put(ecg_result)

    async def embedding_and_retrieval(self):
        """Task 3: CLEF → ChromaDB (every 10 seconds)"""
        while True:
            await asyncio.sleep(10.0)
            window_10s = self.buffer.get_window_for_clef(10)
            if window_10s is None:
                continue

            # CLEF embedding (~3-5ms on GPU)
            embedding = self.clef.encode(window_10s)

            # ChromaDB retrieval (~5-15ms)
            retrieval = self.retrieval.query(embedding)

            # Store for next narrative generation
            self.session.latest_retrieval = retrieval
            self.session.latest_embedding = embedding

    async def narrative_generator(self):
        """Task 4: Features → LLM → WebSocket (every 30 seconds)"""
        while True:
            await asyncio.sleep(30.0)
            if not self.session.latest_retrieval:
                continue

            features = aggregate_features(
                self.session.latest_signal,
                self.session.latest_embedding,
                self.session.latest_retrieval,
                self.session.get_context(),
                self.session.patient_profile
            )

            # RAG: fetch relevant guidelines
            query = f"cardiac rehab {features['current_window']['activity_phase']} HR {features['current_window']['hr_bpm']}"
            rag_context = self.rag.retrieve(query, k=2)

            prompt = build_llm_prompt(features, mode="PATIENT_COACH", rag_context=rag_context)

            narrative = ""
            async for token in stream_narrative(prompt, SYSTEM_PROMPT):
                narrative += token
                await self.narrative_queue.put({"type": "token", "text": token})

            await self.narrative_queue.put({"type": "complete", "text": narrative})

    async def websocket_server(self):
        """Task 5: Stream everything to Flutter via WebSocket"""
        from fastapi import FastAPI, WebSocket
        import uvicorn

        app = FastAPI()

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                # Multiplex: send ECG data, features, and narrative tokens
                try:
                    feature = self.feature_queue.get_nowait()
                    await websocket.send_json({"type": "ecg_features", "data": feature})
                except asyncio.QueueEmpty:
                    pass

                try:
                    narrative = self.narrative_queue.get_nowait()
                    await websocket.send_json({"type": "narrative", "data": narrative})
                except asyncio.QueueEmpty:
                    pass

                await asyncio.sleep(0.05)  # 20Hz update rate

        config = uvicorn.Config(app, host="0.0.0.0", port=8080)
        server = uvicorn.Server(config)
        await server.serve()

    async def run(self):
        """Launch all concurrent tasks"""
        await asyncio.gather(
            self.ble_collector(),
            self.rr_collector(),
            self.signal_processor(),
            self.embedding_and_retrieval(),
            self.narrative_generator(),
            self.websocket_server(),
        )

if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    asyncio.run(orchestrator.run())
```

---

## Memory Budget: What Actually Fits

| Component | Memory | When Loaded |
|-----------|--------|-------------|
| Qwen3-32B-FP4 (via vLLM) | ~18 GB | Hour 0 |
| KV cache (4096 ctx, 2 seqs, FP8) | ~8 GB | Dynamic |
| CLEF-Medium (FP16) | 60 MB | Hour 4 |
| BGE-base-en-v1.5 (text embedding) | 440 MB | Hour 8 |
| ChromaDB (80K ECG + guidelines) | 500 MB | Hour 12 |
| NormWear (optional, ECG+ACC) | 200 MB | Stretch goal |
| ECG-FM finetuned head (optional) | 175 MB | Stretch goal |
| OS + Docker + Python + CUDA | ~10 GB | Always |
| **Total** | **~28 GB** | |
| **Free** | **~100 GB** | |

If the demo is going well and you want to upgrade: swap Qwen3-32B-FP4 for HuatuoGPT-o1-72B-AWQ (~40 GB) during hours 36-48. You still have 60+ GB free. The decode will be slower (4-8 tok/s vs 15-20) but the medical reasoning quality jumps dramatically. Pre-generate the session summary and SOAP note offline (not streamed live) to hide the latency.

---

## 48-Hour Build Schedule

| Hours | Owner | Deliverable |
|-------|-------|-------------|
| **0-4** | Viggi (DGX) | vLLM running with Qwen3-32B-FP4; verify with test prompt |
| **0-4** | Rumon (HW) | Polar H10 BLE streaming confirmed via polar-python |
| **0-4** | Shiva (AI) | CLEF repo cloned, quickstart notebook runs, embedding verified |
| **0-4** | Sansrit (Flutter) | WebSocket client connecting to FastAPI stub |
| **4-12** | Shiva | Reference bank precomputation: MIT-BIH → CLEF → ChromaDB |
| **4-12** | Viggi | AHA guidelines chunked → BGE embedded → ChromaDB |
| **4-12** | Rumon | Ring buffer + SQI + ECG processing pipeline complete |
| **4-12** | Sansrit | Live ECG trace rendering + HR gauge in Flutter |
| **12-24** | Shiva | Retrieval engine (3 collections) + feature aggregator |
| **12-24** | Rumon | HRV from RR intervals + activity phase detection |
| **12-24** | Viggi | LLM prompt templates + RAG integration + streaming |
| **12-24** | Sansrit | Coach narrative panel + session summary UI |
| **24-36** | ALL | Integration: full pipeline end-to-end test with live H10 |
| **24-36** | Shiva | UMAP embedding visualization (the "wow" visual) |
| **24-36** | Viggi | SOAP note generation + PDF report |
| **24-36** | Sansrit | Clinician dashboard view |
| **36-48** | ALL | Polish, demo rehearsal, edge case handling |
| **36-48** | Optional | Swap in HuatuoGPT-o1-72B for better reasoning quality |

---

## Alert Rules (Deterministic, NOT LLM-Driven)

These fire independently of the LLM and are always active:

| Condition | Threshold | Action |
|-----------|----------|--------|
| HR too high | > 85% age-predicted max | Yellow alert + coach warning |
| HR dangerously high | > 100% age-predicted max | Red alert + "Stop and rest" |
| HR too low | < 40 bpm | Red alert |
| Poor HR recovery | HRR₁ < 12 bpm | Flag for clinician review |
| High PVC burden | > 10% of beats PVC-like | Yellow alert |
| Signal loss | SQI < 0.3 for > 30s | "Check electrode contact" |
| Electrode off | HR = 0 for > 5s | "Sensor disconnected" |
| RR irregularity | > 20% variation at rest | Flag for clinician |

These never go through the LLM. They are `if` statements in Python that push directly to the WebSocket.