# Talk to Your Heart — End-to-End Pipeline Blueprint

## Model Decision: The Three-Tier Strategy

Before detailing the pipeline, here's the model recommendation with a clear decision framework.

### Primary LLM: Pick ONE based on your hackathon risk tolerance

|Strategy|Model|INT4 Memory|Decode Speed|MedQA|Risk Level|
|-|-|-|-|-|-|
|**Safe bet**|`nvidia/Qwen3-32B-FP4`|\~18GB|\~15-20 tok/s|Competitive|LOW — NVIDIA pre-validated on Spark|
|**Best medical**|MedGemma-27B-text v1.5|\~14GB|\~18-22 tok/s|**87.7%**|MEDIUM — needs Gemma tokenizer setup|
|**Maximum reasoning**|HuatuoGPT-o1-72B (AWQ)|\~40GB|\~4-8 tok/s|Top-tier|HIGH — slow decode, large KV cache|
|**Fastest fallback**|m1-7B-23K|\~4GB|\~40-50 tok/s|60.3%|LOWEST — snappy but less knowledgeable|

**Recommended approach**: Start with `nvidia/Qwen3-32B-FP4` (pre-quantized, validated on DGX Spark, native `<think>` reasoning blocks). If you get it running in hours 0-4, try swapping to MedGemma-27B or HuatuoGPT-o1-72B during polish time. Keep m1-7B-23K as an emergency fallback that will always be fast enough for live demos.

**Why not HuatuoGPT-o1-72B as primary?** At \~40GB INT4, it leaves only \~78GB for everything else. More critically, 4-8 tok/s decode means a 200-token narrative takes 25-50 seconds — too slow for a live demo's "talk to your heart and it talks back" moment. Use it only if you pre-generate session summaries (not real-time).

**Why Qwen3-32B over m1-32B-1K?** Qwen3's native `<think>` / `</think>` blocks give you test-time scaling without any custom inference hacking. m1-32B-1K requires manual "Wait" token injection that vLLM doesn't natively support. Qwen3-32B with a medical system prompt + RAG gets you 90% of m1's benefit with zero custom inference code.

\---

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

\---

### STAGE 1: Polar H10 BLE Data Acquisition

**What happens**: The Polar H10 chest strap streams two independent data channels over Bluetooth Low Energy to the DGX Spark backend.

**Data streams**:

* **Raw ECG**: 130 Hz, 14-bit resolution, \~73 samples per BLE packet (\~560ms intervals). Single-lead approximating V4-V5 chest position.
* **RR Intervals**: 1ms resolution timestamps between consecutive R-peaks. Computed on-device by Polar's firmware. This is your primary HRV source (bypasses the 130Hz jitter problem).
* **Accelerometer** (optional): 25-200 Hz configurable, 3-axis, 16-bit. For activity phase detection.

**Implementation**:

```python
# polar\_client.py
from polar\_python import PolarDevice, MeasurementType
import asyncio

class PolarH10Client:
    def \_\_init\_\_(self, ecg\_queue, rr\_queue, acc\_queue=None):
        self.ecg\_queue = ecg\_queue
        self.rr\_queue = rr\_queue
        self.acc\_queue = acc\_queue

    async def connect\_and\_stream(self, device\_id: str):
        device = PolarDevice(device\_id)
        await device.connect()

        # Start ECG stream (130Hz)
        await device.start\_measurement(
            MeasurementType.ECG,
            callback=self.\_on\_ecg\_data
        )
        # Start RR interval stream
        await device.start\_measurement(
            MeasurementType.RR\_INTERVAL,
            callback=self.\_on\_rr\_data
        )
        # Optional: accelerometer for activity detection
        if self.acc\_queue:
            await device.start\_measurement(
                MeasurementType.ACC,
                settings={"sample\_rate": 50, "range": 8},
                callback=self.\_on\_acc\_data
            )

    async def \_on\_ecg\_data(self, data):
        # data.samples = list of int (µV values)
        # data.timestamp = nanosecond timestamp
        await self.ecg\_queue.put({
            "samples": data.samples,
            "timestamp": data.timestamp,
            "type": "ecg"
        })

    async def \_on\_rr\_data(self, data):
        await self.rr\_queue.put({
            "rr\_intervals\_ms": data.rr\_intervals,
            "timestamp": data.timestamp,
            "type": "rr"
        })
```

**Output**: Raw ECG samples and RR intervals flowing into bounded asyncio queues.

**Failure modes to handle**: BLE disconnection (auto-reconnect with exponential backoff), electrode contact loss (HR drops to 0 — detect and flag), packet loss (timestamp gaps > 600ms).

\---

### STAGE 2: Ring Buffer with Sliding Windows

**What happens**: Raw ECG samples accumulate in a circular buffer. Every 1 second, a 5-second window (650 samples at 130Hz) is extracted for signal processing. Every 10 seconds, a 10-second window (1,300 samples) is extracted and resampled for CLEF embedding.

**Two buffer outputs**:

* **5-second windows** (1s stride) → Stage 3 (signal processing, SQI, R-peaks)
* **10-second windows** (5s stride) → Stage 4 (CLEF embedding inference)

```python
# ring\_buffer.py
import numpy as np
from collections import deque

class ECGRingBuffer:
    def \_\_init\_\_(self, max\_seconds=30, sample\_rate=130):
        self.sr = sample\_rate
        self.buffer = deque(maxlen=max\_seconds \* sample\_rate)
        self.timestamps = deque(maxlen=max\_seconds \* sample\_rate)

    def append(self, samples, timestamp):
        for i, s in enumerate(samples):
            self.buffer.append(s)
            self.timestamps.append(timestamp + i \* (1e9 / self.sr))

    def get\_window(self, seconds=5):
        n = seconds \* self.sr
        if len(self.buffer) < n:
            return None
        return np.array(list(self.buffer)\[-n:], dtype=np.float64)

    def get\_window\_for\_clef(self, seconds=10):
        """Returns 10s window resampled to 500Hz (5000 samples)"""
        raw = self.get\_window(seconds)
        if raw is None:
            return None
        from scipy.signal import resample
        return resample(raw, 5000)  # 130Hz→500Hz
```

**Output**: Numpy arrays of ECG samples, ready for processing.

\---

### STAGE 3: Signal Processing \& Quality Gating

**What happens**: Each 5-second window goes through a deterministic signal processing pipeline. This stage produces ALL the hard numbers — HR, HRV, SQI, R-peaks, QRS morphology. No AI involved. These are the safety-critical metrics.

**Sub-steps**:

**3a. Signal Quality Index (SQI)** — computed FIRST, gates everything else:

```python
# sqi.py
import neurokit2 as nk
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import welch

def compute\_sqi(ecg\_window, sr=130):
    """4-component SQI → float 0.0-1.0"""

    # pSQI: QRS power ratio (5-15Hz / 5-40Hz)
    f, psd = welch(ecg\_window, fs=sr, nperseg=min(256, len(ecg\_window)))
    qrs\_power = np.trapz(psd\[(f >= 5) \& (f <= 15)], f\[(f >= 5) \& (f <= 15)])
    total\_power = np.trapz(psd\[(f >= 5) \& (f <= 40)], f\[(f >= 5) \& (f <= 40)])
    p\_sqi = qrs\_power / (total\_power + 1e-10)
    p\_sqi\_score = 1.0 if 0.4 <= p\_sqi <= 0.8 else max(0, 1 - abs(p\_sqi - 0.6) \* 2)

    # kSQI: kurtosis (sharp R-peaks → high kurtosis)
    k = kurtosis(ecg\_window, fisher=True)
    k\_sqi = min(1.0, max(0, (k - 2) / 8))  # normalize: 2→0, 10→1

    # basSQI: baseline wander (power 0-1Hz / 0-40Hz)
    baseline\_power = np.trapz(psd\[(f >= 0) \& (f <= 1)], f\[(f >= 0) \& (f <= 1)])
    total\_40 = np.trapz(psd\[(f >= 0) \& (f <= 40)], f\[(f >= 0) \& (f <= 40)])
    bas\_sqi = 1 - (baseline\_power / (total\_40 + 1e-10))

    # qSQI: dual-detector agreement
    try:
        peaks\_nk = nk.ecg\_findpeaks(ecg\_window, sampling\_rate=sr, method="neurokit")\["ECG\_R\_Peaks"]
        peaks\_ham = nk.ecg\_findpeaks(ecg\_window, sampling\_rate=sr, method="hamilton2002")\["ECG\_R\_Peaks"]
        if len(peaks\_nk) > 0 and len(peaks\_ham) > 0:
            matches = sum(1 for p in peaks\_nk if any(abs(p - h) < sr \* 0.15 for h in peaks\_ham))
            q\_sqi = matches / max(len(peaks\_nk), len(peaks\_ham))
        else:
            q\_sqi = 0.0
    except:
        q\_sqi = 0.5

    # Weighted combination
    combined = 0.3 \* q\_sqi + 0.3 \* p\_sqi\_score + 0.2 \* k\_sqi + 0.2 \* bas\_sqi
    return round(combined, 3), {"q": q\_sqi, "p": p\_sqi\_score, "k": k\_sqi, "bas": bas\_sqi}
```

**3b. R-peak detection \& HR**:

```python
# ecg\_processing.py
import neurokit2 as nk
import heartpy as hp

def process\_ecg\_window(ecg\_window, sr=130, sqi\_score=1.0):
    """Full signal processing on a 5-second window"""
    result = {}

    # R-peak detection (NeuroKit2 default method)
    try:
        signals, info = nk.ecg\_process(ecg\_window, sampling\_rate=sr)
        r\_peaks = info\["ECG\_R\_Peaks"]
        result\["r\_peaks"] = r\_peaks.tolist()
        result\["hr\_bpm"] = round(signals\["ECG\_Rate"].mean(), 1)
    except:
        result\["r\_peaks"] = \[]
        result\["hr\_bpm"] = None

    # High-precision R-peak timing via HeartPy (for HRV)
    if sqi\_score > 0.5:
        try:
            wd, m = hp.process(ecg\_window, sample\_rate=sr,
                              high\_precision=True, high\_precision\_fs=1000)
            result\["rr\_intervals\_hp"] = wd\["RR\_list\_cor"]  # artifact-corrected
        except:
            pass

    # QRS morphology (only when SQI > 0.6)
    if sqi\_score > 0.6:
        try:
            \_, waves = nk.ecg\_delineate(ecg\_window, r\_peaks,
                                         sampling\_rate=sr, method="dwt")
            # Approximate QRS duration
            qrs\_onsets = waves.get("ECG\_Q\_Peaks", \[])
            qrs\_offsets = waves.get("ECG\_S\_Peaks", \[])
            if qrs\_onsets and qrs\_offsets:
                durations = \[(s - q) / sr \* 1000
                            for q, s in zip(qrs\_onsets, qrs\_offsets)
                            if q is not None and s is not None]
                result\["qrs\_ms"] = round(np.median(durations), 1) if durations else None
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
    def \_\_init\_\_(self, window\_minutes=5):
        self.rr\_buffer = \[]  # milliseconds
        self.window\_ms = window\_minutes \* 60 \* 1000

    def add\_rr(self, rr\_intervals\_ms):
        self.rr\_buffer.extend(rr\_intervals\_ms)
        # Trim to window
        total = sum(self.rr\_buffer)
        while total > self.window\_ms and len(self.rr\_buffer) > 10:
            total -= self.rr\_buffer.pop(0)

    def compute(self):
        if len(self.rr\_buffer) < 30:  # need minimum beats
            return {"status": "insufficient\_data"}

        rr\_array = np.array(self.rr\_buffer)

        # Time-domain (always reliable from RR intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr\_array) \*\* 2))
        sdnn = np.std(rr\_array, ddof=1)
        pnn50 = np.sum(np.abs(np.diff(rr\_array)) > 50) / len(np.diff(rr\_array)) \* 100

        return {
            "rmssd\_ms": round(rmssd, 1),
            "sdnn\_ms": round(sdnn, 1),
            "pnn50\_pct": round(pnn50, 1),
            "mean\_rr\_ms": round(np.mean(rr\_array), 1),
            "rr\_count": len(rr\_array),
            "status": "ok"
        }
```

**Stage 3 Output** — a structured JSON every 1 second:

```json
{
    "timestamp": "2026-03-27T14:32:15.000Z",
    "sqi": 0.82,
    "sqi\_label": "good",
    "hr\_bpm": 98,
    "r\_peaks": \[45, 128, 211, 296, 382, 468, 553],
    "qrs\_ms": 92,
    "hrv": {
        "rmssd\_ms": 28.4,
        "sdnn\_ms": 45.2,
        "pnn50\_pct": 12.3,
        "status": "ok"
    },
    "activity\_phase": "walking",
    "ecg\_window\_raw": \[/\* 650 samples for UI display \*/]
}
```

\---

### STAGE 4: CLEF Embedding (Every 10 Seconds)

**What happens**: A 10-second ECG window (resampled to 500Hz = 5,000 samples) is passed through the frozen CLEF-Medium encoder. The output is a dense vector that encodes the cardiac morphology in a clinically meaningful space — similar patients produce similar embeddings.

```python
# clef\_encoder.py
import torch
import numpy as np
from scipy.signal import resample, butter, sosfilt

class CLEFEncoder:
    def \_\_init\_\_(self, checkpoint\_path="clef\_medium.ckpt", device="cuda"):
        self.device = device
        # Load CLEF model (ResNeXt1D architecture)
        self.model = self.\_load\_model(checkpoint\_path)
        self.model.eval()
        self.model.to(device)

    def \_load\_model(self, path):
        # Follow clef\_quickstart.ipynb loading pattern
        from models.resnet1d import ResNeXt1D  # from CLEF repo
        checkpoint = torch.load(path, map\_location="cpu")
        model = ResNeXt1D(\*\*checkpoint\["hyper\_parameters"])
        model.load\_state\_dict(checkpoint\["state\_dict"])
        return model

    def preprocess(self, ecg\_130hz, target\_sr=500, target\_len=5000):
        """130Hz raw → 500Hz normalized 10s window"""
        # Resample
        ecg\_500 = resample(ecg\_130hz, target\_len)

        # Bandpass 0.5-40Hz
        sos = butter(4, \[0.5, 40], btype='band', fs=target\_sr, output='sos')
        ecg\_filtered = sosfilt(sos, ecg\_500)

        # Z-score normalize
        ecg\_norm = (ecg\_filtered - np.mean(ecg\_filtered)) / (np.std(ecg\_filtered) + 1e-8)

        return ecg\_norm

    @torch.no\_grad()
    def encode(self, ecg\_130hz\_10s):
        """Raw 130Hz 10s window → embedding vector"""
        processed = self.preprocess(ecg\_130hz\_10s)
        tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)  # → (1, h)
        # L2 normalize for cosine similarity
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().flatten()
```

**What the embedding captures**: Because CLEF was trained with SCORE2 cardiovascular risk score weighting, nearby embeddings come from patients with similar cardiovascular risk profiles. Two ECGs that "look similar" to a cardiologist will have high cosine similarity. This is more powerful than hand-crafted features because it captures subtle morphological patterns that rules-based systems miss.

**Stage 4 Output**: A single float vector (512-2048 dimensions depending on CLEF checkpoint), produced every 10 seconds, taking \~3-5ms on GPU.

\---

### STAGE 5: Feature Aggregation

**What happens**: All outputs from Stages 3 and 4 are merged into a single structured payload that represents "everything we know about this patient right now." This is the bridge between the signal world and the language world.

```python
# feature\_aggregator.py

def aggregate\_features(
    signal\_result,      # from Stage 3
    clef\_embedding,     # from Stage 4
    retrieval\_result,   # from Stage 6
    session\_context,    # running session state
    patient\_profile     # demographics, meds, history
):
    return {
        "current\_window": {
            "timestamp": signal\_result\["timestamp"],
            "sqi": signal\_result\["sqi"],
            "hr\_bpm": signal\_result\["hr\_bpm"],
            "qrs\_ms": signal\_result.get("qrs\_ms"),
            "activity\_phase": signal\_result.get("activity\_phase", "unknown"),
        },
        "hrv": signal\_result\["hrv"],
        "foundation\_model": {
            "beat\_classification": retrieval\_result\["beat\_class"],
            "beat\_similarity": retrieval\_result\["beat\_confidence"],
            "exercise\_phase\_match": retrieval\_result\["exercise\_phase"],
            "exercise\_similarity": retrieval\_result\["exercise\_confidence"],
            "anomaly\_score": retrieval\_result\["anomaly\_score"],
            "nearest\_frailty\_profile": retrieval\_result\["nearest\_frailty"],
        },
        "session": {
            "duration\_min": session\_context\["elapsed\_minutes"],
            "peak\_hr": session\_context\["peak\_hr"],
            "resting\_hr": session\_context\["resting\_hr"],
            "hr\_recovery\_1min": session\_context.get("hrr\_1min"),
            "phase": session\_context\["current\_phase"],  # rest/exercise/recovery
            "hr\_trend\_5min": session\_context\["hr\_trend"],
        },
        "patient": {
            "age": patient\_profile.get("age"),
            "nyha\_class": patient\_profile.get("nyha\_class"),
            "beta\_blocker": patient\_profile.get("beta\_blocker", False),
            "surgery\_type": patient\_profile.get("surgery\_type"),
            "days\_post\_surgery": patient\_profile.get("days\_post\_surgery"),
        }
    }
```

\---

### STAGE 6: ChromaDB Similarity Retrieval

**What happens**: The CLEF embedding is searched against three pre-computed reference banks. The system finds the most similar labeled ECG patterns from clinical datasets, giving the LLM grounded population context.

**Three reference collections** (pre-computed during hours 4-12 of the hackathon):

|Collection|Source|Embeddings|What It Tells You|
|-|-|-|-|
|`mitbih\_beats`|MIT-BIH|\~75,000|"This beat looks like a PVC / Normal / BBB"|
|`epfl\_exercise`|EPFL|\~100|"This pattern matches near-VO2max intensity"|
|`frailty\_rehab`|Wearable Frailty|\~3,000|"This resembles Patient 47, EFS 6, during 6MWT"|

```python
# retrieval.py
import chromadb
import numpy as np

class SimilarityEngine:
    def \_\_init\_\_(self, db\_path="./chroma\_db"):
        self.client = chromadb.PersistentClient(path=db\_path)
        self.beat\_collection = self.client.get\_collection("mitbih\_beats")
        self.exercise\_collection = self.client.get\_collection("epfl\_exercise")
        self.frailty\_collection = self.client.get\_collection("frailty\_rehab")

        # Pre-compute normal centroid for anomaly scoring
        normal\_embeddings = self.beat\_collection.get(
            where={"aami\_class": "N"}, include=\["embeddings"], limit=1000
        )
        self.normal\_centroid = np.mean(normal\_embeddings\["embeddings"], axis=0)

    def query(self, embedding, k=5):
        result = {}

        # Beat classification (k-NN vote from MIT-BIH)
        beat\_results = self.beat\_collection.query(
            query\_embeddings=\[embedding.tolist()],
            n\_results=k,
            include=\["metadatas", "distances"]
        )
        classes = \[m\["aami\_class"] for m in beat\_results\["metadatas"]\[0]]
        from collections import Counter
        vote = Counter(classes).most\_common(1)\[0]
        result\["beat\_class"] = vote\[0]
        result\["beat\_confidence"] = round(vote\[1] / k, 2)
        result\["beat\_neighbors"] = \[
            {"class": m\["aami\_class"], "record": m\["record"],
             "distance": round(d, 4)}
            for m, d in zip(beat\_results\["metadatas"]\[0],
                           beat\_results\["distances"]\[0])
        ]

        # Exercise phase matching (from EPFL)
        ex\_results = self.exercise\_collection.query(
            query\_embeddings=\[embedding.tolist()],
            n\_results=3,
            include=\["metadatas", "distances"]
        )
        result\["exercise\_phase"] = ex\_results\["metadatas"]\[0]\[0].get("segment", "unknown")
        result\["exercise\_confidence"] = round(1 - ex\_results\["distances"]\[0]\[0], 3)

        # Frailty population context
        frailty\_results = self.frailty\_collection.query(
            query\_embeddings=\[embedding.tolist()],
            n\_results=3,
            include=\["metadatas", "distances"]
        )
        nearest = frailty\_results\["metadatas"]\[0]\[0]
        result\["nearest\_frailty"] = {
            "efs\_score": nearest.get("efs\_score"),
            "exercise\_type": nearest.get("exercise\_type"),
            "days\_post\_surgery": nearest.get("days\_post\_surgery"),
            "similarity": round(1 - frailty\_results\["distances"]\[0]\[0], 3)
        }

        # Anomaly score (cosine distance from normal centroid)
        cosine\_dist = 1 - np.dot(embedding, self.normal\_centroid) / (
            np.linalg.norm(embedding) \* np.linalg.norm(self.normal\_centroid) + 1e-8
        )
        result\["anomaly\_score"] = round(cosine\_dist, 4)

        return result
```

**What "92% match to record 207-PVC" means concretely**: The CLEF embedding of the current 10-second ECG window has cosine similarity of 0.92 with a pre-computed embedding from MIT-BIH record 207, which was annotated as a premature ventricular contraction by two independent cardiologists. The LLM doesn't interpret the ECG — it receives this structured retrieval result and narrates it.

\---

### STAGE 7: LLM Narrative Generation (Dual Persona)

**What happens**: The aggregated features + retrieval results are formatted into a structured prompt and sent to the local LLM. Two personas generate different outputs from the same data.

**7a. The System Prompt** (loaded once at startup):

```python
SYSTEM\_PROMPT = """You are a cardiac rehabilitation AI assistant running locally
on-device. You analyze structured ECG metrics and foundation model retrieval
results from a Polar H10 chest strap. You have two modes:

MODE: PATIENT\_COACH
- Speak in warm, encouraging language. No jargon.
- Celebrate specific achievements ("Great job on those stairs!")
- Frame metrics as progress indicators, not diagnoses
- If anything concerning: "You might want to mention this to your care team"
- NEVER diagnose. NEVER say "you have \[condition]"

MODE: CLINICIAN\_SCRIBE
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
def build\_llm\_prompt(features, mode="PATIENT\_COACH", rag\_context=""):
    return f"""## Current Cardiac Analysis

\*\*Signal Quality:\*\* {features\["current\_window"]\["sqi"]} ({"good" if features\["current\_window"]\["sqi"] > 0.7 else "moderate" if features\["current\_window"]\["sqi"] > 0.5 else "poor"})
\*\*Heart Rate:\*\* {features\["current\_window"]\["hr\_bpm"]} bpm
\*\*Activity Phase:\*\* {features\["current\_window"]\["activity\_phase"]}

\*\*HRV (from RR intervals):\*\*
- RMSSD: {features\["hrv"].get("rmssd\_ms", "N/A")} ms
- SDNN: {features\["hrv"].get("sdnn\_ms", "N/A")} ms

\*\*Foundation Model Analysis (CLEF → ChromaDB):\*\*
- Beat classification (k=5 neighbors): {features\["foundation\_model"]\["beat\_classification"]} (confidence: {features\["foundation\_model"]\["beat\_similarity"]})
- Exercise intensity match: {features\["foundation\_model"]\["exercise\_phase\_match"]} (similarity: {features\["foundation\_model"]\["exercise\_similarity"]})
- Anomaly score: {features\["foundation\_model"]\["anomaly\_score"]} (threshold: 0.3)
- Nearest rehab profile: EFS={features\["foundation\_model"]\["nearest\_frailty\_profile"].get("efs\_score")}, {features\["foundation\_model"]\["nearest\_frailty\_profile"].get("exercise\_type")}, {features\["foundation\_model"]\["nearest\_frailty\_profile"].get("days\_post\_surgery")} days post-surgery

\*\*Session Context:\*\*
- Duration: {features\["session"]\["duration\_min"]} minutes
- Peak HR: {features\["session"]\["peak\_hr"]} | Resting HR: {features\["session"]\["resting\_hr"]}
- HR Recovery (1 min): {features\["session"].get("hr\_recovery\_1min", "N/A")} bpm
- Phase: {features\["session"]\["phase"]}

\*\*Patient Profile:\*\*
- Age: {features\["patient"].get("age")} | NYHA: {features\["patient"].get("nyha\_class")}
- Beta-blocker: {features\["patient"].get("beta\_blocker")}

\*\*Relevant Guidelines:\*\*
{rag\_context}

---
MODE: {mode}
Provide a concise response (3-4 sentences for coach, structured SOAP for scribe)."""
```

**7c. RAG Retrieval** (fetches relevant AHA/AACVPR guidelines):

```python
# rag.py
from sentence\_transformers import SentenceTransformer

class GuidelineRAG:
    def \_\_init\_\_(self, chroma\_client):
        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.collection = chroma\_client.get\_collection("aha\_guidelines")

    def retrieve(self, query, k=3):
        query\_emb = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query\_embeddings=\[query\_emb], n\_results=k,
            include=\["documents", "metadatas"]
        )
        context\_parts = \[]
        for doc, meta in zip(results\["documents"]\[0], results\["metadatas"]\[0]):
            context\_parts.append(
                f"\[{meta\['source']}, {meta\['section']}]: {doc\[:300]}..."
            )
        return "\\n".join(context\_parts)
```

**7d. vLLM Streaming Call**:

```python
# vllm\_client.py
import httpx

async def stream\_narrative(prompt, system\_prompt, max\_tokens=512):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "nvidia/Qwen3-32B-FP4",
                "messages": \[
                    {"role": "system", "content": system\_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max\_tokens": max\_tokens,
                "temperature": 0.3,  # low for clinical consistency
                "stream": True
            },
            timeout=60.0
        )
        async for line in response.aiter\_lines():
            if line.startswith("data: ") and line != "data: \[DONE]":
                chunk = json.loads(line\[6:])
                token = chunk\["choices"]\[0]\["delta"].get("content", "")
                if token:
                    yield token
```

\---

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

* Patient demographics + session metadata
* HR curve overlaid on age/sex percentile bands (from 992 treadmill tests)
* HR recovery waterfall chart (color-coded: red <12, yellow 12-20, green >20 bpm)
* ECG embedding trajectory on UMAP (rest → exercise → recovery path)
* Morphological stability trend (% normal beats per 5-minute epoch)
* VO2max estimate via Uth-Sørensen formula with percentile
* Foundation model retrieval summary with confidence scores
* Disclaimer: "Not a diagnostic device. For wellness and educational purposes only."

\---

## The Orchestrator: Tying It All Together

```python
# orchestrator.py
import asyncio

class PipelineOrchestrator:
    def \_\_init\_\_(self):
        self.ecg\_queue = asyncio.Queue(maxsize=1000)
        self.rr\_queue = asyncio.Queue(maxsize=500)
        self.feature\_queue = asyncio.Queue(maxsize=50)
        self.narrative\_queue = asyncio.Queue(maxsize=10)

        self.polar = PolarH10Client(self.ecg\_queue, self.rr\_queue)
        self.buffer = ECGRingBuffer()
        self.clef = CLEFEncoder("clef\_medium.ckpt")
        self.retrieval = SimilarityEngine("./chroma\_db")
        self.hrv\_computer = HRVComputer()
        self.rag = GuidelineRAG(self.retrieval.client)
        self.session = SessionState()

    async def ble\_collector(self):
        """Task 1: BLE → Ring Buffer"""
        await self.polar.connect\_and\_stream("POLAR\_H10\_DEVICE\_ID")
        while True:
            data = await self.ecg\_queue.get()
            self.buffer.append(data\["samples"], data\["timestamp"])

    async def rr\_collector(self):
        """Task 1b: RR intervals → HRV computer"""
        while True:
            data = await self.rr\_queue.get()
            self.hrv\_computer.add\_rr(data\["rr\_intervals\_ms"])

    async def signal\_processor(self):
        """Task 2: Ring Buffer → Features (every 1 second)"""
        while True:
            await asyncio.sleep(1.0)
            window = self.buffer.get\_window(5)
            if window is None:
                continue

            # Signal quality gate
            sqi, sqi\_detail = compute\_sqi(window, sr=130)

            # ECG processing
            ecg\_result = process\_ecg\_window(window, sr=130, sqi\_score=sqi)
            ecg\_result\["sqi"] = sqi
            ecg\_result\["hrv"] = self.hrv\_computer.compute()

            # Update session state
            self.session.update(ecg\_result)

            # Send to feature queue for display
            await self.feature\_queue.put(ecg\_result)

    async def embedding\_and\_retrieval(self):
        """Task 3: CLEF → ChromaDB (every 10 seconds)"""
        while True:
            await asyncio.sleep(10.0)
            window\_10s = self.buffer.get\_window\_for\_clef(10)
            if window\_10s is None:
                continue

            # CLEF embedding (\~3-5ms on GPU)
            embedding = self.clef.encode(window\_10s)

            # ChromaDB retrieval (\~5-15ms)
            retrieval = self.retrieval.query(embedding)

            # Store for next narrative generation
            self.session.latest\_retrieval = retrieval
            self.session.latest\_embedding = embedding

    async def narrative\_generator(self):
        """Task 4: Features → LLM → WebSocket (every 30 seconds)"""
        while True:
            await asyncio.sleep(30.0)
            if not self.session.latest\_retrieval:
                continue

            features = aggregate\_features(
                self.session.latest\_signal,
                self.session.latest\_embedding,
                self.session.latest\_retrieval,
                self.session.get\_context(),
                self.session.patient\_profile
            )

            # RAG: fetch relevant guidelines
            query = f"cardiac rehab {features\['current\_window']\['activity\_phase']} HR {features\['current\_window']\['hr\_bpm']}"
            rag\_context = self.rag.retrieve(query, k=2)

            prompt = build\_llm\_prompt(features, mode="PATIENT\_COACH", rag\_context=rag\_context)

            narrative = ""
            async for token in stream\_narrative(prompt, SYSTEM\_PROMPT):
                narrative += token
                await self.narrative\_queue.put({"type": "token", "text": token})

            await self.narrative\_queue.put({"type": "complete", "text": narrative})

    async def websocket\_server(self):
        """Task 5: Stream everything to Flutter via WebSocket"""
        from fastapi import FastAPI, WebSocket
        import uvicorn

        app = FastAPI()

        @app.websocket("/ws")
        async def ws\_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                # Multiplex: send ECG data, features, and narrative tokens
                try:
                    feature = self.feature\_queue.get\_nowait()
                    await websocket.send\_json({"type": "ecg\_features", "data": feature})
                except asyncio.QueueEmpty:
                    pass

                try:
                    narrative = self.narrative\_queue.get\_nowait()
                    await websocket.send\_json({"type": "narrative", "data": narrative})
                except asyncio.QueueEmpty:
                    pass

                await asyncio.sleep(0.05)  # 20Hz update rate

        config = uvicorn.Config(app, host="0.0.0.0", port=8080)
        server = uvicorn.Server(config)
        await server.serve()

    async def run(self):
        """Launch all concurrent tasks"""
        await asyncio.gather(
            self.ble\_collector(),
            self.rr\_collector(),
            self.signal\_processor(),
            self.embedding\_and\_retrieval(),
            self.narrative\_generator(),
            self.websocket\_server(),
        )

if \_\_name\_\_ == "\_\_main\_\_":
    orchestrator = PipelineOrchestrator()
    asyncio.run(orchestrator.run())
```

\---

## Memory Budget: What Actually Fits

|Component|Memory|When Loaded|
|-|-|-|
|Qwen3-32B-FP4 (via vLLM)|\~18 GB|Hour 0|
|KV cache (4096 ctx, 2 seqs, FP8)|\~8 GB|Dynamic|
|CLEF-Medium (FP16)|60 MB|Hour 4|
|BGE-base-en-v1.5 (text embedding)|440 MB|Hour 8|
|ChromaDB (80K ECG + guidelines)|500 MB|Hour 12|
|NormWear (optional, ECG+ACC)|200 MB|Stretch goal|
|ECG-FM finetuned head (optional)|175 MB|Stretch goal|
|OS + Docker + Python + CUDA|\~10 GB|Always|
|**Total**|**\~28 GB**||
|**Free**|**\~100 GB**||

If the demo is going well and you want to upgrade: swap Qwen3-32B-FP4 for HuatuoGPT-o1-72B-AWQ (\~40 GB) during hours 36-48. You still have 60+ GB free. The decode will be slower (4-8 tok/s vs 15-20) but the medical reasoning quality jumps dramatically. Pre-generate the session summary and SOAP note offline (not streamed live) to hide the latency.

\---

## 48-Hour Build Schedule

|Hours|Owner|Deliverable|
|-|-|-|
|**0-4**|Viggi (DGX)|vLLM running with Qwen3-32B-FP4; verify with test prompt|
|**0-4**|Rumon (HW)|Polar H10 BLE streaming confirmed via polar-python|
|**0-4**|Shiva (AI)|CLEF repo cloned, quickstart notebook runs, embedding verified|
|**0-4**|Sansrit (Flutter)|WebSocket client connecting to FastAPI stub|
|**4-12**|Shiva|Reference bank precomputation: MIT-BIH → CLEF → ChromaDB|
|**4-12**|Viggi|AHA guidelines chunked → BGE embedded → ChromaDB|
|**4-12**|Rumon|Ring buffer + SQI + ECG processing pipeline complete|
|**4-12**|Sansrit|Live ECG trace rendering + HR gauge in Flutter|
|**12-24**|Shiva|Retrieval engine (3 collections) + feature aggregator|
|**12-24**|Rumon|HRV from RR intervals + activity phase detection|
|**12-24**|Viggi|LLM prompt templates + RAG integration + streaming|
|**12-24**|Sansrit|Coach narrative panel + session summary UI|
|**24-36**|ALL|Integration: full pipeline end-to-end test with live H10|
|**24-36**|Shiva|UMAP embedding visualization (the "wow" visual)|
|**24-36**|Viggi|SOAP note generation + PDF report|
|**24-36**|Sansrit|Clinician dashboard view|
|**36-48**|ALL|Polish, demo rehearsal, edge case handling|
|**36-48**|Optional|Swap in HuatuoGPT-o1-72B for better reasoning quality|

\---

## Alert Rules (Deterministic, NOT LLM-Driven)

These fire independently of the LLM and are always active:

|Condition|Threshold|Action|
|-|-|-|
|HR too high|> 85% age-predicted max|Yellow alert + coach warning|
|HR dangerously high|> 100% age-predicted max|Red alert + "Stop and rest"|
|HR too low|< 40 bpm|Red alert|
|Poor HR recovery|HRR₁ < 12 bpm|Flag for clinician review|
|High PVC burden|> 10% of beats PVC-like|Yellow alert|
|Signal loss|SQI < 0.3 for > 30s|"Check electrode contact"|
|Electrode off|HR = 0 for > 5s|"Sensor disconnected"|
|RR irregularity|> 20% variation at rest|Flag for clinician|

These never go through the LLM. They are `if` statements in Python that push directly to the WebSocket.

