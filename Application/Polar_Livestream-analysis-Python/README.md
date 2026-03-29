# Polar ECG Dashboard: Real-Time Biosignal Analytics Platform

## Abstract

The **Polar ECG Dashboard** is a high-performance, multithreaded Python application designed for real-time acquisition, visualization, and physiological analysis of biosignals from the Polar H10 chest strap. Built to serve as a foundation for advanced AI/ML cardiovascular research, it emphasizes zero-copy memory management, low-latency rendering, and robust Bluetooth Low Energy (BLE) stream handling. The platform extracts native 130 Hz Electrocardiogram (ECG), 100 Hz tri-axial Accelerometry (ACC), and 1 Hz Heart Rate (HR) data, performing rolling Heart Rate Variability (HRV) analysis and beat-by-beat ECG delineation.

## System Architecture

The application is structured into four primary decoupled layers, orchestrated via Qt's signal-slot mechanism across multiple threads to ensure UI fluidity during I/O and heavy computation.

### 1. Data Acquisition Layer (`ble_worker.py`)

- **Library Stack**: Uses `bleak` for asynchronous BLE communication and `bleakheart` for parsing Polar Measurement Data (PMD) byte streams.
- **Worker Model**: Runs in an isolated `QThread` utilizing an internal `asyncio` event loop.
- **Robustness Strategy**: The Polar H10's PMD service is notorious for silent failures (accepting `START` commands but never emitting data) on certain OS stacks (e.g., Windows). Our BLE worker implements a hardened state machine:
  1. **HR-First Synchronization**: Waits for the first real HR notification to ensure the BLE link is "warm" and electrodes have skin contact.
  2. **MTU Negotiation**: Requests an ATT MTU of 247 bytes to prevent fragmentation bottlenecks for high-bandwidth ECG frames.
  3. **PMD Priming**: Triggers `available_settings` reads to force the BLE stack to register the PMD control point notifications *before* sending streaming commands.
  4. **Frame Verification & Retries**: Employs a strict timeout (e.g., 8 seconds for ECG). If the first decoded frame isn't received, it issues a `STOP` command, pauses, and retries up to 4 times per modality.

### 2. Signal Processing Layer (`processing_worker.py`)

- **Bandpass Filtering**: 4th-order Butterworth bandpass (0.5–40 Hz) applied to the raw 130 Hz ECG stream before any analysis.
- **Signal Quality Index (SQI)**: Three-metric pipeline evaluated over 5-second windows:
  - **NeuroKit2** template-matching SQI (primary driver of the dashboard quality label)
  - **QRS Band Energy** via Welch PSD (5–15 Hz / 1–40 Hz power ratio)
  - **Vital Kurtosis** from the `vital_sqi` package
- **SQI Thresholds**: Good (≥ 0.6), Fair (0.3–0.6), Poor (≤ 0.3)
- **HRV Analytics (Time & Frequency Domain)**:
  - Extracts R-peaks using `neurokit2` (`nk.ecg_peaks`).
  - Computes time-domain metrics (RMSSD, SDNN, Mean HR) on a rolling 30-second window.
  - Computes frequency-domain metrics (LF/HF ratio) utilizing the **Lomb-Scargle periodogram** (`pyhrv.frequency_domain.lomb_psd`).
- **Morphological Delineation**: Uses Discrete Wavelet Transform (DWT) via `neurokit2` (`method="dwt"`) to dynamically delineate P, QRS, and T wave onsets and offsets. Computes P-width, QRS-width, ST-segment, QT-interval, and QTc (Bazett's corrected).

### 3. UI and Memory Management Layer (`dashboard.py`, `intake_form.py` & `ring_buffer.py`)

- **Patient Intake Form**: A comprehensive 15-question Patient Intake form capturing Demographics, Clinical History, and Risk/Symptoms. The form auto-loads previously saved responses from `intake_state.json` and supports loading from external JSON files.
- **Rendering Engine**: `pyqtgraph` stacked plots with shared X-axes and peak-decimation downsampling (`antialias=False`, `setDownsampling(mode="peak")`).
- **Memory Optimization**: Custom `RingBuffer` backed by contiguous `np.ndarray` memory with slice-based concatenation for near zero-copy reads at 30 FPS.

### 4. Real-Time Telemetry & Export (`mqtt_worker.py` & `data_exporter.py`)

- **JSON Telemetry Exporter**: Automatically initializes a session folder upon pressing "Record". Saves Intake Metadata and appends synchronous 5-second `window_result` dictionaries to a local JSON file.
- **Unified MQTT Streaming**: A background `QThread` publishes to `broker.emqx.io` via `paho-mqtt`. Every 5 seconds it pushes a consolidated JSON payload over `pulseforgeai/{subject_id}/raw` containing the raw 130 Hz ECG array tightly bundled with all computed SQI, HRV, morphology, and accelerometer metrics for that exact window.

## Extracted Physiological Metrics

The pipeline outputs the following metrics updated periodically:

- **Time-Domain HRV**: RMSSD (ms), SDNN (ms)
- **Frequency-Domain HRV**: LF/HF Ratio
- **Basic Vitals**: Mean HR (bpm), VO2max estimation
- **ECG Morphology**: P-Wave Width (ms), QRS Complex Width (ms), ST Segment Duration (ms), QT/QTc Interval (Bazett) (ms)
- **Signal Quality**: SQI (3-Tier: Good/Fair/Poor), QRS Band Energy, NeuroKit2 SQI, Vital Kurtosis
- **Accelerometer HAR**: Mean magnitude (mg), variance, spectral entropy, median frequency (Hz)

## Codebase Structure

```text
polar_ecg/
├── ui/
│   ├── dashboard.py          # PyQt5 main window, layout, and 30FPS plotting timer
│   └── intake_form.py        # Comprehensive multi-tab patient intake dialog
├── workers/
│   ├── ble_worker.py         # Hardened async BLE acquisition (QThread)
│   ├── processing_worker.py  # Bandpass filter, SQI, DWT delineation, Lomb-Scargle HRV
│   └── mqtt_worker.py        # Paho-MQTT v2 unified streaming (QThread)
├── utils/
│   ├── data_exporter.py      # Local JSON session recording
│   ├── ring_buffer.py        # Numpy-backed circular buffer for zero-copy rendering
│   ├── mock_sensor.py        # Synthetic PQRST/ACC generator for offline testing
│   └── constants.py          # Centralized configuration (rates, colors, timeouts)
└── main.py                   # Application entry point
```

## Setup & Execution

### Prerequisites

Tested on Python 3.12+ (Conda environment recommended).

```bash
pip install -r requirements.txt
```

### Running the Application

**Hardware Mode (Requires Polar H10):**

```bash
python main.py
```

*Workflow: Click "Scan" → Select "Polar H10 XXXX" → Click "Connect" → Click "Start Recording" to begin MQTT stream.*

**Simulation Mode (No hardware required):**

```bash
python main.py --mock
```

*Workflow: Click "Mock Sensor" to stream synthetic 130 Hz ECG and 100 Hz ACC data.*

### Monitoring MQTT Output

Subscribe to the live telemetry using any MQTT client (e.g., [MQTTX](https://mqttx.app/)):

```
Broker:  broker.emqx.io
Port:    1883
Topic:   pulseforgeai/#
```

## Future Work & Roadmap

- **ECG Foundation Model (CLEF) Integration**: Route 5-second MQTT ECG windows through the [Nokia Bell Labs CLEF model](https://github.com/Nokia-Bell-Labs/ecg-foundation-model) for zero-shot diagnostic predictions (LVEF, arrhythmia classification, BP estimation).
- **DGX Spark Deployment**: Migrate the MQTT subscriber and multi-agent AI pipeline to on-premise NVIDIA DGX Spark for HIPAA-compliant inference.