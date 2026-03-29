# Polar ECG Dashboard: Real-Time Biosignal Analytics Platform

## Abstract

The **Polar ECG Dashboard** is a high-performance, multithreaded Python application designed for real-time acquisition, visualization, and physiological analysis of biosignals from the Polar H10 chest strap. Built to serve as a foundation for advanced AI/ML cardiovascular research, it emphasizes zero-copy memory management, low-latency rendering, and robust Bluetooth Low Energy (BLE) stream handling. The platform extracts native 130 Hz Electrocardiogram (ECG), 100 Hz tri-axial Accelerometry (ACC), and 1 Hz Heart Rate (HR) data, performing rolling Heart Rate Variability (HRV) analysis and beat-by-beat ECG delineation.

## System Architecture

The application is structured into three primary decoupled layers, orchestrated via Qt's signal-slot mechanism across multiple threads to ensure UI fluidity during I/O and heavy computation.

### 1. Data Acquisition Layer (`ble_worker.py`)
- **Library Stack**: Uses `bleak` for asynchronous BLE communication and `bleakheart` for parsing Polar Measurement Data (PMD) byte streams.
- **Worker Model**: Runs in an isolated `QThread` utilizing an internal `asyncio` event loop.
- **Robustness Strategy**: The Polar H10's PMD service is notorious for silent failures (accepting `START` commands but never emitting data) on certain OS stacks (e.g., Windows). Our BLE worker implements a hardened state machine:
  1. **HR-First Synchronization**: Waits for the first real HR notification to ensure the BLE link is "warm" and electrodes have skin contact.
  2. **MTU Negotiation**: Requests an ATT MTU of 247 bytes to prevent fragmentation bottlenecks for high-bandwidth ECG frames.
  3. **PMD Priming**: Triggers `available_settings` reads to force the BLE stack to register the PMD control point notifications *before* sending streaming commands.
  4. **Frame Verification & Retries**: Employs a strict timeout (e.g., 8 seconds for ECG). If the first decoded frame isn't received, it issues a `STOP` command, pauses, and retries up to 4 times per modality.

### 2. Signal Processing Layer (`processing_worker.py`)
- **Native Frequency Processing**: To prevent filter edge artifacts and maintain raw signal fidelity, polyphase resampling was intentionally avoided. The pipeline operates strictly on the native 130 Hz ECG stream.
- **Signal Quality**: Computes the ECG Signal Quality Index (SQI) over 5-second windows using standard statistical metrics derived from the `vital_sqi` package.
- **HRV Analytics (Time & Frequency Domain)**: 
  - Extracts R-peaks using `neurokit2` (`nk.ecg_peaks`).
  - Computes time-domain metrics (RMSSD, SDNN, Mean HR) on a rolling 30-second window.
  - Computes frequency-domain metrics (LF/HF ratio) utilizing the **Lomb-Scargle periodogram** (`pyhrv.frequency_domain.lomb_psd`). Lomb-Scargle is specifically chosen over Welch's method as it natively handles the unevenly sampled nature of RR intervals without requiring cubic spline interpolation, making it highly robust for short (30s) rolling windows.
- **Morphological Delineation**: Uses Discrete Wavelet Transform (DWT) via `neurokit2` (`method="dwt"`) to dynamically delineate P, QRS, and T wave onsets and offsets. Computes mean spatial features: P-width, QRS-width, ST-segment, QT-interval, and QTc (Bazett's corrected).

### 3. UI and Memory Management Layer (`dashboard.py`, `intake_form.py` & `ring_buffer.py`)
- **Patient Intake Form**: A comprehensive 15-question Patient Intake form capturing Demographics, Clinical History, and Risk/Symptoms. The form auto-loads previously saved responses from `intake_state.json` and must be completed prior to opening the Dashboard view or starting a recording session.
- **Rendering Engine**: `pyqtgraph` stacked plots with shared X-axes and peak-decimation downsampling (`antialias=False`, `setDownsampling(mode="peak")`).
- **Memory Optimization**: Python's `collections.deque` incurs massive overhead when converted to `numpy` arrays at 30 FPS. We implemented a custom `RingBuffer` backed by contiguous `np.ndarray` memory. The `get_last_n()` method uses slice-based concatenation, achieving near zero-copy reads for the rendering loop.

## Extracted Physiological Metrics

The pipeline outputs the following metrics updated periodically:
* **Time-Domain HRV**: RMSSD (ms), SDNN (ms)
* **Frequency-Domain HRV**: LF/HF Ratio
* **Basic Vitals**: Mean HR (bpm)
* **ECG Morphology**: P-Wave Width (ms), QRS Complex Width (ms), ST Segment Duration (ms), QT Interval (ms), QTc Interval (Bazett) (ms)

## Codebase Structure

```text
polar_ecg/
├── ui/
│   ├── dashboard.py          # PyQt5 main window, layout, and 30FPS plotting timer
│   └── intake_form.py        # Comprehensive multi-tab patient intake QDialog
├── workers/
│   ├── ble_worker.py         # Hardened async BLE acquisition (QThread)
│   └── processing_worker.py  # DWT delineation and Lomb-Scargle HRV (QThread)
├── utils/
│   ├── constants.py          # Centralized configuration (rates, colors, timeouts)
│   ├── mock_sensor.py        # Synthetic PQRST/ACC generator for offline testing
│   └── ring_buffer.py        # Numpy-backed circular buffer for zero-copy rendering
└── main.py                   # Application entry point
```

## Setup & Execution

### Prerequisites
Tested on Python 3.12+ (Conda environment recommended).

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**Hardware Mode (Requires Polar H10):**
```bash
python main.py
```
*Workflow: Click "Scan" -> Select "Polar H10 XXXX" from dropdown -> Click "Connect".*

**Simulation Mode (No hardware required):**
```bash
python main.py --mock
```
*Workflow: Click "Mock Sensor" to stream synthetic 130Hz ECG and 100Hz ACC data.*

## Future Work & Roadmap

- **ECG Foundation Model (CLEF) Integration**: The architecture is designed to support a secondary inference thread. Future iterations will route 5-second ECG windows through the [Nokia Bell Labs CLEF model](https://github.com/Nokia-Bell-Labs/ecg-foundation-model) for zero-shot diagnostic predictions including Left Ventricular Ejection Fraction (LVEF), Arrhythmia classification, and Blood Pressure (SBP/DBP) estimation.
- **Session Recording**: Implementing an HDF5 or Parquet writer thread to save raw 130Hz/100Hz streams alongside synchronized metric outputs for retrospective dataset curation.