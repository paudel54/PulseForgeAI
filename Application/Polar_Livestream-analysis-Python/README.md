# Polar H10 BLE Signal Processing Application

**PyQt5 desktop application for real-time cardiac rehabilitation monitoring using the Polar H10 chest strap.**

This is the signal acquisition and processing frontend for the Talk to Your Heart platform. It connects to a Polar H10 via BLE, processes ECG and accelerometer streams through a dual-window analysis pipeline, publishes structured physiologic features over MQTT, and displays real-time clinical metrics.

## Features

- **Polar H10 BLE Acquisition** — ECG at 130 Hz, ACC at 100 Hz, HR + RR intervals
- **Dual-Window Signal Processing** — 5-second and 30-second concurrent analysis windows
- **Mock Sensor Mode** — synthetic ECG/ACC generation for testing without hardware (`--mock`)
- **MQTT Publishing** — structured vitals JSON to per-patient MQTT topics
- **Google Fit Integration** — 7-day longitudinal baseline (HR, steps, sleep, body temp)
- **Clinical Intake Form** — patient demographics, cardiac history, comorbidities, PHQ-2
- **Real-Time Dashboard** — ECG waveform, HRV metrics, activity phase, SQI display

## Signal Processing Pipeline

### 5-Second Window
- **SQI** — Signal Quality Index (0.0–1.0): template matching (0.4) + SNR (0.3) + motion correlation (0.3)
- **Instantaneous HR** — from Pan-Tompkins + Hamilton consensus QRS detection
- **HAR Features** — mean magnitude, variance, spectral entropy, median frequency

### 30-Second Window
- **Time-domain HRV** — RMSSD, SDNN, pNN50
- **Frequency-domain HRV** — LF/HF ratio via Lomb-Scargle periodogram
- **ECG Morphology** — DWT delineation: P-width, QRS-width, QT/QTc, ST-width

## Quickstart

### Prerequisites
- Python 3.10+
- Polar H10 chest strap (optional — use `--mock` for testing)
- MQTT broker (optional — EMQX cloud or local Mosquitto)

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
# With real Polar H10 hardware
python main.py

# Mock sensor mode (no hardware needed)
python main.py --mock
```

## MQTT Payload Schema

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

## Directory Structure

```
├── main.py                    ← Entry point (real/mock sensor selection)
├── intake_state.json          ← Combined clinical + Google Fit data schema
├── polar_ecg/
│   ├── workers/
│   │   ├── processing_worker.py  ← Dual-window signal processing (451 lines)
│   │   ├── ble_worker.py      ← Polar H10 BLE connection management
│   │   └── mqtt_worker.py     ← QThread MQTT publisher (paho-mqtt v2)
│   ├── ui/
│   │   ├── dashboard.py       ← Real-time visualization dashboard
│   │   └── intake_form.py     ← Clinical intake form
│   └── utils/
│       ├── google_fit_fetcher.py  ← Google Fit REST API integration
│       ├── har_inference.py   ← HAR fusion model inference
│       ├── mock_sensor.py     ← Synthetic ECG/ACC/HR generator
│       ├── data_exporter.py   ← Session data export
│       └── ring_buffer.py     ← Efficient circular buffer
├── requirements.txt
└── test_*.py                  ← BLE, MQTT, Google Fit test scripts
```

---
*Part of the Talk to Your Heart platform. See the [main README](../../Readme.md) for the full system architecture.*