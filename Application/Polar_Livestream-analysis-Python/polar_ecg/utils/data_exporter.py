"""
Per-subject data export: appends 5-second window payloads as JSON Lines.

File layout
-----------
exports/
  <subject_id>/
    <subject_id>_<YYYYMMDD_HHMMSS>.json   ← one file per recording session
                                             (newline-delimited JSON, one record per line)

JSON record structure (one line per 5-second window)
─────────────────────────────────────────────────────
{
  "unix_timestamp": 1743170400.123,
  "subject_id": "S001",
  "window_s": 5,
  "ecg_quality":    { "sqi": 0.923 },
  "heart_rate":     { "avg_bpm_ble": 72.4, "n_ble_samples": 5,
                      "avg_bpm_ecg": 71.8, "n_r_peaks": 6 },
  "hrv":            { "rmssd_ms": 38.5, "sdnn_ms": 29.1,
                      "lf_hf_ratio": 1.24, "analysis_window_s": 30 },
  "ecg_morphology": { "qrs_ms": 88.5, "qt_ms": 390.2, "qtc_ms": 418.1,
                      "st_ms": 150.3, "p_ms": 94.7 },
  "accelerometer": {
    "mean_mag_mg":      overall activity intensity (signal magnitude),
    "var_mag_mg2":      energy — high active, low sedentary,
    "spectral_entropy": 0 = periodic (walk/cycle), 1 = noise/rest,
    "median_freq_hz":   ~1-2 Hz walking, higher cycling, ~0 sitting
  }
}
"""

import json
import time
from pathlib import Path
from typing import Optional

EXPORT_ROOT = Path(__file__).parent.parent.parent / "exports"


class DataExporter:
    """Manages a single recording session for one subject.

    Usage
    -----
    exporter = DataExporter()
    path = exporter.start_session("S001")
    exporter.append_window(payload)   # called every 5 s while recording
    exporter.stop_session()
    """

    def __init__(self):
        self._subject_id:   Optional[str]  = None
        self._session_file: Optional[Path] = None
        self._recording    = False
        self._window_count = 0

    # ------------------------------------------------------------------
    #  Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, subject_id: str) -> Path:
        """Start a new recording session. Returns the path of the new file."""
        subject_id = subject_id.strip()
        if not subject_id:
            raise ValueError("Subject ID must not be empty")

        self._subject_id = subject_id
        folder = EXPORT_ROOT / subject_id
        folder.mkdir(parents=True, exist_ok=True)

        ts_label = time.strftime("%Y%m%d_%H%M%S")
        self._session_file = folder / f"{subject_id}_{ts_label}.json"
        self._recording    = True
        self._window_count = 0
        return self._session_file

    def stop_session(self):
        self._recording = False

    # ------------------------------------------------------------------
    #  Data writing
    # ------------------------------------------------------------------

    def append_window(self, payload: dict) -> None:
        """Append one 5-second window payload as a single JSON line."""
        if not self._recording or self._session_file is None:
            return
        with open(self._session_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, allow_nan=False) + "\n")
        self._window_count += 1

    def build_payload(
        self,
        *,
        subject_id: str,
        unix_timestamp: float,
        window_s: int = 5,
        # ECG quality
        ecg_sqi: Optional[float] = None,
        # Heart rate (BLE notifications averaged over the window)
        avg_hr_bpm: Optional[float] = None,
        n_hr_samples: int = 0,
        # ECG-derived instant HR (from R-peaks in the 5 s ECG chunk)
        avg_hr_ecg_bpm: Optional[float] = None,
        n_r_peaks: int = 0,
        # HRV (from the most recent 30 s analysis)
        rmssd_ms: Optional[float] = None,
        sdnn_ms: Optional[float] = None,
        lf_hf: Optional[float] = None,
        hrv_window_s: int = 30,
        # ECG morphology (from the most recent 30 s analysis)
        qrs_ms: Optional[float] = None,
        qt_ms: Optional[float] = None,
        qtc_ms: Optional[float] = None,
        st_ms: Optional[float] = None,
        p_ms: Optional[float] = None,
        # Accelerometer HAR feature dict (produced by compute_acc_har_features)
        acc_features: Optional[dict] = None,
        # PyTorch unified HAR activity inference
        har_activity: Optional[dict] = None,
    ) -> dict:
        """Build a structured payload dict ready for `append_window()`."""

        def _r(v, digits=2):
            return round(v, digits) if v is not None else None

        payload = {
            "unix_timestamp": round(unix_timestamp, 3),
            "subject_id": subject_id,
            "window_s": window_s,
            "ecg_quality": {
                "sqi": _r(ecg_sqi, 3),
            },
            "heart_rate": {
                "avg_bpm_ble":  _r(avg_hr_bpm, 1),
                "n_ble_samples": n_hr_samples,
                "avg_bpm_ecg":  _r(avg_hr_ecg_bpm, 1),
                "n_r_peaks":    n_r_peaks,
            },
            "hrv": {
                "rmssd_ms":          _r(rmssd_ms, 2),
                "sdnn_ms":           _r(sdnn_ms, 2),
                "lf_hf_ratio":       _r(lf_hf, 3),
                "analysis_window_s": hrv_window_s,
            },
            "ecg_morphology": {
                "qrs_ms": _r(qrs_ms, 1),
                "qt_ms":  _r(qt_ms, 1),
                "qtc_ms": _r(qtc_ms, 1),
                "st_ms":  _r(st_ms, 1),
                "p_ms":   _r(p_ms, 1),
            },
            "accelerometer": {
                "features": acc_features or {},
                "activity": har_activity or {"label": "unknown", "confidence": {}}
            }
        }

        return payload

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def window_count(self) -> int:
        return self._window_count

    @property
    def subject_id(self) -> Optional[str]:
        return self._subject_id

    @property
    def session_file(self) -> Optional[Path]:
        return self._session_file
