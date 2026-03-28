"""
Signal processing worker.
Handles rolling HRV analysis and beat-by-beat ECG delineation directly at 130Hz.
"""

import matplotlib
matplotlib.use('Agg')  # Prevent pyhrv from opening plot windows in background thread

import time
import numpy as np
from collections import deque

from PyQt5.QtCore import QObject, pyqtSignal

from polar_ecg.utils.constants import (
    ECG_NATIVE_HZ,
    HRV_ANALYSIS_INTERVAL_S,
)


class ProcessingWorker(QObject):
    """
    Consumes raw 130Hz ECG data and periodically runs NeuroKit2 HRV analysis.
    """

    hrv_result = pyqtSignal(object)      # dict with RMSSD, LF/HF, QT, etc.
    status = pyqtSignal(str)

    def __init__(self, buffer_seconds: int = 120):
        super().__init__()
        self._running = False
        self._ecg_buffer = deque(maxlen=ECG_NATIVE_HZ * buffer_seconds)
        self._last_hrv_time = 0.0
        self._hrv_enabled = True

    def set_hrv_enabled(self, enabled: bool):
        self._hrv_enabled = enabled

    def add_raw_ecg(self, samples: list):
        """Called from the main thread when new ECG data arrives."""
        self._ecg_buffer.extend(samples)

    def run(self):
        """Main processing loop. Called when the thread starts."""
        self._running = True
        self.status.emit("Processing worker started")

        while self._running:
            self._maybe_run_hrv()
            time.sleep(0.02)  # ~50Hz processing rate

    def stop(self):
        self._running = False

    def _maybe_run_hrv(self):
        """Run NeuroKit2 HRV analysis if enough time has passed."""
        now = time.time()
        if now - self._last_hrv_time < HRV_ANALYSIS_INTERVAL_S:
            return
        if not self._hrv_enabled:
            return

        # 30s window for stable LF/HF
        window_samples = ECG_NATIVE_HZ * 30
        if len(self._ecg_buffer) < window_samples:
            return

        self._last_hrv_time = now

        signal = np.array(
            list(self._ecg_buffer)[-window_samples:], dtype=np.float64
        )

        try:
            result = self._compute_hrv(signal)
            self.hrv_result.emit(result)
        except Exception as e:
            self.status.emit(f"HRV analysis error: {e}")

    def _compute_hrv(self, signal: np.ndarray) -> dict:
        """
        Run NeuroKit2 ECG processing and extract HRV metrics + widths.
        Input: 130Hz raw ECG signal (30s window).
        """
        import neurokit2 as nk
        import pandas as pd

        try:
            ecg_cleaned = nk.ecg_clean(signal, sampling_rate=ECG_NATIVE_HZ)
            _, peak_info = nk.ecg_peaks(
                ecg_cleaned, sampling_rate=ECG_NATIVE_HZ
            )
            r_peaks = peak_info.get("ECG_R_Peaks", [])

            if not hasattr(r_peaks, '__len__'):
                r_peaks = list(r_peaks)

            if len(r_peaks) < 4:
                return {
                    "rmssd": None,
                    "lf_hf": None,
                    "mean_hr": None,
                    "sdnn": None,
                    "status": "Insufficient R-peaks detected",
                }

            r_peaks = np.array(r_peaks)
            rr_intervals = np.diff(r_peaks) / ECG_NATIVE_HZ * 1000  # ms

            # Filter physiologically implausible RR intervals (< 300ms or > 2000ms)
            valid = (rr_intervals > 300) & (rr_intervals < 2000)
            rr_intervals = rr_intervals[valid]

            if len(rr_intervals) < 3:
                return {
                    "rmssd": None,
                    "lf_hf": None,
                    "mean_hr": None,
                    "sdnn": None,
                    "status": "Insufficient valid RR intervals",
                }

            # Time-domain: RMSSD
            rr_diffs = np.diff(rr_intervals)
            rmssd = float(np.sqrt(np.mean(rr_diffs ** 2)))

            # Time-domain: SDNN
            sdnn = float(np.std(rr_intervals, ddof=1))

            # Mean HR
            mean_rr = np.mean(rr_intervals)
            mean_hr = 60000.0 / mean_rr if mean_rr > 0 else None

            # Frequency-domain: LF/HF ratio via pyHRV Lomb-Scargle
            lf_hf = None
            if len(rr_intervals) >= 10:
                try:
                    import pyhrv.frequency_domain as fd
                    res = fd.lomb_psd(
                        nni=rr_intervals.tolist(),
                        show=False,
                        show_param=False,
                        legend=False
                    )
                    # biosppy ReturnTuple: use subscript, not `in` (may not implement __contains__ for names)
                    try:
                        v = res["lomb_ratio"]
                        if v is not None and np.isfinite(float(v)):
                            lf_hf = float(v)
                    except (KeyError, TypeError, ValueError):
                        pass
                except Exception as e:
                    print(f"pyHRV error: {e}")
                    pass

            # Beat-by-beat delineation for segment widths
            p_width = None
            qrs_width = None
            st_width = None
            qt_width = None
            qtc_width = None
            
            try:
                # Use DWT method for robust delineation
                _, waves = nk.ecg_delineate(
                    ecg_cleaned, 
                    r_peaks, 
                    sampling_rate=ECG_NATIVE_HZ, 
                    method="dwt", 
                    show=False
                )

                def get_mean_width(onset_key, offset_key):
                    onsets = waves.get(onset_key, [])
                    offsets = waves.get(offset_key, [])
                    if not onsets or not offsets or len(onsets) != len(offsets):
                        return None
                    widths = []
                    for on, off in zip(onsets, offsets):
                        if (on is not None and off is not None 
                                and not np.isnan(on) and not np.isnan(off)):
                            widths.append(off - on)
                    if widths:
                        return float(np.mean(widths)) / ECG_NATIVE_HZ * 1000
                    return None

                p_width = get_mean_width("ECG_P_Onsets", "ECG_P_Offsets")
                qrs_width = get_mean_width("ECG_R_Onsets", "ECG_R_Offsets")
                st_width = get_mean_width("ECG_R_Offsets", "ECG_T_Onsets")
                qt_width = get_mean_width("ECG_R_Onsets", "ECG_T_Offsets")

                if qt_width is not None and mean_rr > 0:
                    # Bazett's formula: QTc = QT / sqrt(RR in seconds)
                    qtc_width = qt_width / np.sqrt(mean_rr / 1000.0)

            except Exception:
                pass

            return {
                "rmssd": round(rmssd, 2) if rmssd else None,
                "lf_hf": round(lf_hf, 3) if lf_hf is not None else None,
                "mean_hr": round(mean_hr, 1) if mean_hr is not None else None,
                "sdnn": round(sdnn, 2) if sdnn else None,
                "p_width": round(p_width, 1) if p_width is not None else None,
                "qrs_width": round(qrs_width, 1) if qrs_width is not None else None,
                "st_width": round(st_width, 1) if st_width is not None else None,
                "qt_width": round(qt_width, 1) if qt_width is not None else None,
                "qtc_width": round(qtc_width, 1) if qtc_width is not None else None,
                "n_peaks": len(r_peaks),
                "status": "OK",
            }

        except Exception as e:
            return {
                "rmssd": None,
                "lf_hf": None,
                "mean_hr": None,
                "sdnn": None,
                "p_width": None,
                "qrs_width": None,
                "st_width": None,
                "qt_width": None,
                "qtc_width": None,
                "status": f"Analysis failed: {e}",
            }
