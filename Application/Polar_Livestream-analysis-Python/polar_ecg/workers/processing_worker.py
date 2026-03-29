"""
Signal processing worker.
Handles rolling HRV analysis, ECG delineation, and ACC HAR feature extraction.

Two analysis loops run independently:
  • 5-second window  → ECG SQI + instant HR + ACC HAR features  (window_result signal)
  • 30-second window → RMSSD, SDNN, LF/HF, ECG morphology widths (hrv_result signal)

ACC HAR features — top 4 most discriminative for chest-belt HAR:
  1. mean_mag_mg      — signal magnitude   (overall activity intensity)
  2. var_mag_mg2      — variance / energy  (high active, low sedentary)
  3. spectral_entropy — spectral entropy   (low = periodic walk/cycle, high = noise/rest)
  4. median_freq_hz   — median frequency   (walking ~1-2 Hz, cycling higher, sitting ~0)
"""

import matplotlib
matplotlib.use('Agg')  # keep pyhrv from opening windows in the background thread

import time
import os
import numpy as np
from collections import deque

from PyQt5.QtCore import QObject, pyqtSignal

from polar_ecg.utils.har_inference import HARInferenceEngine
from polar_ecg.utils.constants import (
    ECG_NATIVE_HZ,
    ACC_HZ,
    HRV_ANALYSIS_INTERVAL_S,
)

_5S_ECG_SAMPLES  = ECG_NATIVE_HZ * 5    # 650
_5S_ACC_SAMPLES  = ACC_HZ * 5           # 500
_10S_ACC_SAMPLES = ACC_HZ * 10          # 1000
_30S_ECG_SAMPLES = ECG_NATIVE_HZ * 30   # 3 900


# ---------------------------------------------------------------------------
#  ACC HAR feature extraction — top 4 features
# ---------------------------------------------------------------------------

def compute_acc_har_features(acc_xyz: np.ndarray, fs: float = ACC_HZ) -> dict:
    """
    Extract the 4 most discriminative HAR features from a window of 3-axis
    accelerometer data (chest belt, 100 Hz).

    Features
    --------
    mean_mag_mg      — signal magnitude   : overall activity intensity
    var_mag_mg2      — variance           : energy; high active, low sedentary
    spectral_entropy — spectral entropy   : low = periodic (walk/cycle), high = noise/rest
    median_freq_hz   — median frequency   : ~1-2 Hz walking, higher cycling, ~0 sitting

    Parameters
    ----------
    acc_xyz : ndarray shape (N, 3)   — raw acceleration in mg, columns X/Y/Z
    fs      : float                  — sampling frequency in Hz (default 100)
    """
    if acc_xyz is None or acc_xyz.ndim != 2 or acc_xyz.shape[1] < 3 or len(acc_xyz) < 20:
        return {
            "mean_mag_mg":      None,
            "var_mag_mg2":      None,
            "spectral_entropy": None,
            "median_freq_hz":   None,
        }

    x   = acc_xyz[:, 0].astype(np.float64)
    y   = acc_xyz[:, 1].astype(np.float64)
    z   = acc_xyz[:, 2].astype(np.float64)
    mag = np.sqrt(x**2 + y**2 + z**2)
    N   = len(mag)

    def _r(v, d=4):
        return round(float(v), d) if (v is not None and np.isfinite(v)) else None

    # 1 & 2 — time-domain
    mean_mag = float(np.mean(mag))
    var_mag  = float(np.var(mag, ddof=1))

    # 3 & 4 — frequency-domain (FFT with Hann window, restricted to 0–25 Hz)
    mag_ac   = mag - mean_mag                   # remove DC
    windowed = mag_ac * np.hanning(N)
    fft_mag  = np.abs(np.fft.rfft(windowed))
    freqs    = np.fft.rfftfreq(N, d=1.0 / fs)

    cutoff  = np.searchsorted(freqs, 25.0)
    fft_mag = fft_mag[:cutoff]
    freqs   = freqs[:cutoff]
    psd     = fft_mag ** 2
    total_p = float(np.sum(psd))

    spectral_entropy = None
    median_freq      = None

    if total_p > 0 and len(freqs) > 1:
        psd_norm = psd / total_p

        # Spectral entropy: normalised Shannon entropy
        # 0 = single perfect peak (very periodic), 1 = flat/white noise
        eps = 1e-12
        spectral_entropy = float(
            -np.sum(psd_norm * np.log2(psd_norm + eps)) / np.log2(len(psd_norm))
        )

        # Median frequency: lowest f at which cumulative power reaches 50 %
        cum_p   = np.cumsum(psd_norm)
        med_idx = int(np.searchsorted(cum_p, 0.5))
        median_freq = float(freqs[min(med_idx, len(freqs) - 1)])

    return {
        "mean_mag_mg":      _r(mean_mag),
        "var_mag_mg2":      _r(var_mag),
        "spectral_entropy": _r(spectral_entropy),
        "median_freq_hz":   _r(median_freq, 3),
    }


# ---------------------------------------------------------------------------
#  Worker
# ---------------------------------------------------------------------------

class ProcessingWorker(QObject):
    """
    Consumes raw 130 Hz ECG + 100 Hz ACC data and periodically runs analysis.

    Signals
    -------
    hrv_result(dict)
        Emitted every ~30 s — RMSSD, SDNN, LF/HF, ECG morphology.
    window_result(dict)
        Emitted every 5 s — ECG SQI, instant HR, full ACC HAR feature set.
    status(str)
        Human-readable status messages.
    """

    hrv_result    = pyqtSignal(object)
    window_result = pyqtSignal(object)
    status        = pyqtSignal(str)

    def __init__(self, buffer_seconds: int = 120):
        super().__init__()
        self._running = False

        self._ecg_buffer = deque(maxlen=ECG_NATIVE_HZ * buffer_seconds)
        # Separate per-axis ACC deques for efficiency
        # We need at least 10s of ACC for the PyTorch HAR models
        self._acc_x_buf  = deque(maxlen=ACC_HZ * buffer_seconds)
        self._acc_y_buf  = deque(maxlen=ACC_HZ * buffer_seconds)
        self._acc_z_buf  = deque(maxlen=ACC_HZ * buffer_seconds)

        self._last_hrv_time = 0.0
        self._last_5s_time  = 0.0
        self._hrv_enabled   = True

    # ------------------------------------------------------------------ #

    def set_hrv_enabled(self, enabled: bool):
        self._hrv_enabled = enabled

    def add_raw_ecg(self, samples: list):
        self._ecg_buffer.extend(samples)

    def add_raw_acc(self, samples):
        """
        Accept ACC samples as a list/array of [x, y, z] rows (mg units).
        Thread-safe; called from the main thread.
        """
        for s in samples:
            self._acc_x_buf.append(float(s[0]))
            self._acc_y_buf.append(float(s[1]))
            self._acc_z_buf.append(float(s[2]))

    def run(self):
        self._running = True
        self.status.emit("Processing worker started")
        
        # Instantiate HAR inference engine inside the thread
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
            act_dir = os.path.join(repo_root, 'Act_Recoginition')
            self.status.emit("Loading PyTorch HAR Fusion Model...")
            self.har_engine = HARInferenceEngine(act_dir)
            self.status.emit("HAR Models loaded successfully.")
        except Exception as e:
            self.har_engine = None
            self.status.emit(f"Failed to load HAR models: {e}")

        while self._running:
            self._maybe_run_5s_window()
            self._maybe_run_hrv()
            time.sleep(0.02)

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------ #
    #  5-second window: ECG SQI + instant HR + ACC HAR features
    # ------------------------------------------------------------------ #

    def _maybe_run_5s_window(self):
        now = time.time()
        if now - self._last_5s_time < 5.0:
            return
        if len(self._ecg_buffer) < _5S_ECG_SAMPLES:
            return

        self._last_5s_time = now
        ecg_5s = np.array(list(self._ecg_buffer)[-_5S_ECG_SAMPLES:], dtype=np.float64)

        # Build ACC snapshots (5s for features, 10s for ML inference)
        acc_xyz = None
        acc_10s = None
        
        if len(self._acc_x_buf) >= _5S_ACC_SAMPLES:
            x = np.array(list(self._acc_x_buf)[-_5S_ACC_SAMPLES:])
            y = np.array(list(self._acc_y_buf)[-_5S_ACC_SAMPLES:])
            z = np.array(list(self._acc_z_buf)[-_5S_ACC_SAMPLES:])
            acc_xyz = np.column_stack([x, y, z])
            
        if len(self._acc_x_buf) >= _10S_ACC_SAMPLES:
            x10 = np.array(list(self._acc_x_buf)[-_10S_ACC_SAMPLES:])
            y10 = np.array(list(self._acc_y_buf)[-_10S_ACC_SAMPLES:])
            z10 = np.array(list(self._acc_z_buf)[-_10S_ACC_SAMPLES:])
            acc_10s = np.column_stack([x10, y10, z10])

        try:
            result = self._compute_5s_window(ecg_5s, acc_xyz, acc_10s)
            result["timestamp"] = now
            self.window_result.emit(result)
        except Exception as exc:
            self.window_result.emit({
                "timestamp":   now,
                "sqi":         None,
                "sqi_metrics": {},
                "instant_hr":  None,
                "n_r_peaks":   0,
                "acc_features": {},
                "error":       str(exc),
            })

    def _compute_5s_window(self, ecg_5s: np.ndarray, acc_xyz, acc_10s=None) -> dict:
        import neurokit2 as nk
        import scipy.signal

        # ---- Bandpass filter 0.5–40 Hz (4th-order Butterworth) ----
        b, a = scipy.signal.butter(4, [0.5, 40], btype='bandpass', fs=ECG_NATIVE_HZ)
        ecg_filtered = scipy.signal.filtfilt(b, a, ecg_5s)

        # ECG quality & instant HR
        ecg_cleaned = nk.ecg_clean(ecg_filtered, sampling_rate=ECG_NATIVE_HZ)
        _, peak_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=ECG_NATIVE_HZ)
        r_peaks = np.array(peak_info.get("ECG_R_Peaks", []), dtype=int)

        sqi = None
        instant_hr = None
        sqi_metrics = {}

        if len(r_peaks) >= 2:
            # 1. QRS Band Energy (bSQI approach via Welch PSD)
            qrs_energy_sqi = 0.5  # safe fallback
            try:
                f, Pxx = scipy.signal.welch(
                    ecg_cleaned, fs=ECG_NATIVE_HZ,
                    nperseg=min(256, len(ecg_cleaned)),
                )
                qrs_band   = (f >= 5) & (f <= 15)
                total_band = (f >= 1) & (f <= 40)

                qrs_power   = np.sum(Pxx[qrs_band])
                total_power = np.sum(Pxx[total_band])

                if total_power > 0:
                    qrs_energy_sqi = float(qrs_power / total_power)
            except Exception:
                pass

            # 2. vital_sqi Kurtosis (raw value, not normalized)
            vital_kurtosis = None
            try:
                import vital_sqi.sqi.standard_sqi as standard_sqi
                if hasattr(standard_sqi, 'kurtosis_sqi'):
                    k = standard_sqi.kurtosis_sqi(ecg_cleaned)
                    if k is not None:
                        vital_kurtosis = float(k)
            except Exception:
                pass

            # 3. NeuroKit2 SQI (template-matching distance)
            nk_sqi = qrs_energy_sqi  # fallback
            try:
                quality_arr = nk.ecg_quality(
                    ecg_cleaned, rpeaks=r_peaks,
                    sampling_rate=ECG_NATIVE_HZ,
                )
                nk_sqi = float(np.mean(quality_arr))
            except Exception:
                pass

            # Primary SQI → NeuroKit (drives the dashboard quality label)
            sqi = nk_sqi

            sqi_metrics = {
                "nk_sqi":          round(nk_sqi, 4),
                "qrs_energy":      round(qrs_energy_sqi, 4),
                "vital_kurtosis":  round(vital_kurtosis, 2) if vital_kurtosis is not None else None,
            }

            # Instant HR from RR intervals
            rr_ms = np.diff(r_peaks) / ECG_NATIVE_HZ * 1000.0
            valid = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
            if len(valid) > 0:
                instant_hr = float(60_000.0 / np.mean(valid))

        # ACC HAR features
        acc_features = compute_acc_har_features(acc_xyz, fs=float(ACC_HZ))
        
        # PyTorch HAR Inference
        har_activity = {"label": "unknown", "confidence": {}}
        if getattr(self, "har_engine", None) is not None and acc_10s is not None:
            try:
                har_activity = self.har_engine.predict(acc_10s)
            except Exception as e:
                pass

        return {
            "sqi":          round(sqi, 4)        if sqi is not None        else None,
            "sqi_metrics":  sqi_metrics,
            "instant_hr":   round(instant_hr, 1) if instant_hr is not None else None,
            "n_r_peaks":    int(len(r_peaks)),
            "acc_features": acc_features,
            "har_activity": har_activity,
            "raw_ecg":      ecg_cleaned.tolist() if 'ecg_cleaned' in locals() else ecg_5s.tolist(),
        }

    # ------------------------------------------------------------------ #
    #  30-second window: full HRV + ECG morphology
    # ------------------------------------------------------------------ #

    def _maybe_run_hrv(self):
        now = time.time()
        if now - self._last_hrv_time < HRV_ANALYSIS_INTERVAL_S:
            return
        if not self._hrv_enabled:
            return
        if len(self._ecg_buffer) < _30S_ECG_SAMPLES:
            return

        self._last_hrv_time = now
        signal = np.array(list(self._ecg_buffer)[-_30S_ECG_SAMPLES:], dtype=np.float64)

        try:
            result = self._compute_hrv(signal)
            self.hrv_result.emit(result)
        except Exception as exc:
            self.status.emit(f"HRV analysis error: {exc}")

    def _compute_hrv(self, signal: np.ndarray) -> dict:
        import neurokit2 as nk

        ecg_cleaned = nk.ecg_clean(signal, sampling_rate=ECG_NATIVE_HZ)
        _, peak_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=ECG_NATIVE_HZ)
        r_peaks = peak_info.get("ECG_R_Peaks", [])

        if not hasattr(r_peaks, '__len__'):
            r_peaks = list(r_peaks)

        if len(r_peaks) < 4:
            return {
                "rmssd": None, "lf_hf": None,
                "mean_hr": None, "sdnn": None,
                "status": "Insufficient R-peaks detected",
            }

        r_peaks      = np.array(r_peaks)
        rr_intervals = np.diff(r_peaks) / ECG_NATIVE_HZ * 1000.0
        valid        = (rr_intervals > 300) & (rr_intervals < 2000)
        rr_intervals = rr_intervals[valid]

        if len(rr_intervals) < 3:
            return {
                "rmssd": None, "lf_hf": None,
                "mean_hr": None, "sdnn": None,
                "status": "Insufficient valid RR intervals",
            }

        rmssd   = float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)))
        sdnn    = float(np.std(rr_intervals, ddof=1))
        mean_rr = float(np.mean(rr_intervals))
        mean_hr = 60_000.0 / mean_rr if mean_rr > 0 else None

        lf_hf = None
        if len(rr_intervals) >= 10:
            try:
                import pyhrv.frequency_domain as fd
                res = fd.lomb_psd(
                    nni=rr_intervals.tolist(),
                    show=False, show_param=False, legend=False,
                )
                try:
                    v = res["lomb_ratio"]
                    if v is not None and np.isfinite(float(v)):
                        lf_hf = float(v)
                except (KeyError, TypeError, ValueError):
                    pass
            except Exception as exc:
                print(f"pyHRV error: {exc}")

        p_width = qrs_width = st_width = qt_width = qtc_width = None
        try:
            _, waves = nk.ecg_delineate(
                ecg_cleaned, r_peaks,
                sampling_rate=ECG_NATIVE_HZ,
                method="dwt", show=False,
            )

            def _mean_width(on_key, off_key):
                onsets  = waves.get(on_key, [])
                offsets = waves.get(off_key, [])
                if not onsets or not offsets or len(onsets) != len(offsets):
                    return None
                widths = [
                    off - on for on, off in zip(onsets, offsets)
                    if on is not None and off is not None
                    and not np.isnan(on) and not np.isnan(off)
                ]
                return float(np.mean(widths)) / ECG_NATIVE_HZ * 1000 if widths else None

            p_width   = _mean_width("ECG_P_Onsets",  "ECG_P_Offsets")
            qrs_width = _mean_width("ECG_R_Onsets",  "ECG_R_Offsets")
            st_width  = _mean_width("ECG_R_Offsets", "ECG_T_Onsets")
            qt_width  = _mean_width("ECG_R_Onsets",  "ECG_T_Offsets")

            if qt_width is not None and mean_rr > 0:
                qtc_width = qt_width / np.sqrt(mean_rr / 1000.0)
        except Exception:
            pass

        return {
            "rmssd":     round(rmssd, 2)     if rmssd     else None,
            "lf_hf":     round(lf_hf, 3)     if lf_hf is not None else None,
            "mean_hr":   round(mean_hr, 1)   if mean_hr is not None else None,
            "sdnn":      round(sdnn, 2)      if sdnn      else None,
            "p_width":   round(p_width, 1)   if p_width is not None else None,
            "qrs_width": round(qrs_width, 1) if qrs_width is not None else None,
            "st_width":  round(st_width, 1)  if st_width is not None else None,
            "qt_width":  round(qt_width, 1)  if qt_width is not None else None,
            "qtc_width": round(qtc_width, 1) if qtc_width is not None else None,
            "n_peaks":   int(len(r_peaks)),
            "status":    "OK",
        }
