"""
Neural-network VO2max predictor using the pretrained CardioFitness model.

Loads the Keras model + sklearn StandardScaler + PCA transform published by
Spathis et al. (2022) in Nature Digital Medicine (github.com/sdimi/cardiofitness).

Feature extraction maps Polar H10 live streams to the 68-variable Fenland
feature space, then applies the same scaler → PCA → NN pipeline used during
training to output a VO2max estimate (ml/kg/min).

Usage
-----
    nn = CardioFitnessNN()
    nn.update_profile(age=29, sex="male", height_m=1.80, weight_kg=78)
    nn.add_hr(72.0)                 # call once per HR notification
    nn.add_acc_mag_epoch(250.0)     # call once per second (avg ACC vector mag)
    nn.add_hrv_rmssd(42.3)          # call after each HRV analysis window
    result = nn.predict()           # dict: vo2max, status, n_samples

Notes
-----
The CardioFitness model was trained on 3.5-day free-living recordings from the
Fenland Study (N ≈ 11 000).  Predictions improve as more session data
accumulates; allow at least 2 minutes before the first estimate.
"""

import math
import time
from collections import deque
from pathlib import Path

import numpy as np

_REPO_DIR   = Path(__file__).parent.parent.parent / "cardiofitness-main"
_MODEL_DIR  = _REPO_DIR / "models" / "20201109-013142"
_DATA_DIR   = _REPO_DIR / "data"

# Minimum HR samples (seconds) before the first prediction is attempted.
MIN_HR_SAMPLES = 120  # 2 minutes

# The 68 raw feature columns, in alphabetical order matching the scaler.
# Derived by filtering extracted_features.csv columns:
#   remove count_*, real_*, *_MET_*, ending *daily_count, id
FEATURE_COLUMNS = [
    "25%_ACC",          "25%_ENMO",          "25%_MVPA_noncal",
    "25%_Sed_noncal",   "25%_VPA_noncal",    "25%_hrv_milliseconds",
    "25%_mean_hr",
    "50%_ACC",          "50%_ENMO",          "50%_MVPA_noncal",
    "50%_Sed_noncal",   "50%_VPA_noncal",    "50%_hrv_milliseconds",
    "50%_mean_hr",
    "75%_ACC",          "75%_ENMO",          "75%_MVPA_noncal",
    "75%_Sed_noncal",   "75%_VPA_noncal",    "75%_hrv_milliseconds",
    "75%_mean_hr",
    "RHR",              "age",               "bmi",              "height",
    "max_ACC",          "max_ENMO",          "max_MVPA_noncal",
    "max_Sed_noncal",   "max_VPA_noncal",    "max_hrv_milliseconds",
    "max_mean_hr",
    "mean_ACC",         "mean_ENMO",         "mean_MVPA_noncal",
    "mean_Sed_noncal",  "mean_VPA_noncal",   "mean_hrv_milliseconds",
    "mean_mean_hr",
    "min_ACC",          "min_ENMO",          "min_MVPA_noncal",
    "min_Sed_noncal",   "min_VPA_noncal",    "min_hrv_milliseconds",
    "min_mean_hr",
    "month",            "month_cos_time",    "month_sin_time",
    "mvpa_daily_count_noncal", "sed_daily_count_noncal",
    "sex",
    "slope_ACC",        "slope_ENMO",        "slope_MVPA_noncal",
    "slope_Sed_noncal", "slope_VPA_noncal",  "slope_hrv_milliseconds",
    "slope_mean_hr",
    "std_ACC",          "std_ENMO",          "std_MVPA_noncal",
    "std_Sed_noncal",   "std_VPA_noncal",    "std_hrv_milliseconds",
    "std_mean_hr",
    "vpa_daily_count_noncal", "weight",
]  # 68 total


def _slope(arr: np.ndarray) -> float:
    """Least-squares slope of the sequence (same as np.polyfit(..., 1)[0])."""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    xm, ym = x.mean(), arr.mean()
    denom = ((x - xm) ** 2).sum()
    return float(((x - xm) * (arr - ym)).sum() / denom) if denom else 0.0


def _stats(arr: np.ndarray) -> dict:
    """Return the 8 statistical descriptors used by CardioFitness."""
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min":  float(np.min(arr)),
        "25%":  float(np.percentile(arr, 25)),
        "50%":  float(np.percentile(arr, 50)),
        "75%":  float(np.percentile(arr, 75)),
        "max":  float(np.max(arr)),
        "slope": _slope(arr),
    }


class CardioFitnessNN:
    """
    Wraps the CardioFitness pretrained Keras model.

    Feature engineering
    -------------------
    Polar H10 → Fenland feature mapping:

    mean_hr       : BLE HR notifications (BPM)
    ACC           : sqrt(x²+y²+z²) per 100-Hz ACC frame, averaged to 1 Hz (mg)
    ENMO          : (ACC / 0.0060321) + 0.057   (standard Actiheart formula)
    Sed_noncal    : ACC epoch value when ACC < 1 mg, else 0
    MVPA_noncal   : ACC epoch value when ACC ≥ 1 mg, else 0
    VPA_noncal    : ACC epoch value when ACC ≥ 4.15 mg, else 0
    hrv_milliseconds : RMSSD (ms) – proxy for max_IBI − min_IBI used in Fenland
    """

    # Rolling buffers — 30 minutes max
    _MAX_HR_SAMPLES  = 1800   # 1 Hz × 30 min
    _MAX_ACC_SAMPLES = 1800   # 1 Hz × 30 min
    _MAX_HRV_SAMPLES = 900    # one per 2-s analysis window × 30 min

    def __init__(self):
        self._model    = None
        self._scaler   = None
        self._pca      = None
        self._loaded   = False
        self._load_err: str = ""

        # User profile
        self.age        = 30
        self.sex_code   = 1          # 0 = female, 1 = male (Fenland convention)
        self.height_m   = 1.75
        self.weight_kg  = 70.0

        # Session rolling buffers
        self._hr_buf:  deque = deque(maxlen=self._MAX_HR_SAMPLES)
        self._acc_buf: deque = deque(maxlen=self._MAX_ACC_SAMPLES)
        self._hrv_buf: deque = deque(maxlen=self._MAX_HRV_SAMPLES)

        self._session_start = time.time()

    # ------------------------------------------------------------------
    #  Public API — profile + data ingestion
    # ------------------------------------------------------------------

    def update_profile(self, age: int, sex: str, height_m: float,
                       weight_kg: float) -> None:
        self.age        = int(age)
        self.sex_code   = 0 if sex == "female" else 1
        self.height_m   = float(height_m)
        self.weight_kg  = float(weight_kg)

    def add_hr(self, hr_bpm: float) -> None:
        if 30 < hr_bpm < 220:
            self._hr_buf.append(float(hr_bpm))

    def add_acc_mag_epoch(self, acc_mag_mg: float) -> None:
        """Add one 1-second-average ACC vector magnitude (mg)."""
        if acc_mag_mg >= 0:
            self._acc_buf.append(float(acc_mag_mg))

    def add_hrv_rmssd(self, rmssd_ms: float) -> None:
        if rmssd_ms is not None and rmssd_ms > 0:
            self._hrv_buf.append(float(rmssd_ms))

    # ------------------------------------------------------------------
    #  Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def model_files_present() -> bool:
        return (
            (_MODEL_DIR / "weights-regression-improvement-8.89.hdf5").exists()
            and (_DATA_DIR / "scaler_FI.save").exists()
            and (_DATA_DIR / "PCA_FI_mapping_09999.save").exists()
        )

    def load_model(self) -> bool:
        """Load Keras model + scaler + PCA from cardiofitness-main/.  Returns True on success."""
        if self._loaded:
            return True
        if not self.model_files_present():
            self._load_err = "Binary model files missing — run setup_cardiofitness.py"
            return False
        try:
            # ── sklearn.externals.joblib shim ──────────────────────────────
            # The .save files were pickled with sklearn < 0.23 which exposed
            # joblib under sklearn.externals.joblib.  Newer sklearn removed that
            # path, so we register a compatibility alias before unpickling.
            import sys
            import types
            import joblib as _jl
            if "sklearn.externals" not in sys.modules:
                _ext = types.ModuleType("sklearn.externals")
                sys.modules["sklearn.externals"] = _ext
            sys.modules["sklearn.externals"].joblib = _jl
            sys.modules.setdefault("sklearn.externals.joblib", _jl)

            self._scaler = _jl.load(_DATA_DIR / "scaler_FI.save")
            self._pca    = _jl.load(_DATA_DIR / "PCA_FI_mapping_09999.save")

            # ── Keras model ────────────────────────────────────────────────
            # model_from_json fails with KeyError on Keras-2.x functional-API
            # JSONs when run under TF2 (integer node-index lookup mismatch).
            # Instead we rebuild the architecture manually with the exact layer
            # names from model_architecture.json, then load weights by_name=True
            # so h5py matches them correctly.
            import os
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")

            self._model = self._build_model()
            self._model.load_weights(
                str(_MODEL_DIR / "weights-regression-improvement-8.89.hdf5"),
                by_name=True,
            )
            self._model.compile(loss="mse", optimizer="adam")
            self._loaded   = True
            self._load_err = ""
            return True
        except Exception as exc:
            import traceback
            traceback.print_exc()   # prints full stack trace to terminal
            self._load_err = f"{type(exc).__name__}: {exc}"
            return False

    # ------------------------------------------------------------------
    #  Model architecture
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model():
        """Rebuild the CardioFitness NN with exact layer names from model_architecture.json.

        Layer names must match those stored in the HDF5 weight file so that
        load_weights(by_name=True) can pair them correctly.
        """
        import tensorflow as tf
        layers = tf.keras.layers

        inp = tf.keras.Input(shape=(46,), name="input_10")
        x   = layers.Dense(128, activation="elu",
                           name="dense_45")(inp)
        x   = layers.BatchNormalization(momentum=0.99, epsilon=0.001,
                                        name="batch_normalization_36")(x)
        x   = layers.Dropout(0.3, name="dropout_19")(x)
        x   = layers.Dense(128, activation="elu",
                           name="dense_46")(x)
        x   = layers.BatchNormalization(momentum=0.99, epsilon=0.001,
                                        name="batch_normalization_37")(x)
        x   = layers.Dropout(0.3, name="dropout_20")(x)
        out = layers.Dense(1, activation="linear",
                           name="dense_47")(x)
        return tf.keras.Model(inputs=inp, outputs=out, name="model_10")

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------

    def predict(self) -> dict:
        """
        Returns
        -------
        dict with keys:
          vo2max        float | None
          status        str
          n_hr          int   — HR samples in buffer
          n_acc         int
          n_hrv         int
          warmup_left_s float — seconds until MIN_HR_SAMPLES reached (≤0 when ready)
        """
        n_hr  = len(self._hr_buf)
        n_acc = len(self._acc_buf)
        n_hrv = len(self._hrv_buf)
        warmup_left = max(0.0, MIN_HR_SAMPLES - n_hr)

        base = {
            "vo2max":        None,
            "status":        "",
            "n_hr":          n_hr,
            "n_acc":         n_acc,
            "n_hrv":         n_hrv,
            "warmup_left_s": warmup_left,
        }

        if not self.model_files_present():
            base["status"] = "Run setup_cardiofitness.py first"
            return base

        if not self._loaded:
            ok = self.load_model()
            if not ok:
                base["status"] = f"Load error: {self._load_err}"
                return base

        if warmup_left > 0:
            m, s = divmod(int(warmup_left), 60)
            base["status"] = f"Warming up… {m}m {s:02d}s"
            return base

        try:
            feat_vec = self._extract_features()
            if feat_vec is None:
                base["status"] = "Insufficient data"
                return base

            scaled  = self._scaler.transform(feat_vec.reshape(1, -1))
            pca_vec = self._pca.transform(scaled)
            vo2max  = float(self._model.predict(pca_vec, verbose=0)[0][0])
            vo2max  = round(max(vo2max, 10.0), 1)
            base["vo2max"] = vo2max
            base["status"] = f"OK ({n_hr}s data)"
            return base

        except Exception as exc:
            base["status"] = f"Predict error: {exc}"
            return base

    # ------------------------------------------------------------------
    #  Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self) -> "np.ndarray | None":
        """Build the 68-element raw feature vector in the column order the scaler expects."""
        hr_arr  = np.array(self._hr_buf,  dtype=np.float64)
        acc_arr = np.array(self._acc_buf, dtype=np.float64) if self._acc_buf else np.array([0.0])
        hrv_arr = np.array(self._hrv_buf, dtype=np.float64) if self._hrv_buf else np.array([0.0])

        if len(hr_arr) < 2:
            return None

        # Derived ACC channels
        enmo_arr    = (acc_arr / 0.0060321) + 0.057
        sed_arr     = np.where(acc_arr < 1.0,   acc_arr, 0.0)
        mvpa_arr    = np.where(acc_arr >= 1.0,  acc_arr, 0.0)
        vpa_arr     = np.where(acc_arr >= 4.15, acc_arr, 0.0)

        hr_s   = _stats(hr_arr)
        acc_s  = _stats(acc_arr)
        enmo_s = _stats(enmo_arr)
        hrv_s  = _stats(hrv_arr)
        sed_s  = _stats(sed_arr)
        mvpa_s = _stats(mvpa_arr)
        vpa_s  = _stats(vpa_arr)

        # Demographics
        bmi     = self.weight_kg / (self.height_m ** 2) if self.height_m > 0 else 25.0
        rhr     = float(np.percentile(hr_arr, 10))
        now     = time.localtime()
        month   = now.tm_mon
        m_sin   = math.sin(2 * math.pi * month / 12)
        m_cos   = math.cos(2 * math.pi * month / 12)

        # Daily count proxies: fraction of ACC epochs in each category
        # (per-minute rate scaled to per-day; rough proxy for the Fenland daily_count columns)
        total_acc  = max(len(acc_arr), 1)
        sed_count  = float(np.sum(acc_arr < 1.0)  / total_acc * 1440)
        mvpa_count = float(np.sum((acc_arr >= 1.0)  & (acc_arr < 4.15)) / total_acc * 1440)
        vpa_count  = float(np.sum(acc_arr >= 4.15) / total_acc * 1440)

        def _v(stats_dict, stat):
            return stats_dict[stat]

        # Build vector matching FEATURE_COLUMNS order (alphabetical)
        vec = np.array([
            # 25% percentiles
            _v(acc_s,  "25%"), _v(enmo_s, "25%"), _v(mvpa_s, "25%"),
            _v(sed_s,  "25%"), _v(vpa_s,  "25%"), _v(hrv_s,  "25%"), _v(hr_s, "25%"),
            # 50%
            _v(acc_s,  "50%"), _v(enmo_s, "50%"), _v(mvpa_s, "50%"),
            _v(sed_s,  "50%"), _v(vpa_s,  "50%"), _v(hrv_s,  "50%"), _v(hr_s, "50%"),
            # 75%
            _v(acc_s,  "75%"), _v(enmo_s, "75%"), _v(mvpa_s, "75%"),
            _v(sed_s,  "75%"), _v(vpa_s,  "75%"), _v(hrv_s,  "75%"), _v(hr_s, "75%"),
            # demographics (sorted alphabetically)
            rhr, float(self.age), bmi, self.height_m,
            # max
            _v(acc_s,  "max"), _v(enmo_s, "max"), _v(mvpa_s, "max"),
            _v(sed_s,  "max"), _v(vpa_s,  "max"), _v(hrv_s,  "max"), _v(hr_s, "max"),
            # mean
            _v(acc_s,  "mean"), _v(enmo_s, "mean"), _v(mvpa_s, "mean"),
            _v(sed_s,  "mean"), _v(vpa_s,  "mean"), _v(hrv_s,  "mean"), _v(hr_s, "mean"),
            # min
            _v(acc_s,  "min"), _v(enmo_s, "min"), _v(mvpa_s, "min"),
            _v(sed_s,  "min"), _v(vpa_s,  "min"), _v(hrv_s,  "min"), _v(hr_s, "min"),
            # month
            float(month), m_cos, m_sin,
            # daily counts
            mvpa_count, sed_count,
            # sex
            float(self.sex_code),
            # slope
            _v(acc_s,  "slope"), _v(enmo_s, "slope"), _v(mvpa_s, "slope"),
            _v(sed_s,  "slope"), _v(vpa_s,  "slope"), _v(hrv_s,  "slope"), _v(hr_s, "slope"),
            # std
            _v(acc_s,  "std"), _v(enmo_s, "std"), _v(mvpa_s, "std"),
            _v(sed_s,  "std"), _v(vpa_s,  "std"), _v(hrv_s,  "std"), _v(hr_s, "std"),
            # vpa daily count + weight
            vpa_count, self.weight_kg,
        ], dtype=np.float64)

        assert len(vec) == len(FEATURE_COLUMNS), \
            f"Feature count mismatch: {len(vec)} vs {len(FEATURE_COLUMNS)}"

        return vec
