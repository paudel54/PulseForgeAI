"""
VO2max estimation using validated non-exercise methods.

Primary:   Uth-Sørensen-Overgaard-Pedersen (2004)
             VO2max = 15.3 × (HRmax / HRrest)
             HRmax  = 208 − 0.7 × age  (Tanaka et al., 2001)
             HRrest = 10th-percentile of recent HR readings

Secondary: HRV-based approximation (RMSSD, age, sex)
             Derived from the log-linear relationship reported by
             Dong (2016) and Plews et al. (2013).

Fitness classification uses ACSM normative VO2max tables.

Inspiration:
  Spathis et al. (2022). "Longitudinal cardio-respiratory fitness prediction
  through wearables in free-living environments." Nature Digital Medicine.
  (github.com/sdimi/cardiofitness) — statistical-feature approach for
  deriving VO2max from ECG/accelerometer wearable streams.

References:
  Uth N et al. (2004). Med Sci Sports Exerc. 36(4):695–701.
  Tanaka H et al. (2001). J Am Coll Cardiol. 37(1):153–6.
  Dong JG (2016). Front Physiol. 7:258.
  ACSM Guidelines for Exercise Testing and Prescription, 10th ed.
"""

import math
from collections import deque


# ACSM normative VO2max thresholds (ml/kg/min) by age-band and sex.
# Each row: (upper_bound_exclusive, category_label).
# Values below the first threshold → first category, etc.
_ACSM_NORMS: dict = {
    "male": [
        ((13, 25), [(33, "Very Poor"), (42, "Poor"), (52, "Fair"), (61, "Good"), (math.inf, "Excellent")]),
        ((25, 35), [(31, "Very Poor"), (39, "Poor"), (49, "Fair"), (58, "Good"), (math.inf, "Excellent")]),
        ((35, 45), [(30, "Very Poor"), (38, "Poor"), (47, "Fair"), (56, "Good"), (math.inf, "Excellent")]),
        ((45, 55), [(26, "Very Poor"), (34, "Poor"), (43, "Fair"), (51, "Good"), (math.inf, "Excellent")]),
        ((55, 65), [(23, "Very Poor"), (30, "Poor"), (38, "Fair"), (46, "Good"), (math.inf, "Excellent")]),
        ((65, 120), [(20, "Very Poor"), (26, "Poor"), (33, "Fair"), (41, "Good"), (math.inf, "Excellent")]),
    ],
    "female": [
        ((13, 25), [(28, "Very Poor"), (35, "Poor"), (43, "Fair"), (52, "Good"), (math.inf, "Excellent")]),
        ((25, 35), [(26, "Very Poor"), (33, "Poor"), (40, "Fair"), (48, "Good"), (math.inf, "Excellent")]),
        ((35, 45), [(23, "Very Poor"), (30, "Poor"), (37, "Fair"), (45, "Good"), (math.inf, "Excellent")]),
        ((45, 55), [(20, "Very Poor"), (27, "Poor"), (34, "Fair"), (42, "Good"), (math.inf, "Excellent")]),
        ((55, 65), [(18, "Very Poor"), (24, "Poor"), (30, "Fair"), (37, "Good"), (math.inf, "Excellent")]),
        ((65, 120), [(16, "Very Poor"), (21, "Poor"), (27, "Fair"), (34, "Good"), (math.inf, "Excellent")]),
    ],
}

# UI accent colours that match the dashboard dark theme.
CATEGORY_COLORS: dict = {
    "Very Poor": "#f38ba8",
    "Poor":      "#fab387",
    "Fair":      "#f9e2af",
    "Good":      "#a6e3a1",
    "Excellent": "#89dceb",
    "Unknown":   "#6c7086",
}

# Minimum HR samples before we trust the resting-HR percentile.
_MIN_HR_SAMPLES = 30


class VO2MaxEstimator:
    """
    Estimates VO2max from heart-rate notifications and HRV metrics produced
    by a Polar H10 (or compatible) device.

    Feed data in real-time:
        estimator.update_hr(hr_bpm)   # once per HR notification (~1 Hz)

    Then query:
        result = estimator.best_estimate(rmssd=42.3)
    """

    def __init__(self, age: int = 30, sex: str = "male", weight_kg: float = 70.0):
        self.age = age
        self.sex = sex          # "male" | "female"
        self.weight_kg = weight_kg

        # Rolling 10-minute HR buffer (1 Hz × 600 s).
        self._hr_history: deque = deque(maxlen=600)

    # ------------------------------------------------------------------
    #  Profile management
    # ------------------------------------------------------------------

    def update_profile(self, age: int, sex: str, weight_kg: float) -> None:
        self.age = int(age)
        self.sex = sex
        self.weight_kg = float(weight_kg)

    def update_hr(self, hr_bpm: float) -> None:
        """Ingest one HR notification.  Physiologically implausible values are dropped."""
        if 30 < hr_bpm < 220:
            self._hr_history.append(float(hr_bpm))

    # ------------------------------------------------------------------
    #  Derived properties
    # ------------------------------------------------------------------

    @property
    def hr_max_predicted(self) -> float:
        """Tanaka et al. (2001): HRmax = 208 − 0.7 × age."""
        return 208.0 - 0.7 * self.age

    @property
    def hr_rest_estimate(self):
        """
        10th-percentile of the rolling HR window as a resting-HR proxy.
        Returns None until _MIN_HR_SAMPLES samples have been collected.
        """
        if len(self._hr_history) < _MIN_HR_SAMPLES:
            return None
        import numpy as np
        return float(np.percentile(list(self._hr_history), 10))

    @property
    def samples_collected(self) -> int:
        return len(self._hr_history)

    # ------------------------------------------------------------------
    #  Estimation methods
    # ------------------------------------------------------------------

    def estimate_uth_sorensen(self) -> tuple:
        """
        Uth-Sørensen formula: VO2max = 15.3 × (HRmax / HRrest).

        Returns
        -------
        (vo2max_float, label_str)   on success
        (None,         reason_str)  when data are insufficient
        """
        hr_rest = self.hr_rest_estimate
        if hr_rest is None:
            remaining = max(0, _MIN_HR_SAMPLES - self.samples_collected)
            return None, f"Collecting resting HR… ({remaining}s left)"

        if hr_rest < 35:
            return None, "Resting HR implausibly low — recheck sensor"

        vo2max = 15.3 * (self.hr_max_predicted / hr_rest)
        return round(vo2max, 1), "Uth-Sørensen (HR)"

    def estimate_from_rmssd(self, rmssd: float) -> tuple:
        """
        HRV-based approximation.

        Uses the log-linear relationship between RMSSD and VO2max reported by
        Dong (2016), with age and sex corrections anchored to ACSM norms.
        Individual error is typically ±8–12 ml/kg/min.

        Returns
        -------
        (vo2max_float, label_str)   on success
        (None,         reason_str)  when RMSSD is unavailable
        """
        if rmssd is None or rmssd <= 0:
            return None, "RMSSD unavailable"

        ln_r = math.log(max(rmssd, 1.0))
        vo2max = 6.5 * ln_r + 15.0

        if self.sex == "female":
            vo2max -= 7.0

        # ~0.25 ml/kg/min per year decline past age 25
        vo2max += (25.0 - self.age) * 0.25

        return round(max(vo2max, 10.0), 1), "HRV (RMSSD)"

    def best_estimate(self, rmssd: float = None) -> dict:
        """
        Return the most reliable available VO2max estimate as a dict.

        Priority: Uth-Sørensen (if resting HR established) → HRV-based → None.

        Keys in the returned dict
        -------------------------
        vo2max          float | None   primary estimate (ml/kg/min)
        method          str            label or pending-reason message
        category        str | None     ACSM fitness category
        category_color  str            hex colour for the category label
        hr_rest         float | None   estimated resting HR (bpm)
        hr_max          float          age-predicted HRmax (bpm)
        rmssd_estimate  float | None   HRV-based secondary estimate
        """
        vo2_uth, label_uth = self.estimate_uth_sorensen()
        vo2_hrv, label_hrv = self.estimate_from_rmssd(rmssd)

        if vo2_uth is not None:
            primary_vo2, primary_method = vo2_uth, label_uth
        elif vo2_hrv is not None:
            primary_vo2, primary_method = vo2_hrv, label_hrv
        else:
            return {
                "vo2max":         None,
                "method":         label_uth,  # pending-reason
                "category":       None,
                "category_color": CATEGORY_COLORS["Unknown"],
                "hr_rest":        None,
                "hr_max":         round(self.hr_max_predicted),
                "rmssd_estimate": None,
            }

        category = self._get_fitness_category(primary_vo2)
        hr_rest  = self.hr_rest_estimate

        return {
            "vo2max":         primary_vo2,
            "method":         primary_method,
            "category":       category,
            "category_color": CATEGORY_COLORS.get(category, CATEGORY_COLORS["Unknown"]),
            "hr_rest":        round(hr_rest) if hr_rest is not None else None,
            "hr_max":         round(self.hr_max_predicted),
            "rmssd_estimate": vo2_hrv,
        }

    # ------------------------------------------------------------------
    #  Fitness classification
    # ------------------------------------------------------------------

    def _get_fitness_category(self, vo2max: float) -> str:
        """Look up ACSM normative fitness category for this user's age and sex."""
        sex_key = "female" if self.sex == "female" else "male"
        for (age_lo, age_hi), brackets in _ACSM_NORMS[sex_key]:
            if age_lo <= self.age < age_hi:
                for threshold, cat in brackets:
                    if vo2max < threshold:
                        return cat
        return "Unknown"
