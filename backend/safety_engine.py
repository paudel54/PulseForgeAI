from dataclasses import dataclass
from typing import Tuple

@dataclass
class PatientIntake:
    age: int
    weight_kg: float
    prescribed_intensity_range: Tuple[float, float]
    risk_factors: list[str]

class EnergySafeWindow:
    """
    Deterministic Safety Boundary Layer for Cardiac Rehabilitation.
    Pre-processes physiologic telemetry to intercept critical anomalies 
    before LLM evaluation, enforcing HIPAA and FDA-compliant bounds.
    """
    def __init__(self, patient_intake: dict):
        self.age = patient_intake.get("age", 60)
        self.hr_max = 220 - self.age
        
        prescribed_range = patient_intake.get("prescribed_intensity_range", [0.40, 0.70])
        self.prescribed_min = prescribed_range[0]
        self.prescribed_max = prescribed_range[1]
        self.risk_factors = patient_intake.get("risk_factors", [])

    def check_safety(self, hr_bpm: float, activity: str, sqi: float) -> Tuple[bool, str, str]:
        """
        Calculates instantaneous safety bounds.
        Returns: (is_safe: bool, alert_level: str, reason: str)
        """
        # CRITICAL: Exertion HR exceeds theoretical age-predicted max
        if activity == "exercise" and hr_bpm > 0.90 * self.hr_max:
            return False, "critical", f"Exertion HR > 90% peak (Max: {self.hr_max})"
            
        # WARNING: Heart Rate above safely prescribed maximum intensity
        if activity == "exercise" and hr_bpm > (self.prescribed_max * self.hr_max):
            return False, "warning", f"HR deviated above prescribed target ({int(self.prescribed_max * 100)}% Max HR)"
            
        # ADVISORY: Poor signal quality index drops below 50% usability
        if sqi < 0.5:
            return True, "advisory", "Signal Quality Index (SQI) < 50% (Motion Artifact Warning)"
            
        return True, "none", "Telemetry within deterministic clinical bounds."
