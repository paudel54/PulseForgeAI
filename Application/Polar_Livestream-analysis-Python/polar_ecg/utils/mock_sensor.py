"""
Mock Polar H10 sensor for UI development and testing without hardware.
Generates synthetic ECG (130Hz), Accelerometer (100Hz), and HR (1Hz) data.
"""

import time
import math
import random
import numpy as np
from collections import deque


class MockPolarH10:
    """Generates realistic synthetic biosensor data mimicking a Polar H10."""

    def __init__(self):
        self._running = False
        self._ecg_phase = 0.0
        self._acc_phase = 0.0
        self._hr_bpm = 72.0
        self._rr_ms = 60000.0 / self._hr_bpm

    def _generate_ecg_beat(self, t: float, hr: float) -> float:
        """Generate a single ECG sample using a simplified PQRST waveform model."""
        period = 60.0 / hr
        phase = (t % period) / period

        value = 0.0

        # P wave
        p_center, p_width = 0.10, 0.04
        if abs(phase - p_center) < 3 * p_width:
            value += 150.0 * math.exp(-((phase - p_center) ** 2) / (2 * p_width ** 2))

        # Q wave
        q_center, q_width = 0.20, 0.01
        if abs(phase - q_center) < 3 * q_width:
            value -= 100.0 * math.exp(-((phase - q_center) ** 2) / (2 * q_width ** 2))

        # R wave (sharp peak)
        r_center, r_width = 0.23, 0.008
        if abs(phase - r_center) < 3 * r_width:
            value += 900.0 * math.exp(-((phase - r_center) ** 2) / (2 * r_width ** 2))

        # S wave
        s_center, s_width = 0.26, 0.012
        if abs(phase - s_center) < 3 * s_width:
            value -= 200.0 * math.exp(-((phase - s_center) ** 2) / (2 * s_width ** 2))

        # T wave
        t_center, t_width = 0.42, 0.06
        if abs(phase - t_center) < 3 * t_width:
            value += 250.0 * math.exp(-((phase - t_center) ** 2) / (2 * t_width ** 2))

        noise = random.gauss(0, 8)
        baseline_wander = 20.0 * math.sin(2 * math.pi * 0.15 * t)

        return value + noise + baseline_wander

    def get_ecg_frame(self, n_samples: int = 73) -> tuple:
        """
        Generate an ECG data frame matching bleakheart format.
        Returns: ('ECG', timestamp_ns, [sample1_uV, ..., sampleN_uV])
        """
        dt = 1.0 / 130.0
        samples = []
        for _ in range(n_samples):
            sample = self._generate_ecg_beat(self._ecg_phase, self._hr_bpm)
            samples.append(int(sample))
            self._ecg_phase += dt

        self._hr_bpm = 72.0 + 5.0 * math.sin(self._ecg_phase * 0.05)

        return ('ECG', time.time_ns(), samples)

    def get_acc_frame(self, n_samples: int = 16) -> tuple:
        """
        Generate an accelerometer data frame matching bleakheart format.
        Returns: ('ACC', timestamp_ns, [(x,y,z), ...])
        """
        dt = 1.0 / 100.0
        samples = []
        for _ in range(n_samples):
            # Simulate chest-mounted sensor: gravity in Z, slight movement in X/Y
            x = int(30 * math.sin(2 * math.pi * 0.3 * self._acc_phase) + random.gauss(0, 5))
            y = int(20 * math.cos(2 * math.pi * 0.25 * self._acc_phase) + random.gauss(0, 5))
            z = int(1000 + 15 * math.sin(2 * math.pi * 1.2 * self._acc_phase) + random.gauss(0, 3))
            samples.append((x, y, z))
            self._acc_phase += dt

        return ('ACC', time.time_ns(), samples)

    def get_hr_frame(self) -> tuple:
        """
        Generate a heart rate data frame matching bleakheart format.
        Returns: ('HR', timestamp_ns, (hr_bpm, rr_ms), None)
        """
        hr = self._hr_bpm + random.gauss(0, 1.5)
        rr = 60000.0 / hr + random.gauss(0, 15)
        return ('HR', time.time_ns(), (round(hr), round(rr)), None)


class MockECGGenerator:
    """
    Continuous mock ECG generator that produces data at realistic intervals.
    Designed for integration with the data acquisition worker.
    """

    def __init__(self):
        self._sensor = MockPolarH10()
        self._ecg_buffer = deque()
        self._acc_buffer = deque()
        self._last_hr_time = 0.0

    def generate_ecg_chunk(self) -> tuple:
        return self._sensor.get_ecg_frame()

    def generate_acc_chunk(self) -> tuple:
        return self._sensor.get_acc_frame()

    def generate_hr_if_due(self) -> tuple | None:
        now = time.time()
        if now - self._last_hr_time >= 1.0:
            self._last_hr_time = now
            return self._sensor.get_hr_frame()
        return None
