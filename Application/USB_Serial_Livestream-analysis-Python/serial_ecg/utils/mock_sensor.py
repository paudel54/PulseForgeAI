"""
Mock serial sensor for UI development and testing without hardware.
Generates synthetic ECG (100Hz), Accelerometer (100Hz), and HR data
mimicking the CSV output of the ESP32 custom sensor hardware.
"""

import time
import math
import random
import numpy as np


class MockSerialSensor:
    """Generates realistic synthetic sensor data mimicking the ESP32 serial CSV output.

    Each call to get_sample() returns one row of data at 100 Hz, matching
    the firmware column format: ECG, HR, AccX, AccY, AccZ.
    """

    def __init__(self):
        self._phase = 0.0
        self._dt = 1.0 / 100.0
        self._hr_bpm = 72.0
        self._seq = 0

    def _generate_ecg_sample(self, t: float, hr: float) -> float:
        """Generate a single noisy ECG sample using a simplified PQRST model.

        Adds extra noise compared to the Polar version to simulate the
        noisier custom hardware ECG channel.
        """
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

        noise = random.gauss(0, 25)
        baseline_wander = 40.0 * math.sin(2 * math.pi * 0.15 * t)
        powerline_noise = 15.0 * math.sin(2 * math.pi * 50.0 * t)

        return value + noise + baseline_wander + powerline_noise

    def get_sample(self) -> dict:
        """Return one sample matching the serial CSV data columns."""
        self._hr_bpm = 72.0 + 5.0 * math.sin(self._phase * 0.05)

        ecg = self._generate_ecg_sample(self._phase, self._hr_bpm)

        acc_x = int(30 * math.sin(2 * math.pi * 0.3 * self._phase) + random.gauss(0, 5))
        acc_y = int(20 * math.cos(2 * math.pi * 0.25 * self._phase) + random.gauss(0, 5))
        acc_z = int(1000 + 15 * math.sin(2 * math.pi * 1.2 * self._phase) + random.gauss(0, 3))

        hr = round(self._hr_bpm + random.gauss(0, 0.5), 1)

        self._phase += self._dt
        self._seq += 1

        return {
            'ecg': ecg,
            'hr': hr,
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
        }
