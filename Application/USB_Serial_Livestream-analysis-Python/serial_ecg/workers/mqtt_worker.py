"""mqtt_worker.py -- Non-blocking MQTT publisher for PulseForgeAI telemetry streaming.

Implements a PyQt QThread-backed publisher that bridges the real-time signal
processing pipeline with an external MQTT broker (broker.emqx.io, port 1883).

Topic schema (all payloads JSON, keyed by subject_id)
-----------------------------------------------------
  pulseforgeai/{subject_id}/info  -- participant intake form (sent once on record-start)
  pulseforgeai/{subject_id}/raw   -- unified 5-second burst:
      raw_ecg       : 500 filtered ECG samples (100 Hz x 5 s)
      heart_rate    : avg_bpm_ecg, n_r_peaks
      hrv_metrics   : rmssd_ms, sdnn_ms, lf_hf
      ecg_morphology: p_ms, qrs_ms, qt_ms, qtc_ms, st_ms
      ecg_quality   : sqi, sqi_metrics (nk, qrs_energy, kurtosis)
      accelerometer : mean_mag_mg, var_mag_mg2, spectral_entropy, median_freq_hz
      unix_timestamp, window_s
"""

import json
import queue
from PyQt5.QtCore import QThread, pyqtSignal
import paho.mqtt.client as mqtt


class MQTTWorker(QThread):
    """Thread-safe MQTT publisher. See module docstring for full API/topic reference."""
    log_msg = pyqtSignal(str)
    
    def __init__(self, broker="broker.emqx.io", port=1883):
        super().__init__()
        self.broker = broker
        self.port = port
        self.msg_queue = queue.Queue()
        self.running = False
        
        try:
            callback_ver = mqtt.CallbackAPIVersion.VERSION2
            self.client = mqtt.Client(callback_ver)
        except AttributeError:
            self.client = mqtt.Client()

        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish    = self._on_publish

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            self.log_msg.emit(f"MQTT connected securely to {self.broker}:{self.port}")
        else:
            self.log_msg.emit(f"MQTT connect rejected: code {reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        self.log_msg.emit(f"MQTT disconnected (code {reason_code})")

    def _on_publish(self, client, userdata, mid, reason_code=None, properties=None):
        pass

    def publish(self, topic: str, payload_dict: dict):
        """Thread-safe method to add a message to the publish queue."""
        try:
            safe_payload = json.dumps(payload_dict)
            self.msg_queue.put((topic, safe_payload))
        except TypeError as e:
            self.log_msg.emit(f"MQTT Serialization Error on topic {topic}: {e}")

    def run(self):
        self.running = True
        try:
            self.log_msg.emit(f"Connecting to {self.broker}:{self.port}...")
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            self.log_msg.emit(f"MQTT socket initiation failed: {e}")
            self.running = False
            return
            
        while self.running:
            try:
                topic, payload_str = self.msg_queue.get(timeout=0.1)
                self.client.publish(topic, payload_str, qos=0)
                self.msg_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                self.log_msg.emit(f"MQTT publish loop error: {e}")
                
    def stop(self):
        self.running = False
        while not self.msg_queue.empty():
            try:
                self.msg_queue.get_nowait()
                self.msg_queue.task_done()
            except queue.Empty:
                break
                
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except:
            pass
            
        self.quit()
        self.wait()
