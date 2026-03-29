import json
import logging
import random
import time
from dataclasses import dataclass
try:
    import paho.mqtt.client as mqtt
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MQTT_Pipeline")

@dataclass
class PatientVitals:
    patient_id: str
    timestamp: float
    hr_bpm: float
    hrv_rmssd_ms: float
    sqi: float
    activity_class: str
    alert_level: str

class VitalsPublisher:
    """
    DGX Spark Edge-Node MQTT Publisher.
    Publishes parsed Polar H10 telemetry to the local Mosquitto broker
    for multi-patient concurrent monitoring.
    """
    def __init__(self, broker_host="test.mosquitto.org", port=1883):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        try:
            self.client.connect(broker_host, port)
            logger.info(f"Connected to Edge MQTT Broker at {broker_host}:{port}")
        except Exception as e:
            logger.warning(f"MQTT Broker offline. Simulating dispatch. Error: {e}")
            self.client = None

    def publish_vitals(self, vitals: PatientVitals):
        topic = f"pulseforgeai/patients/{vitals.patient_id}/vitals"
        payload = json.dumps(vitals.__dict__)
        if self.client:
            self.client.publish(topic, payload, qos=0)
            logger.info(f"[QoS 0] Published Vitals: {topic} -> {vitals.hr_bpm} BPM")
            
            # Elevate QoS to exactly-once delivery for critical clinical alerts
            if vitals.alert_level in ("warning", "critical"):
                alert_topic = f"pulseforgeai/alerts/{vitals.patient_id}"
                self.client.publish(alert_topic, payload, qos=2)
                logger.warning(f"[QoS 2] Emitted Critical Alert: {alert_topic}")

class PulseForgeSubscriber:
    """
    Lead Orchestrator Subscriber.
    Listens to the Edge MQ bus to trigger MedGemma-27B interventions.
    """
    def __init__(self, broker_host="test.mosquitto.org"):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_message = self.on_message
        try:
            self.client.connect(broker_host, 1883)
            self.client.subscribe("pulseforgeai/alerts/#", qos=2)
            self.client.loop_start()
        except:
            pass

    def on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        logger.error(f"🚨 INTERVENTION REQUIRED 🚨 : Patient {payload['patient_id']} HR {payload['hr_bpm']} ({payload['alert_level']})")

if __name__ == "__main__":
    # Simulate a multi-patient clinic scenario running on DGX Spark Edge
    pub = VitalsPublisher()
    sub = PulseForgeSubscriber()
    
    patients = ["patient_A_bed1", "patient_B_bed2", "patient_C_bed3"]
    
    for _ in range(3):
        for pid in patients:
            hr = random.uniform(60, 155)
            alert = "critical" if hr > 150 else "none"
            vitals = PatientVitals(
                patient_id=pid,
                timestamp=time.time(),
                hr_bpm=hr,
                hrv_rmssd_ms=random.uniform(20, 50),
                sqi=0.98,
                activity_class="exercise",
                alert_level=alert
            )
            pub.publish_vitals(vitals)
            time.sleep(0.5)
