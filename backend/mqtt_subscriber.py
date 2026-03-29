import json
import logging
import os
import chromadb
try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Please install paho-mqtt: pip install paho-mqtt")
    exit()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MQTT_Live_Subscriber")

# Initialize Local ChromaDB for streaming state
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "chroma_db"))
chroma_client = chromadb.PersistentClient(path=DB_PATH)
live_collection = chroma_client.get_or_create_collection(name="live_patients")

BROKER_HOST = "broker.emqx.io"
PORT = 1883

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info(f"Connected successfully to {BROKER_HOST}:{PORT}")
        # Subscribe to both info and raw telemetry streams
        client.subscribe("pulseforgeai/+/info")
        client.subscribe("pulseforgeai/+/raw")
        logger.info("Subscribed to pulseforgeai/+/info and pulseforgeai/+/raw")
    else:
        logger.error(f"Connection failed with code {rc}")

def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload_str = msg.payload.decode('utf-8')
        data = json.loads(payload_str)
        
        # Topic structure: pulseforgeai/S000/info or pulseforgeai/S000/raw
        parts = topic.split("/")
        if len(parts) >= 3:
            subject_id = parts[1]
            stream_type = parts[2]
            
            # Use ChromaDB to store a historical semantic memory of the telemetry
            import time
            from datetime import datetime
            current_time = int(time.time())
            human_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            doc_id = f"{subject_id}_{stream_type}_{current_time}"
            
            if stream_type == "raw":
                hr = data.get("heart_rate", {}).get("avg_bpm_ecg", "N/A")
                hrv = data.get("hrv", {}).get("rmssd_ms", "N/A")
                if isinstance(hrv, float): hrv = round(hrv, 1)
                act = data.get("accelerometer", {}).get("activity", {}).get("label", "Unknown").replace("_", " ")
                
                # Semantic mapping for LLM RAG retrieval
                semantic_log = f"[{human_time} / Unix {current_time}] Patient {subject_id} recorded a heart rate of {hr} bpm and HRV of {hrv} ms. Current activity: {act}."
                
                live_collection.upsert(
                    documents=[semantic_log],
                    metadatas=[{"subject_id": subject_id, "type": stream_type, "timestamp": current_time, "json": payload_str}],
                    ids=[doc_id]
                )
                logger.info(f"[{subject_id} RAW] Appended history tick -> HR: {hr} bpm | Act: {act}")
            else:
                # ---- Intake / Google Fit / Info topic handler ----
                # Build a human-readable semantic summary for LLM RAG
                summary_lines = [f"[{human_time}] Patient {subject_id} intake/context data received."]

                # Patient info block
                info = data.get("patient_info", {})
                if info:
                    summary_lines.append(f"Age: {info.get('age', 'N/A')}, Sex: {info.get('sex', 'N/A')}, Weight: {info.get('weight', 'N/A')} kg")
                    h = info.get("history", {})
                    if h:
                        summary_lines.append(f"Medical History: Smoking={h.get('smoking','N/A')}, Surgery={h.get('surgery','N/A')}, Drinking={h.get('drinking','N/A')}")

                # Data payload block (Google Fit / vitals)
                payload = data.get("data_payload", {})
                if payload:
                    summary_lines.append(f"Resting HR: {payload.get('heart_rate', 'N/A')} bpm, Activity: {payload.get('activity_type', 'N/A')}")
                    hrv_d = payload.get("hRV", payload.get("hrv", {}))
                    summary_lines.append(f"HRV RMSSD: {hrv_d.get('Rmssd', hrv_d.get('rmssd', 'N/A'))} ms, LF/HF: {hrv_d.get('LF/HF', 'N/A')}")
                    vo2 = payload.get("VO2", {})
                    summary_lines.append(f"VO2 Max: {vo2.get('max', 'N/A')} ml/kg/min, VO2 Current: {vo2.get('current', 'N/A')} ml/kg/min")
                    summary_lines.append(f"Steps: {payload.get('step_count', 'N/A')}, Sleep: {payload.get('sleep_hours', 'N/A')} hrs, Exercise Distance: {payload.get('exercise_dist', 'N/A')} km")
                    summary_lines.append(f"Time Window: {payload.get('time_window', 'N/A')}")

                # If no recognized structure, flatten whatever came in
                if len(summary_lines) == 1:
                    for k, v in data.items():
                        summary_lines.append(f"{k}: {v}")

                semantic_doc = "\n".join(summary_lines)

                live_collection.upsert(
                    documents=[semantic_doc],
                    metadatas=[{"subject_id": subject_id, "type": "info", "timestamp": current_time, "json": payload_str}],
                    ids=[doc_id]
                )
                logger.info(f"[{subject_id} INFO] Semantic intake summary stored in ChromaDB.")

                
    except Exception as e:
        logger.error(f"Failed to process message from {msg.topic}: {e}")

if __name__ == "__main__":
    logger.info("Initializing PulseForge Edge MQTT Ingestion Service...")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(BROKER_HOST, PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Disconnecting...")
        client.disconnect()
    except Exception as e:
        logger.error(f"MQTT Error: {e}")
