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
            
            # Use ChromaDB to store the latest state for the RAG pipeline
            doc_id = f"{subject_id}_{stream_type}"
            
            live_collection.upsert(
                documents=[payload_str],
                metadatas=[{"subject_id": subject_id, "type": stream_type}],
                ids=[doc_id]
            )
            
            if stream_type == "raw":
                hr = data.get("heart_rate", {}).get("avg_bpm_ecg", "N/A")
                act = data.get("accelerometer", {}).get("activity", {}).get("label", "N/A")
                logger.info(f"[{subject_id} RAW] Upserted tick -> HR: {hr} bpm | Act: {act}")
            else:
                logger.info(f"[{subject_id} INFO] Intake state synchronized to ChromaDB.")
                
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
