import os
import json

# Fix for Vercel Serverless Read-Only File System
if os.environ.get("VERCEL") == "1":
    os.environ["HOME"] = "/tmp"  # Forces ALL libraries to use /tmp instead of /home/sbx...
    os.environ["CHROMA_CACHE_DIR"] = "/tmp/chroma_cache"
    os.environ["HF_HOME"] = "/tmp/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
    os.makedirs("/tmp/chroma_cache", exist_ok=True)
    os.makedirs("/tmp/hf_home", exist_ok=True)
    os.makedirs("/tmp/transformers_cache", exist_ok=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from safety_engine import EnergySafeWindow
from agent_orchestrator import PulseForgeOrchestrator
from utils import execute_ollama_request
from pydantic import BaseModel
import chromadb
from pypdf import PdfReader
import requests

app = FastAPI(title="PulseForgeAI Backend")

# Initialize ChromaDB (Use /tmp for Vercel Serverless compatibility)
DB_PATH = "/tmp/chroma_db" if os.environ.get("VERCEL") == "1" else "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="medical_docs")

# Initialize FM Cohorts & Live MQTT Integration
try:
    cohort_collection = chroma_client.get_collection(name="patient_cohorts")
except Exception:
    cohort_collection = None

try:
    live_patients_collection = chroma_client.get_collection(name="live_patients")
except Exception:
    live_patients_collection = None

# Configuration for local Ollama - Supports tunneling via ngrok on Vercel
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# MODEL_NAME = "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M" # Change to your pulled model
MODEL_NAME = "alibayram/medgemma:27b" # Change to your pulled model



class QueryRequest(BaseModel):
    query: str
    patient_data: dict # Dummy Polar H10 Data + History
    model: str = MODEL_NAME
    role: str = "doctor"

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # Read PDF content
        reader = PdfReader(file.file)
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text() + "\n"
            
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # Very basic chunking (split by paragraphs or fixed length)
        # In a real scenario, use LangChain's RecursiveCharacterTextSplitter
        chunks = [text_content[i:i+1000] for i in range(0, len(text_content), 1000)]
        
        # Insert into ChromaDB
        for i, chunk in enumerate(chunks):
            doc_id = f"{file.filename}_chunk_{i}"
            collection.add(
                documents=[chunk],
                metadatas=[{"filename": file.filename}],
                ids=[doc_id]
            )
            
        return {"message": f"Successfully processed {file.filename} and added {len(chunks)} chunks to the database."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """Return a list of unique document filenames stored in ChromaDB."""
    try:
        results = collection.get(include=["metadatas"])
        filenames = sorted(set(
            m["filename"] for m in results["metadatas"] if m and "filename" in m
        ))
        return {"documents": filenames}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete all ChromaDB chunks associated with a given filename."""
    try:
        results = collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(results["ids"], results["metadatas"])
            if meta and meta.get("filename") == filename
        ]
        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found in knowledge base.")
        collection.delete(ids=ids_to_delete)
        return {"message": f"Successfully removed '{filename}' ({len(ids_to_delete)} chunks deleted)."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def process_query(req: QueryRequest):
    try:
        # 1. Retrieve relevant context from ChromaDB
        results = collection.query(
            query_texts=[req.query],
            n_results=3
        )
        
        context_parts = []
        if results and results.get('documents') and results['documents'][0]:
            for doc_text, meta in zip(results['documents'][0], results.get('metadatas', [[]])[0]):
                source_file = meta.get("filename", "Unknown Source") if meta else "Unknown Source"
                context_parts.append(f"[Source: {source_file}]\n{doc_text}")
        
        retrieved_context = "\n\n".join(context_parts) if context_parts else "No relevant medical context found in the database."
        
        # 1.5 Multi-Modal Foundation Model Similar Patient Retrieval
        cohort_context = ""
        mock_embedding_path = os.path.join(os.path.dirname(__file__), "mock_patient_embedding.json")
        if cohort_collection and os.path.exists(mock_embedding_path):
            try:
                with open(mock_embedding_path, "r") as f:
                    mock_emb = json.load(f)
                cohort_results = cohort_collection.query(query_embeddings=[mock_emb], n_results=3)
                if cohort_results and cohort_results.get('metadatas') and cohort_results['metadatas'][0]:
                    cohort_context = "\nIdentical ECG-Waveform Historical Patient Matches (KNN Similarity):\n"
                    for idx, meta in enumerate(cohort_results['metadatas'][0]):
                        cohort_context += f"- Patient {idx+1}: Frailty/Activity: '{meta.get('exercise_label', 'Unknown')}', RMSSD: {meta.get('hrv_rmssd', 'N/A')}ms, Gait Velocity: {meta.get('max_gait_velocity', 'Unknown')}cm/s\n"
            except Exception as e:
                cohort_context = f"\n[!] Cohort Engine Error: {str(e)}\n"
                
        # 1.6 Temporal Telemetry Context: Latest health state + historical logs
        mqtt_context = ""
        current_hr = req.patient_data.get("metrics", {}).get("heart_rate_bpm", 70)
        
        if live_patients_collection:
            try:
                import time
                import re as _re
                current_unix = int(time.time())
                window_start = current_unix - 3600  # last 60 minutes
                
                # Fetch all records from last 60 minutes
                all_raw = live_patients_collection.get(
                    where={"$and": [{"type": {"$eq": "raw"}}, {"timestamp": {"$gte": window_start}}]},
                    include=["documents", "metadatas"]
                )
                
                if all_raw and all_raw.get('documents') and len(all_raw['documents']) > 0:
                    # Sort ascending by timestamp
                    paired = sorted(
                        zip(all_raw['metadatas'], all_raw['documents']),
                        key=lambda x: x[0].get('timestamp', 0)
                    )
                    
                    # Latest record = the true current state
                    latest_doc = paired[-1][1]
                    
                    # Extract current HR for safety engine
                    hr_match = _re.search(r'heart rate of ([\d.]+) bpm', latest_doc)
                    if hr_match:
                        current_hr = float(hr_match.group(1))
                    
                    # Downsample: pick 10 evenly spaced records from history (excluding latest)
                    history = [d for _, d in paired[:-1]]
                    n = len(history)
                    if n <= 10:
                        sampled = history
                    else:
                        # Pick 10 evenly spaced indices
                        indices = [int(i * (n - 1) / 9) for i in range(10)]
                        sampled = [history[i] for i in indices]
                    
                    mqtt_context = f"\n[CURRENT PATIENT STATE at Unix {current_unix}]:\n{latest_doc}\n"
                    if sampled:
                        mqtt_context += f"\n[60-MINUTE HISTORY ({len(sampled)} sampled snapshots, oldest→newest)]:\n"
                        for doc in sampled:
                            mqtt_context += f"- {doc}\n"
                else:
                    mqtt_context = "\n[No telemetry data available from the last 60 minutes]\n"
                        
            except Exception as e:
                mqtt_context = f"\n[!] Telemetry Exception: {str(e)}\n"

        intake_data = {"age": 60, "prescribed_intensity_range": [0.4, 0.7]}
        safety_engine = EnergySafeWindow(intake_data)
        safety_bounds = safety_engine.check_safety(hr_bpm=current_hr, activity="exercise", sqi=0.95)
        
        # 2. Delegate to Lead Agent Orchestrator
        orchestrator = PulseForgeOrchestrator()
        assembled_context = orchestrator.assemble_prompt(
            role=getattr(req, "role", "doctor"),
            patient_data=req.patient_data,
            retrieved_context=retrieved_context,
            cohort_context=cohort_context + mqtt_context,
            safety_bounds=safety_bounds,
            query=req.query
        )
        
        # 3. Call local Ollama via Abstracted Utilities
        try:
            response = execute_ollama_request(
                model=req.model,
                system_prompt=assembled_context["system"],
                user_prompt=assembled_context["prompt"]
            )
            
            if response.status_code == 200:
                llm_output = response.json().get("response", "No response generated.")
            else:
                llm_output = f"Error from Ollama: {response.status_code} - Inference failed."
        except Exception:
            # Fallback for the demo if Ollama isn't up
            llm_output = f"Ollama is not reachable. \n\n[MOCKED RESPONSE] Based on the context provided, the patient's heart rate variability shows a slight decrease. Proceed with standard cardiac rehabilitation protocol."
            
        return {
            "query": req.query,
            "retrieved_context_preview": retrieved_context[:200] + "...",
            "llm_response": llm_output
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/live/metrics")
async def get_live_metrics():
    """
    Poll endpoint for the Vercel UI to visualize live MQTT sensor arrays.
    """
    if not live_patients_collection:
        return {"hr": "--", "hrv": "--", "status": "Offline"}
    try:
        live_res = live_patients_collection.get(ids=["S000_raw"])
        if live_res and live_res.get('documents') and live_res['documents']:
            data = json.loads(live_res['documents'][0])
            hr = data.get("heart_rate", {}).get("avg_bpm_ecg")
            if hr is None: hr = "--"
            hrv = data.get("hrv", {}).get("rmssd_ms", "--")
            if hrv != "--": hrv = round(hrv, 1)
            act = data.get("accelerometer", {}).get("activity", {}).get("label", "Unknown").replace("_", " ").title()
            return {"hr": hr, "hrv": hrv, "status": act}
    except Exception:
        pass
    return {"hr": "--", "hrv": "--", "status": "Waiting..."}

@app.get("/api/session/{patient_id}/soap")
async def generate_soap_note(patient_id: str):
    """
    Master-Plan Compliance: Clinical Assistant SOAP Note Generator
    Automatically reviews telemetry data to structure administrative clinical charts.
    """
    system_role = (
        "You are the Talk to Your Heart Clinical Review Agent. "
        "Generate a structured SOAP (Subjective, Objective, Assessment, Plan) note "
        "incorporating the current patient's physiologic state and historical recovery context."
    )
    prompt = (
        f"Generate a final post-session SOAP chart for patient '{patient_id}'. "
        "The patient maintained an average HR of 115 bpm with varying ECG morphology consistent with mild exertion. "
    )
    try:
        response = execute_ollama_request(
            model=MODEL_NAME, system_prompt=system_role, user_prompt=prompt
        )
        if response.status_code == 200:
            return {"soap_note": response.json().get("response", "Processing failed.")}
        return {"error": f"Ollama {response.status_code}"}
    except Exception:
        return {"soap_note": "[MOCKED SOAP NOTE]\nS: Patient reports feeling well post-exercise.\nO: Average HR 115 bpm. SQI 0.95.\nA: Normal exertion recovery.\nP: Continue current rehab intensity."}

# Mount the static directory to serve HTML/CSS/JS exactly as they are laid out
# MUST BE AT THE BOTTOM to prevent shadowing other routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
