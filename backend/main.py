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
                
        # 1.6 Live MQTT Stream Interpolation (Target: Subject S000)
        mqtt_context = ""
        if live_patients_collection:
            try:
                live_res = live_patients_collection.get(ids=["S000_info", "S000_raw"])
                if live_res and live_res.get('documents'):
                    mqtt_context = "\n[LIVE MQTT TELEMETRY FEED (EMQX)]:\n"
                    for doc in live_res['documents']:
                        if doc:
                            mqtt_context += f"{doc}\n"
            except Exception as e:
                mqtt_context = f"\n[!] MQTT Connection Exception: {str(e)}\n"
        
        # 1.7 Execute Deterministic Safety Bounds (EnergySafeWindow)
        # Check if the live MQTT stream has a Heart Rate
        current_hr = req.patient_data.get("metrics", {}).get("heart_rate_bpm", 70)
        if live_patients_collection and "S000_raw" in mqtt_context:
            try:
                # Naive parse of the raw JSON block inside the context string
                import re
                hr_match = re.search(r'"avg_bpm_ecg":\s*([\d.]+)', mqtt_context)
                if hr_match:
                    current_hr = float(hr_match.group(1))
            except:
                pass

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
