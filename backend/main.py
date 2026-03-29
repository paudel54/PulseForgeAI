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
from pydantic import BaseModel
import chromadb
from pypdf import PdfReader
import requests

app = FastAPI(title="PulseForgeAI Backend")

# Initialize ChromaDB (Use /tmp for Vercel Serverless compatibility)
DB_PATH = "/tmp/chroma_db" if os.environ.get("VERCEL") == "1" else "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="medical_docs")

# Initialize FM Cohorts Integration if it exists
try:
    cohort_collection = chroma_client.get_collection(name="patient_cohorts")
except ValueError:
    # Not yet instantiated
    cohort_collection = None

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
        
        # 2. Construct Prompt dynamically based on Role
        if getattr(req, "role", "doctor") == "patient":
            system_role = (
                "You are an empathetic, friendly, and highly professional clinical nurse speaking directly to a patient. "
                "Use simple language and a warm, reassuring tone. "
                "The patient might ask for a health report comparing their current vitals to their 5-day and 15-day history. "
                "If they ask for a progress report, you MUST provide a detailed text report that uses clean ASCII bar charts to visualize their Heart Rate and HRV history. "
                "Draw the charts clearly and explain what the trends mean in an encouraging way."
            )
        else:
            system_role = (
                "You are a knowledgeable clinical assistant specializing in cardiac rehabilitation and physiology. "
                "Provide a clear, accurate, and medically-informed response. "
                "CRITICAL: When your answer utilizes information from the Knowledge Base Context, you MUST explicitly cite the document by its [Source: filename] within your text."
            )

        prompt = f"""
{system_role}

Patient Physiological Data & History (Polar H10):
{json.dumps(req.patient_data, indent=2)}

Knowledge Base Context (from uploaded medical documents):
{retrieved_context}
{cohort_context}

User Query:
{req.query}

Instructions: Use the Knowledge Base Context and Patient Data when they are relevant and helpful. 
When extracting facts or reasoning from the Knowledge Base, distinctly cite the original filename (e.g. "According to [Source: AHA_Guidelines.pdf]...").
If the context is not relevant, rely on your broad medical expertise. 
Always be helpful and never refuse a question due to lack of uploaded context.
"""
        
        # 3. Call local Ollama
        # Note: If Ollama isn't running, this will fail. For the hackathon demo, we'll try/except.
        try:
            payload = {
                "model": req.model,
                "prompt": prompt,
                "stream": False
            }
            headers = {
                "Content-Type": "application/json",
                "Bypass-Tunnel-Reminder": "true",
                "ngrok-skip-browser-warning": "true"
            }

            # 3. Query the external/local Ollama instance through the secure tunnel
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                headers=headers,
                timeout=300
            )    
            if response.status_code == 200:
                llm_output = response.json().get("response", "No response generated.")
            else:
                llm_output = f"Error from Ollama: {response.status_code} - Make sure Ollama and the {req.model} model are running locally."
        except requests.exceptions.RequestException:
            # Fallback for the demo if Ollama isn't up
            llm_output = f"Ollama is not reachable at {OLLAMA_URL}. \n\n[MOCKED RESPONSE] Based on the context provided, the patient's heart rate variability shows a slight decrease. Proceed with standard cardiac rehabilitation protocol."
            
        return {
            "query": req.query,
            "retrieved_context_preview": retrieved_context[:200] + "...",
            "llm_response": llm_output
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount the static directory to serve HTML/CSS/JS exactly as they are laid out
# MUST BE AT THE BOTTOM to prevent shadowing other routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
