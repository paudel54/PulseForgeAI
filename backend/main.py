import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from pypdf import PdfReader
import requests

app = FastAPI(title="PulseForgeAI Backend")

# Mount the static directory to serve HTML/CSS/JS exactly as they are laid out
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Initialize ChromaDB locally
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="medical_docs")

# Configuration for local Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "medgemma" # Change to your pulled model

class QueryRequest(BaseModel):
    query: str
    patient_data: dict # Dummy Polar H10 Data

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

@app.post("/query")
async def process_query(req: QueryRequest):
    try:
        # 1. Retrieve relevant context from ChromaDB
        results = collection.query(
            query_texts=[req.query],
            n_results=3
        )
        
        retrieved_context = "\n".join(results['documents'][0]) if results['documents'] else "No relevant medical context found in the database."
        
        # 2. Construct Prompt for MedGemma
        prompt = f"""
        System: You are an expert clinical cardiologist assistant.
        
        Medical Context (from uploaded documents):
        {retrieved_context}
        
        Patient Physiological Data (Polar H10):
        {json.dumps(req.patient_data, indent=2)}
        
        User Query:
        {req.query}
        
        Action: Based ONLY on the provided medical context and patient data, generate a clinical summary report addressing the user query.
        """
        
        # 3. Call local Ollama
        # Note: If Ollama isn't running, this will fail. For the hackathon demo, we'll try/except.
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }, timeout=30)
            
            if response.status_code == 200:
                llm_output = response.json().get("response", "No response generated.")
            else:
                llm_output = f"Error from Ollama: {response.status_code} - Make sure Ollama and the {MODEL_NAME} model are running locally."
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
