import os
import chromadb
from pypdf import PdfReader

# Directory Configurations
LITERATURE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Reference_Literature"))
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

def main():
    print("==================================================")
    print(" PulseForgeAI Offline Literature Ingestion Engine")
    print("==================================================")
    
    # Ensure literature directory exists
    if not os.path.exists(LITERATURE_DIR):
        print(f"[*] Creating directory at: {LITERATURE_DIR}")
        os.makedirs(LITERATURE_DIR)
        print("[!] No PDFs found. Please drop medical PDFs into the Reference_Literature folder and run this script again.")
        return

    # Scan for PDFs
    pdf_files = [f for f in os.listdir(LITERATURE_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"[!] No PDFs found in {LITERATURE_DIR}. Please add clinical guidelines or papers and run again.")
        return
        
    print(f"[*] Found {len(pdf_files)} PDF(s) to process. Initializing offline ChromaDB...")
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name="medical_docs")
    
    total_chunks = 0
    
    for filename in pdf_files:
        filepath = os.path.join(LITERATURE_DIR, filename)
        print(f"\n[>] Processing: {filename}...")
        
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            if not text.strip():
                print(f"    [!] Warning: No extractable text found in {filename}.")
                continue
                
            # Naive chunking (1000 chars roughly)
            chunk_size = 1000
            overlap = 100
            chunks = []
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size].strip()
                if len(chunk) > 50:
                    chunks.append(chunk)
                    
            if not chunks:
                continue
                
            print(f"    [*] Extracted {len(chunks)} contextual chunks. Injecting into vector database...")
            
            # Use chromadb backend
            collection.add(
                documents=chunks,
                metadatas=[{"filename": filename}] * len(chunks),
                ids=[f"{filename}_bulk_{i}" for i in range(len(chunks))]
            )
            
            total_chunks += len(chunks)
            print(f"    [OK] Successfully ingested {filename}.")
            
        except Exception as e:
            print(f"    [X] Error processing {filename}: {str(e)}")

    print("\n==================================================")
    print(f" [SUCCESS] Ingestion Complete!")
    print(f" Total new knowledge chunks synthesized: {total_chunks}")
    print(" PulseForgeAI Agent Knowledge Base is updated and ready.")
    print("==================================================")

if __name__ == "__main__":
    main()
