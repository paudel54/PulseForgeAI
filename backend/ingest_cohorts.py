import os
import json
import chromadb
try:
    import pandas as pd
except ImportError:
    print("Please run: pip install pandas pyarrow")
    exit(1)

# Directory Configurations
BASE_DIR = os.path.dirname(__file__)
PARQUET_PATH = os.path.abspath(os.path.join(BASE_DIR, "lookup_table", "ecg_fm_lookup_table_.parquet"))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
MOCK_EMB_PATH = os.path.join(BASE_DIR, "mock_patient_embedding.json")

def main():
    print("==================================================")
    print(" PulseForgeAI ECG Cohort Ingestion Engine")
    print("==================================================")
    
    if not os.path.exists(PARQUET_PATH):
        print(f"[!] Critical Error: Parquet table not found at {PARQUET_PATH}")
        return

    print("[*] Loading large Parquet feature matrix into memory...")
    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        print(f"[!] Error parsing Parquet: {e}. (Ensure pyarrow is installed)")
        return
        
    print(f"[*] Successfully loaded {len(df)} patient sessions.")
    
    # Isolate embedding columns (emb_0 to emb_X)
    emb_cols = [c for c in df.columns if str(c).startswith("emb_")]
    if not emb_cols:
        print("[!] No 'emb_' columns found in Parquet table!")
        return
        
    print(f"[*] Found Foundation Model multi-modal matrix with {len(emb_cols)} dimensions.")

    print("[*] Initializing ChromaDB Cohort Spatial Index...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    # We use cosine similarity space for deep learning embeddings
    collection = chroma_client.get_or_create_collection(
        name="patient_cohorts",
        metadata={"hnsw:space": "cosine"}
    )
    
    # We process in batches of 1000 to avoid ChromaDB memory limits
    batch_size = 1000
    total_ingested = 0

    # Extract the very first row to act as our "Live Test Patient" mock embedding for the Vercel demo
    first_row_emb = df.iloc[0][emb_cols].tolist()
    with open(MOCK_EMB_PATH, "w") as f:
        json.dump(first_row_emb, f)
    print(f"[*] Exported baseline reference embedding to {MOCK_EMB_PATH} for live Vercel testing.")

    def safe_float(val):
        if pd.isna(val) or val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def safe_str(val):
        if pd.isna(val) or val is None:
            return "Unknown"
        return str(val)

    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        embeddings = batch_df[emb_cols].values.tolist()
        
        # Prepare rich clinical metadata for the LLM
        metadatas = []
        ids = []
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            meta = {
                "source_record": safe_str(row.get("record_name", f"Patient_{start_idx+i}")),
                "hrv_rmssd": safe_float(row.get("hrv_rmssd")),
                "exercise_label": safe_str(row.get("exercise_label", "Unknown")),
                "max_gait_velocity": safe_float(row.get("bal_max_gait_line_velocity_cm_s")),
                "mean_hr": safe_float(row.get("hrv_mean_hr"))
            }
            
            metadatas.append(meta)
            ids.append(f"physionet_cohort_{start_idx + i}")
            
        print(f"    [>] Injecting spatial batch {start_idx} to {end_idx}...")
        collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        total_ingested += len(batch_df)

    print("\n==================================================")
    print(f" [SUCCESS] Multi-Modal Cohort Ingestion Complete!")
    print(f" Total clinical arrays mathematically indexed: {total_ingested}")
    print(" PulseForgeAI will now execute KNN spatial retrieval.")
    print("==================================================")

if __name__ == "__main__":
    main()
