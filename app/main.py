import os
import shutil
import gc
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.rag_engine import RAGEngine


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("--- System Initialization ---")
    engine.initialize()
    yield


app = FastAPI(title="Memory-Aware Company RAG API", lifespan=lifespan)

# --- CENTRALIZED PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Instantiate the engine
engine = RAGEngine(DATA_DIR, DB_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UPLOAD LIMITS ---
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_user"


class SessionRequest(BaseModel):
    session_id: str

@app.get("/")
def home():
    return {"status": "Online", "data_path": DATA_DIR}

@app.post("/query")
def handle_query(request: QueryRequest):
    # Using the engine instance correctly
    result = engine.query(request.question, session_id=request.session_id)
    if isinstance(result, str) and "Error" in result:
        raise HTTPException(status_code=500, detail=result)
    return result

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Sanitize the filename: strip ALL path components so something like
        # "../../evil.txt" can never escape the data directory.
        raw_name = (file.filename or "").replace("\\", "/")
        safe_name = os.path.basename(raw_name).strip()
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid filename.")

        # Extension allowlist
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{ext}' not allowed. Accepted: {sorted(ALLOWED_EXTENSIONS)}",
            )

        file_path = os.path.join(DATA_DIR, safe_name)

        # Stream to disk with a hard size cap
        written = 0
        overflow = False
        with open(file_path, "wb") as buffer:
            while True:
                blob = await file.read(1024 * 1024)
                if not blob:
                    break
                written += len(blob)
                if written > MAX_UPLOAD_BYTES:
                    overflow = True
                    break
                buffer.write(blob)

        if overflow:
            os.remove(file_path)
            raise HTTPException(status_code=413, detail="File exceeds the 50 MB limit.")
            
        gc.collect()
        engine.initialize()
        
        return {"status": "success", "message": f"Uploaded and indexed {safe_name}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@app.post("/api/clear-history")
def clear_history(request: SessionRequest):
    engine.clear_session(request.session_id)
    return {
        "status": "success",
        "message": f"Conversation memory cleared for session '{request.session_id}'.",
    }


@app.post("/api/reset-database")
def reset_database():
    try:
        gc.collect()
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            os.makedirs(DB_DIR)
        engine.initialize()
        return {"status": "success", "message": "Database reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")