from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.generator import generate_answer
from app.vector_store import VectorStore
from app.ingestion import ingest_document
from app.models import QuestionRequest

import time
import shutil
import os

# ✅ FASTAPI APP MUST BE AT TOP LEVEL
app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")

templates = Jinja2Templates(directory="templates")

# ---------------- RATE LIMITER ---------------- #
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------- STORAGE SETUP ---------------- #
UPLOAD_DIR = "data/documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vector_store = VectorStore(dimension=384)
processing_status = {}

# ---------------- BACKGROUND PROCESSING ---------------- #
def process_document(file_path: str, file_id: str):
    start_time = time.time()
    try:
        print(f"[INGESTION STARTED] {file_path}")
        chunks_count = ingest_document(file_path, vector_store)
        elapsed = round(time.time() - start_time, 2)

        processing_status[file_id] = {
            "status": "completed",
            "time": elapsed
        }

        print(f"[INGESTION SUCCESS] {chunks_count} chunks stored in {elapsed}s")

    except Exception as e:
        processing_status[file_id] = {
            "status": "failed",
            "time": 0,
            "error": str(e)
        }
        print("❌ BACKGROUND INGESTION ERROR:", str(e))

# ---------------- ROUTES ---------------- #

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    return {"message": "RAG QA System is running"}

# 🔹 Upload Document
@app.post("/upload")
def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    clean_name = file.filename.replace(" ", "_")
    file_path = os.path.join(UPLOAD_DIR, clean_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_id = clean_name
    processing_status[file_id] = {"status": "processing", "time": 0}
    background_tasks.add_task(process_document, file_path, file_id)

    return {
        "message": "Document uploaded successfully. Processing started.",
        "document_id": file_id
    }

@app.get("/status/{doc_id}")
def check_status(doc_id: str):
    return processing_status.get(doc_id, {"status": "not_found"})

# 🔹 Ask Question
@app.post("/ask")

def ask_question(request: QuestionRequest):
    contexts = vector_store.search(request.question, top_k=3)

    if not contexts:
        return {
            "question": request.question,
            "answer": "❌ Answer not found in uploaded documents.",
            "retrieved_context": []
        }

    # 🚀 DIRECTLY generate answer from retrieved chunks
    answer = generate_answer(request.question, contexts)

    return {
        "question": request.question,
        "answer": answer,
        "retrieved_context": contexts
    }
