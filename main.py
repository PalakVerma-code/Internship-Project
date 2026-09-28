"""
Milestone 2 - FastAPI Backend
------------------------------------
Exposes the multi-agent + tools workflow over HTTP, so the web dashboard
(or any other client) can:
  - upload compliance PDFs and rebuild the knowledge base
  - kick off a query and get a job_id immediately (non-blocking)
  - poll that job_id for real-time step-by-step progress and the final result

Run with:
    uvicorn main:app --reload --port 8000

Beginner notes:
- "async def" endpoints let FastAPI keep handling OTHER requests while one
  slow request (like a multi-agent LLM pipeline) is still running -
  that's why file uploads and long queries don't block each other.
- We run the actual (synchronous) LangGraph workflow in a background
  thread via `asyncio.to_thread`, so it doesn't block FastAPI's event
  loop while the LLM/tools are working.
- Job state lives in a simple in-memory dict. That's fine for a
  demo/capstone; a production system would use Redis or a database
  instead so state survives a server restart.
"""

import os
import uuid
import asyncio
import shutil
import time
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
from graph.workflow import stream_compliance_workflow

app = FastAPI(title="Enterprise Legal & Compliance Multi-Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local demo; restrict this in real production
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS_DIR = "documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# In-memory job store: {job_id: {status, logs, result, error}}
JOBS: Dict[str, Dict[str, Any]] = {}


class QueryRequest(BaseModel):
    query: str


def _run_workflow_job(job_id: str, query: str) -> None:
    """
    Runs in a background thread. Streams node-by-node so the frontend can
    poll and show real progress instead of a single opaque spinner.
    """
    job = JOBS[job_id]
    try:
        for node_name, node_state in stream_compliance_workflow(query):
            timestamp = datetime.utcnow().strftime("%H:%M:%S")
            job["logs"].append({"time": timestamp, "node": node_name})
            job["latest_state"] = node_state

        final_state = job["latest_state"]
        job["status"] = "completed"
        job["result"] = {
            "route": final_state.get("route"),
            "research_answer": final_state.get("research_answer"),
            "risk_level": final_state.get("risk_level"),
            "key_risk_factors": final_state.get("key_risk_factors"),
            "recommended_urgency": final_state.get("recommended_urgency"),
            "draft": final_state.get("draft"),
            "errors": final_state.get("errors", []),
        }
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


@app.get("/health")
async def health_check():
    return {"status": "ok", "groq_key_configured": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF into documents/. Does NOT auto-rebuild the vector store
    (that can take a while) - call /rebuild-knowledge-base separately."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    save_path = os.path.join(DOCUMENTS_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"filename": file.filename, "saved_to": save_path}


@app.get("/documents")
async def list_documents():
    files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]
    return {"documents": files}


@app.post("/rebuild-knowledge-base")
async def rebuild_knowledge_base():
    """Re-indexes all PDFs currently in documents/. Runs in a thread since
    embedding generation is CPU-bound and can take a few seconds."""
    def _rebuild():
        from rag.vector_store import build_vector_store
        build_vector_store()

    try:
        await asyncio.to_thread(_rebuild)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "rebuilt"}


@app.post("/query")
async def submit_query(request: QueryRequest):
    """Starts a multi-agent workflow run in the background and returns a job_id immediately."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running", "logs": [], "result": None, "error": None, "latest_state": None}

    asyncio.create_task(asyncio.to_thread(_run_workflow_job, job_id, request.query))

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "status": job["status"],
        "logs": job["logs"],
        "result": job["result"],
        "error": job["error"],
    }


# Serve uploaded PDFs directly so the frontend can preview them
if os.path.isdir(DOCUMENTS_DIR):
    app.mount("/documents", StaticFiles(directory=DOCUMENTS_DIR), name="documents")

# Serve the frontend (index.html + assets) from /static, and at root "/"
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    async def serve_dashboard():
        return FileResponse("static/index.html")
