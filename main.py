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
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
from graph.workflow import stream_compliance_workflow, get_thread_history
from memory.long_term_memory import list_recent_decisions, search_past_decisions

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
    thread_id: Optional[str] = None  # omit for a one-off query; reuse to continue a conversation


def _run_workflow_job(job_id: str, query: str, thread_id: str) -> None:
    """
    Runs in a background thread. Streams node-by-node so the frontend can
    poll and show real progress instead of a single opaque spinner.
    """
    job = JOBS[job_id]
    try:
        for node_name, node_state in stream_compliance_workflow(query, thread_id=thread_id):
            timestamp = datetime.utcnow().strftime("%H:%M:%S")
            job["logs"].append({"time": timestamp, "node": node_name})
            job["latest_state"] = node_state

        final_state = job["latest_state"]
        job["status"] = "completed"
        job["result"] = {
            "thread_id": thread_id,
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
    """
    Starts a multi-agent workflow run in the background and returns a
    job_id immediately. If thread_id is omitted, a new one is generated
    and returned in the response - the frontend should save it and send
    it back on the NEXT call to continue the same conversation with
    short-term memory intact (Milestone 3).
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    thread_id = request.thread_id or str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running", "logs": [], "result": None, "error": None, "latest_state": None}

    asyncio.create_task(asyncio.to_thread(_run_workflow_job, job_id, request.query, thread_id))

    return {"job_id": job_id, "thread_id": thread_id}


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


@app.get("/threads/{thread_id}/history")
async def thread_history(thread_id: str):
    """
    Milestone 3 - Short-term memory: returns the full conversation
    (human + AI messages) held in this thread's checkpointed state, for
    display in the dashboard's conversation memory panel.
    """
    history = await asyncio.to_thread(get_thread_history, thread_id)
    return {"thread_id": thread_id, "messages": history}


@app.get("/audit-log")
async def audit_log(limit: int = 20):
    """
    Milestone 3 - Long-term memory: returns the most recent audit log
    entries (past completed decisions), newest first, for the dashboard's
    audit trail panel.
    """
    entries = await asyncio.to_thread(list_recent_decisions, limit)
    return {"entries": entries}


@app.get("/audit-log/search")
async def audit_log_search(q: str, k: int = 5):
    """
    Milestone 3 - Long-term memory: semantic search over past audit log
    entries, e.g. to check whether a similar compliance question has come
    up before and how it was handled.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty.")
    results = await asyncio.to_thread(search_past_decisions, q, k)
    return {"query": q, "results": results}


# Serve uploaded PDFs directly so the frontend can preview them
if os.path.isdir(DOCUMENTS_DIR):
    app.mount("/documents", StaticFiles(directory=DOCUMENTS_DIR), name="documents")

# Serve the frontend (index.html + assets) from /static, and at root "/"
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    async def serve_dashboard():
        return FileResponse("static/index.html")
