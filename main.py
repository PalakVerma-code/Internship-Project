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
from pypdf import PdfReader
from supabase_client import SUPABASE_CONFIGURED, supabase

load_dotenv()

from graph.workflow import stream_compliance_workflow, get_thread_history
from memory.long_term_memory import list_recent_decisions, search_past_decisions

app = FastAPI(title="Enterprise Legal & Compliance Multi-Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS_DIR = "documents"
CASE_DOCUMENTS_DIR = os.path.join(DOCUMENTS_DIR, "cases")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(CASE_DOCUMENTS_DIR, exist_ok=True)

# In-memory job store: {job_id: {status, logs, result, error}}
JOBS: Dict[str, Dict[str, Any]] = {}


class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None  # omit for a one-off query; reuse to continue a conversation
    case_document: Optional[str] = None


def _run_workflow_job(job_id: str, query: str, thread_id: str) -> None:
    """
    Runs in a background thread. Streams node-by-node so the frontend can
    poll and show real progress instead of a single opaque spinner.
    """
    job = JOBS[job_id]
    try:
        for node_name, node_state in stream_compliance_workflow(query, thread_id=thread_id):
            if node_name == "__final__":
                if isinstance(node_state, dict):
                    job["latest_state"] = node_state
                continue
            timestamp = datetime.utcnow().strftime("%H:%M:%S")
            job["logs"].append({"time": timestamp, "node": node_name})
            # A terminal/no-op graph node may yield no state update. Keep the
            # last complete state so result assembly still has the answer.
            if isinstance(node_state, dict) and node_state:
                job["latest_state"] = node_state

        final_state = job.get("latest_state") or {}
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
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    ready = groq_configured and SUPABASE_CONFIGURED
    return {
        "status": "ok" if ready else "degraded",
        "groq_key_configured": groq_configured,
        "supabase_configured": SUPABASE_CONFIGURED,
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a policy PDF without automatically rebuilding the index."""
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if os.getenv("VECTOR_BACKEND", "local").strip().lower() == "supabase":
        try:
            from rag.supabase_vector_store import upload_document as upload_supabase_document
            content = await file.read()
            saved_to = upload_supabase_document(filename, content)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Could not upload policy to Supabase: {e}")
        return {"filename": filename, "saved_to": saved_to, "backend": "supabase"}

    save_path = os.path.join(DOCUMENTS_DIR, filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"filename": filename, "saved_to": save_path, "backend": "local"}


@app.post("/case-documents")
async def upload_case_document(file: UploadFile = File(...)):
    """Store a case agreement separately from the indexed knowledge base."""
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    save_path = os.path.join(CASE_DOCUMENTS_DIR, filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"filename": filename, "message": "Case document uploaded and ready for review."}


@app.get("/case-documents")
async def list_case_documents():
    files = [
        f for f in os.listdir(CASE_DOCUMENTS_DIR)
        if f.lower().endswith(".pdf")
    ]
    return {"documents": sorted(files)}


@app.get("/documents")
async def list_documents():
    if os.getenv("VECTOR_BACKEND", "local").strip().lower() == "supabase":
        try:
            from rag.supabase_vector_store import list_documents as list_supabase_documents
            return {"documents": list_supabase_documents(), "backend": "supabase"}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Could not list Supabase policies: {e}")

    files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]
    return {"documents": files, "backend": "local"}


@app.post("/rebuild-knowledge-base")
async def rebuild_knowledge_base():
    """Re-indexes all policy PDFs for the selected vector backend."""
    def _rebuild():
        from rag.vector_store import build_vector_store
        build_vector_store()

    try:
        await asyncio.to_thread(_rebuild)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not rebuild knowledge base: {e}")
    return {
        "status": "rebuilt",
        "backend": os.getenv("VECTOR_BACKEND", "local").strip().lower(),
    }


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

    query = request.query.strip()
    if request.case_document:
        safe_name = os.path.basename(request.case_document)
        case_path = os.path.join(CASE_DOCUMENTS_DIR, safe_name)
        if not os.path.isfile(case_path):
            raise HTTPException(status_code=404, detail="Selected case document was not found.")
        try:
            reader = PdfReader(case_path)
            extracted_text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read case document: {e}")
        if not extracted_text:
            raise HTTPException(status_code=400, detail="The selected PDF contains no extractable text.")
        query = (
            f"{query}\n\n"
            f"CASE DOCUMENT: {safe_name}\n"
            "Analyze the following case document for this request. "
            "Treat it as the current agreement under review, not as permanent policy knowledge.\n"
            f"{extracted_text[:30000]}"
        )

    thread_id = request.thread_id or str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running", "logs": [], "result": None, "error": None, "latest_state": None}

    asyncio.create_task(asyncio.to_thread(_run_workflow_job, job_id, query, thread_id))

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
    app.mount("/case-documents", StaticFiles(directory=CASE_DOCUMENTS_DIR), name="case-documents")

# Serve the frontend (index.html + assets) from /static, and at root "/"
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    async def serve_dashboard():
        return FileResponse("static/index.html")
