"""
Milestone 3 - Long-Term Knowledge Retention (Audit Log Memory)
--------------------------------------------------------------------
A durable, searchable record of every completed workflow run, stored in
its OWN ChromaDB collection (separate from the policy-document knowledge
base in rag/vector_store.py). This is what lets the system say "has a
similar compliance question come up before, and how was it handled?" -
real institutional memory, not just a per-conversation memory.

Beginner notes:
- This is intentionally a SEPARATE ChromaDB collection from the policy
  PDFs. The policy vector store answers "what does our documentation
  say?"; this audit store answers "what have WE (the agents) previously
  decided?" - different question, different index.
- Every completed workflow run gets embedded and stored automatically
  (see the `log_memory_node` in graph/workflow.py) - no one has to
  remember to call this manually.
- Agents can then search this history via the `search_audit_history` tool
  in tools.py, the same way they search policy documents.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.vector_store import get_embedding_model

AUDIT_PERSIST_DIRECTORY = os.getenv("AUDIT_MEMORY_DIR", "memory_db")
AUDIT_COLLECTION_NAME = "audit_logs"

_audit_store: Optional[Chroma] = None


def get_audit_store(persist_directory: Optional[str] = None) -> Chroma:
    """Builds (or reuses) the persistent audit-log vector store.

    persist_directory defaults to the CURRENT value of the module-level
    AUDIT_PERSIST_DIRECTORY, read at call time (not at import time) - this
    matters for tests, which need to point this at a temp directory by
    reassigning `long_term_memory.AUDIT_PERSIST_DIRECTORY` before the
    first call. A plain `= AUDIT_PERSIST_DIRECTORY` default parameter
    binds once at function-definition time and would silently ignore that
    override.
    """
    global _audit_store
    if _audit_store is not None:
        return _audit_store

    if persist_directory is None:
        persist_directory = AUDIT_PERSIST_DIRECTORY

    embeddings = get_embedding_model()
    _audit_store = Chroma(
        collection_name=AUDIT_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return _audit_store


def log_decision_outcome(
    thread_id: str,
    query: str,
    route: str,
    risk_level: str,
    key_risk_factors: str,
    recommended_urgency: str,
    research_answer: str,
    had_draft: bool,
) -> str:
    """
    Persists one completed workflow run as a searchable audit log entry.
    Called automatically by graph/workflow.py's log_memory_node after
    every run (including out-of-scope ones, for a complete audit trail).
    Returns the new entry's unique id.
    """
    entry_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    summary_text = (
        f"Query: {query}\n"
        f"Route: {route}\n"
        f"Risk Level: {risk_level}\n"
        f"Key Risk Factors: {key_risk_factors}\n"
        f"Recommended Urgency: {recommended_urgency}\n"
        f"Research Summary: {(research_answer or '')[:500]}\n"
        f"Draft Produced: {'Yes' if had_draft else 'No'}"
    )

    doc = Document(
        page_content=summary_text,
        metadata={
            "thread_id": thread_id,
            "timestamp": timestamp,
            "route": route,
            "risk_level": risk_level,
            "query": query,
        },
    )

    store = get_audit_store()
    store.add_documents([doc], ids=[entry_id])
    return entry_id


def search_past_decisions(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Finds past audit log entries most similar in meaning to the given
    query. Returns an empty list (not an error) if the audit log is empty
    or doesn't exist yet - a brand new system has no history, which is a
    normal state, not a failure.
    """
    store = get_audit_store()
    try:
        if store._collection.count() == 0:
            return []
    except Exception:
        return []

    results = store.similarity_search(query, k=k)
    return [
        {
            "content": doc.page_content,
            "timestamp": doc.metadata.get("timestamp"),
            "thread_id": doc.metadata.get("thread_id"),
            "risk_level": doc.metadata.get("risk_level"),
            "route": doc.metadata.get("route"),
        }
        for doc in results
    ]


def list_recent_decisions(limit: int = 20) -> List[Dict[str, Any]]:
    """Returns the most recent audit log entries, newest first - used by
    the dashboard's audit log panel."""
    store = get_audit_store()
    try:
        raw = store._collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    entries = []
    for doc_text, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
        entries.append({
            "content": doc_text,
            "timestamp": meta.get("timestamp"),
            "thread_id": meta.get("thread_id"),
            "risk_level": meta.get("risk_level"),
            "route": meta.get("route"),
            "query": meta.get("query"),
        })

    entries.sort(key=lambda e: e.get("timestamp") or "", reverse=True)
    return entries[:limit]
