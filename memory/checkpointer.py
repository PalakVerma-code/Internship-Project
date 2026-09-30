"""
Milestone 3 - Short-Term Conversational Memory
--------------------------------------------------------------------
This file now keeps the app compatible with the LangGraph workflow while
also exposing Supabase-based helpers for database-backed conversation
storage. The workflow needs a valid checkpointer object to compile, so we
restore the default LangGraph in-memory checkpointer for startup safety.
"""

import os
import sqlite3
from typing import List, Dict, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from supabase_client import supabase

SQLITE_MEMORY_PATH = os.getenv("SQLITE_MEMORY_PATH", "checkpoints.sqlite")

_checkpointer = None
_sqlite_conn = None


def get_checkpointer():
    """Compatibility function expected by graph/workflow.py."""
    global _checkpointer, _sqlite_conn

    if _checkpointer is not None:
        return _checkpointer

    use_sqlite = os.getenv("USE_SQLITE_MEMORY", "false").lower() == "true"
    if use_sqlite:
        _sqlite_conn = sqlite3.connect(SQLITE_MEMORY_PATH, check_same_thread=False)
        _checkpointer = SqliteSaver(_sqlite_conn)
    else:
        _checkpointer = MemorySaver()
    return _checkpointer


def ensure_thread(thread_id: str) -> None:
    """Create the thread row if it doesn't already exist."""
    if not thread_id:
        return
    if supabase is None:
        return
    try:
        supabase.table("threads").upsert({"thread_id": thread_id}).execute()
    except Exception:
        pass


def save_message(thread_id: str, role: str, content: str) -> None:
    """Persist one message in the database for this conversation thread."""
    if not thread_id or not role or not content:
        return
    if supabase is None:
        return
    try:
        ensure_thread(thread_id)
        supabase.table("messages").insert({
            "thread_id": thread_id,
            "role": role,
            "content": content,
        }).execute()
    except Exception:
        pass


def get_messages(thread_id: str) -> List[Dict[str, Any]]:
    """Fetch all messages for a given thread, ordered oldest to newest."""
    if not thread_id or supabase is None:
        return []
    try:
        response = supabase.table("messages") \
            .select("*") \
            .eq("thread_id", thread_id) \
            .order("created_at", desc=False) \
            .execute()
        return response.data or []
    except Exception:
        return []


def get_thread_config(thread_id: str) -> dict:
    """Compatibility function for existing graph workflow code."""
    return {"configurable": {"thread_id": thread_id}}
