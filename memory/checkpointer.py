"""
Milestone 3 - Short-Term Conversational Memory (Checkpointers)
--------------------------------------------------------------------
Gives the multi-agent workflow thread-level session memory: if the same
thread_id is reused across multiple calls, the Manager Agent, Research
Agent, Risk Analysis Agent, and Drafting Agent all see the full prior
conversation - not just the latest message.

Beginner notes:
- A LangGraph "checkpointer" snapshots the graph's State after every node
  runs, keyed by a thread_id. The NEXT time you invoke the graph with that
  same thread_id, LangGraph loads the saved state first, then merges your
  new input into it using each field's reducer (see
  `graph/workflow.py`'s `Annotated[List[AnyMessage], add_messages]` -
  that's what makes new messages APPEND to history instead of replacing
  it).
- MemorySaver keeps everything in this process's RAM. It's fast and needs
  no setup, but a server restart wipes all conversation history - fine
  for local dev/demo.
- SqliteSaver writes checkpoints to a real .sqlite file on disk, so
  conversation history survives a server restart. Use this for anything
  closer to production. Toggle it with USE_SQLITE_MEMORY=true in .env.
"""

import os
import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

SQLITE_MEMORY_PATH = os.getenv("SQLITE_MEMORY_PATH", "checkpoints.sqlite")

_checkpointer = None
_sqlite_conn = None  # kept open for the lifetime of the process


def get_checkpointer():
    """
    Returns the process-wide checkpointer instance (built once, reused).
    Set USE_SQLITE_MEMORY=true in .env to persist conversations to disk
    across server restarts; otherwise defaults to fast in-memory storage.
    """
    global _checkpointer, _sqlite_conn

    if _checkpointer is not None:
        return _checkpointer

    use_sqlite = os.getenv("USE_SQLITE_MEMORY", "false").lower() == "true"

    if use_sqlite:
        # check_same_thread=False: FastAPI's background threads (see main.py's
        # asyncio.to_thread calls) run the workflow off the main thread, so the
        # sqlite connection needs to be usable from more than one thread.
        _sqlite_conn = sqlite3.connect(SQLITE_MEMORY_PATH, check_same_thread=False)
        _checkpointer = SqliteSaver(_sqlite_conn)
    else:
        _checkpointer = MemorySaver()

    return _checkpointer


def get_thread_config(thread_id: str) -> dict:
    """Builds the LangGraph `config` dict a given thread_id needs to be passed with."""
    return {"configurable": {"thread_id": thread_id}}
