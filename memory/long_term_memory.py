"""
Milestone 3 - Long-Term Knowledge Retention (Supabase-backed Audit Log)
--------------------------------------------------------------------
This stores each completed workflow result in a database rather than a
local Chroma index. It gives the app durable, searchable institutional
memory for past compliance decisions.
"""

import os
from typing import Dict, Any, List

from supabase_client import supabase


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
    """Insert one completed workflow run into the audit log."""
    if not query:
        return ""

    response = supabase.table("audit_logs").insert({
        "thread_id": thread_id,
        "query": query,
        "route": route,
        "risk_level": risk_level,
        "key_risk_factors": key_risk_factors,
        "recommended_urgency": recommended_urgency,
        "research_summary": research_answer,
        "draft_produced": had_draft,
    }).execute()

    data = response.data or []
    if not data:
        return ""
    return str(data[0].get("id", ""))


def search_past_decisions(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Simple text search over past audit entries using Supabase."""
    if not query or not query.strip():
        return []

    response = supabase.table("audit_logs") \
        .select("*") \
        .or_(f"query.ilike.%{query}%,research_summary.ilike.%{query}%") \
        .limit(k) \
        .execute()
    return response.data or []


def list_recent_decisions(limit: int = 20) -> List[Dict[str, Any]]:
    """Return newest decisions, sorted newest-first."""
    response = supabase.table("audit_logs") \
        .select("*") \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return response.data or []
