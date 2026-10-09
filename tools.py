"""
Milestone 2 - Production-Grade Custom Tools
------------------------------------------------
Four REAL tools an agent can call. No mock/fake logic anywhere here:

1. retrieve_legal_clauses    -> real ChromaDB vector search over your indexed PDFs
2. search_regulatory_updates -> real live internet search (DuckDuckGo, no API key needed;
                                 upgrades to Tavily automatically if TAVILY_API_KEY is set)
3. calculate_compliance_risk_score -> real deterministic scoring algorithm over contract text
4. verify_corporate_entity   -> real HTTP call to the public OpenCorporates REST API

Beginner notes:
- The `@tool` decorator from LangChain turns a normal Python function into
  something an LLM can "call" by name with structured arguments. The
  docstring you write becomes the tool's description - the LLM reads it
  to decide WHEN to use this tool, so keep docstrings precise.
- Every tool below returns a STRING. Agents work with text in and text
  out, so even structured results (like a risk score) get formatted into
  readable text before being returned.
"""

import os
import re
import logging
from typing import List

import requests
from langchain_core.tools import tool

from rag.vector_store import get_or_build_vector_store
from memory.long_term_memory import search_past_decisions

logger = logging.getLogger("compliance_tools")
logging.basicConfig(level=logging.INFO)


# =====================================================================
# TOOL 1: Real RAG Retriever (wraps Phase 2 ChromaDB vector store)
# =====================================================================
@tool
def retrieve_legal_clauses(query: str, doc_id: str = "") -> str:
    """
    Search the company's indexed compliance policy documents and Indian
    legal framework PDFs for clauses relevant to the query. Use this when
    the user asks about internal policy, a specific act/law, or wants a
    grounded answer citing an actual document. Returns the most relevant
    clauses with their source filename and page number. The optional doc_id
    is accepted for compatibility with model-generated calls; retrieval
    searches the complete indexed collection.
    """
    try:
        store = get_or_build_vector_store()
        results = store.similarity_search(query, k=4)
    except FileNotFoundError as e:
        return f"ERROR: No documents indexed yet. {e}"
    except Exception as e:
        logger.exception("retrieve_legal_clauses failed")
        return f"ERROR: Vector store retrieval failed: {e}"

    if not results:
        return "No relevant clauses found in the indexed documents for this query."

    formatted = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[{i}] ({source}, page {page})\n{doc.page_content.strip()}")

    return "\n\n".join(formatted)


# =====================================================================
# TOOL 2: Real Web Search (DuckDuckGo by default, Tavily if key present)
# =====================================================================
def _search_with_tavily(query: str, api_key: str) -> str:
    from langchain_community.tools.tavily_search import TavilySearchResults

    search = TavilySearchResults(max_results=4, tavily_api_key=api_key)
    results = search.invoke(query)
    formatted = []
    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "")[:300]
        formatted.append(f"{title}\n{url}\n{content}")
    return "\n\n".join(formatted) if formatted else "No results found."


def _search_with_duckduckgo(query: str) -> str:
    from ddgs import DDGS

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=4):
            title = r.get("title", "Untitled")
            url = r.get("href", "")
            body = r.get("body", "")[:300]
            results.append(f"{title}\n{url}\n{body}")
    return "\n\n".join(results) if results else "No results found."


@tool
def search_regulatory_updates(query: str) -> str:
    """
    Search the live internet for recent regulatory or legal updates - use
    this when the user asks about something that may have changed
    recently (a new amendment, a recent notification, a deadline change)
    that would NOT be in the static indexed policy documents. Returns
    titles, URLs, and short snippets from real search results.
    """
    query = f"{query} India compliance regulation"
    tavily_key = os.getenv("TAVILY_API_KEY")

    try:
        if tavily_key:
            return _search_with_tavily(query, tavily_key)
        return _search_with_duckduckgo(query)
    except Exception as e:
        logger.exception("search_regulatory_updates failed")
        # Graceful degradation: don't crash the pipeline over a search failure
        return (
            f"ERROR: Live search unavailable right now ({type(e).__name__}: {e}). "
            f"Proceed using indexed documents only; note this limitation to the user."
        )


# =====================================================================
# TOOL 3: Real Algorithmic Risk Calculator (no LLM - pure deterministic logic)
# =====================================================================
# Each risk keyword maps to a weight. This is a real, auditable, rule-based
# scoring model - the kind compliance teams can actually inspect and trust,
# as opposed to an opaque LLM guess.
_RISK_KEYWORDS = {
    r"\bindemnif\w*\b": 15,
    r"\bunlimited liability\b": 20,
    r"\bpenalty\b": 10,
    r"\btermination for convenience\b": 8,
    r"\bnon-compete\b": 10,
    r"\bconfidential\w*\b": 5,
    r"\bgoverning law\b": -5,      # presence of a governing-law clause reduces ambiguity risk
    r"\barbitration\b": -5,        # defined dispute resolution reduces risk
    r"\bforce majeure\b": -5,
    r"\bauto[- ]?renewal\b": 8,
    r"\bpersonal data\b": 10,
    r"\bdata breach\b": 15,
    r"\bnon-disclosure\b": 3,
}


@tool
def calculate_compliance_risk_score(contract_text: str) -> str:
    """
    Calculate a deterministic compliance risk score (0-100) for a piece of
    contract or policy text, based on presence of known high-risk clause
    patterns (indemnification, unlimited liability, data breach exposure,
    etc.) versus risk-reducing clauses (arbitration, governing law,
    force majeure). Use this when the user provides contract/clause text
    and wants a numeric risk assessment, not just a qualitative one.
    """
    if not contract_text or not contract_text.strip():
        return "ERROR: No contract text provided to analyze."

    text_lower = contract_text.lower()
    score = 30  # baseline score - every contract carries some inherent risk
    matched_factors = []

    for pattern, weight in _RISK_KEYWORDS.items():
        matches = re.findall(pattern, text_lower)
        if matches:
            contribution = weight * len(matches)
            score += contribution
            sign = "+" if contribution >= 0 else ""
            matched_factors.append(f"'{pattern}' x{len(matches)} ({sign}{contribution})")

    # Length-based adjustment: extremely short clauses are hard to assess safely
    word_count = len(contract_text.split())
    if word_count < 20:
        score += 10
        matched_factors.append("very short text, insufficient context (+10)")

    score = max(0, min(100, score))

    if score < 30:
        band = "Low"
    elif score < 60:
        band = "Medium"
    else:
        band = "High"

    result = (
        f"Compliance Risk Score: {score}/100 ({band})\n"
        f"Word count analyzed: {word_count}\n"
        f"Contributing factors:\n"
    )
    result += "\n".join(f"  - {f}" for f in matched_factors) if matched_factors else "  - No specific risk keywords detected"
    return result


# =====================================================================
# TOOL 4: Real External REST API (OpenCorporates - public, no key required
# for limited use) to verify a company's registration
# =====================================================================
OPENCORPORATES_API_URL = "https://api.opencorporates.com/v0.4/companies/search"


@tool
def verify_corporate_entity(company_name: str, jurisdiction_code: str = "in") -> str:
    """
    Verify a corporate entity's registration status by querying the public
    OpenCorporates company registry API. Use this when the user wants to
    confirm whether a vendor, partner, or counterparty is a legitimately
    registered company before drafting a contract or compliance
    correspondence involving them. jurisdiction_code defaults to "in"
    (India); pass a different 2-letter code for other countries.
    """
    if not company_name or not company_name.strip():
        return "ERROR: No company name provided."

    params = {"q": company_name, "jurisdiction_code": jurisdiction_code, "per_page": 3}

    try:
        response = requests.get(OPENCORPORATES_API_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return "ERROR: Corporate registry API timed out. Try again or verify manually."
    except requests.exceptions.HTTPError as e:
        return f"ERROR: Corporate registry API returned an error: {e}"
    except requests.exceptions.RequestException as e:
        return f"ERROR: Could not reach corporate registry API: {e}"

    try:
        data = response.json()
        companies = data.get("results", {}).get("companies", [])
    except (ValueError, KeyError) as e:
        return f"ERROR: Unexpected API response format: {e}"

    if not companies:
        return (
            f"No registered entity found matching '{company_name}' in "
            f"jurisdiction '{jurisdiction_code}'. This does not necessarily "
            f"mean the company is not registered elsewhere - verify the "
            f"correct jurisdiction and spelling."
        )

    formatted = []
    for entry in companies:
        c = entry.get("company", {})
        formatted.append(
            f"- {c.get('name', 'Unknown')} | Status: {c.get('current_status', 'Unknown')} "
            f"| Incorporation Date: {c.get('incorporation_date', 'Unknown')} "
            f"| Company Number: {c.get('company_number', 'Unknown')}"
        )

    return f"Found {len(formatted)} matching entit{'y' if len(formatted)==1 else 'ies'}:\n" + "\n".join(formatted)


# =====================================================================
# TOOL 5: Long-Term Memory Search (Milestone 3 - audit log / past decisions)
# =====================================================================
@tool
def search_audit_history(query: str) -> str:
    """
    Search the system's own audit log of PAST completed compliance
    decisions (not the policy PDFs) for similar prior questions and how
    they were handled - including their risk level and whether a document
    was drafted. Use this when the user asks whether something similar
    has come up before, or when checking for consistency with past
    decisions would be useful. Returns an empty-history message if no
    audit log entries exist yet.
    """
    try:
        results = search_past_decisions(query, k=3)
    except Exception as e:
        logger.exception("search_audit_history failed")
        return f"ERROR: Could not search audit history: {e}"

    if not results:
        return "No past audit log entries found yet - this may be one of the first queries handled by this system."

    formatted = []
    for i, entry in enumerate(results, start=1):
        query_text = entry.get("query", "")
        summary = entry.get("research_summary", "")
        risk_level = entry.get("risk_level", "Unknown")
        draft_produced = entry.get("draft_produced", False)
        formatted.append(
            f"[{i}] ({entry.get('created_at', 'unknown time')})\n"
            f"Question: {query_text}\n"
            f"Risk: {risk_level}; Draft produced: {draft_produced}\n"
            f"Summary: {summary}"
        )
    return "\n\n".join(formatted)


# List of all tools, used by the agent/graph layer to bind them to an LLM
ALL_TOOLS = [
    retrieve_legal_clauses,
    search_regulatory_updates,
    calculate_compliance_risk_score,
    verify_corporate_entity,
    search_audit_history,
]


if __name__ == "__main__":
    # Quick manual smoke test of the pure-logic tool (no network needed)
    sample = (
        "The vendor shall indemnify the company against all claims. "
        "This agreement includes an unlimited liability clause and an "
        "auto-renewal term. Disputes shall be resolved through arbitration."
    )
    print(calculate_compliance_risk_score.invoke({"contract_text": sample}))
