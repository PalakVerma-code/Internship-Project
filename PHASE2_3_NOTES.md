# Phase 2 & 3 — Implementation Notes

## Phase 2: Knowledge Retrieval Pipeline (RAG)

| Task | File | Status |
|---|---|---|
| PDF ingestion | `rag/document_loader.py` | ✅ Done |
| Chunking (RecursiveCharacterTextSplitter) | `rag/document_loader.py` | ✅ Done |
| Embeddings (HuggingFace, local) | `rag/vector_store.py` | ✅ Done |
| Vector DB (ChromaDB, persisted to disk) | `rag/vector_store.py` | ✅ Done |

**Why HuggingFace embeddings instead of Groq:** Groq only serves chat/LLM
inference, not embedding models. `sentence-transformers/all-MiniLM-L6-v2`
runs locally, is free, and needs no API key — a deliberate, correct
architecture choice, not a workaround.

## Phase 3: Multi-Agent Orchestration (LangGraph)

| Agent | File | Specialty |
|---|---|---|
| Manager/Router | `agents/manager_agent.py` | Classifies query → `research_only` or `research_and_draft` |
| Legal RAG Research | `agents/legal_research_agent.py` | Retrieves relevant chunks, answers grounded in real documents, cites sources |
| Risk Analysis | `agents/risk_analysis_agent.py` | Assigns Low/Medium/High risk, flags urgency |
| Drafting | `agents/drafting_agent.py` | Writes a ready-to-use formal document (only runs when routed) |

**Graph wiring:** `graph/workflow.py`
```
START → Manager → Legal Research → Risk Analysis ─┬─→ [route=draft] → Drafting → END
                                                     └─→ [route=research_only] → END
```

Each agent has exactly ONE responsibility (Single Responsibility
Principle) and agents don't call each other directly — they only read
from and write to the shared `ComplianceWorkflowState`, wired together by
the graph. This means any agent can be swapped, tested, or extended
without touching the others.

## Testing

- `tests/test_rag_pipeline.py` — 4 tests covering PDF loading, error
  handling, chunking, and metadata traceability. Run against the real
  sample PDF in `documents/`.
- `tests/test_multi_agent.py` — 6 tests covering routing logic, safe
  fallback behavior, and the two possible paths through the LangGraph
  workflow (with and without drafting). All LLM and vector store calls are
  mocked, so these run free and instantly, and verify the *logic and
  wiring* independent of actual LLM answer quality.

**Result: 10/10 tests passing** (verified before delivery).

## Known limitation carried into Phase 4

The workflow currently has no memory between separate questions — every
query starts fresh with no awareness of prior conversation. That's exactly
what Phase 4 (state & memory management) adds next.

## What to show your mentor
1. The roadmap sidebar in the dashboard (Phases 1–3 now complete)
2. A live "research only" query, showing the pipeline status expand and
   collapse, sources cited
3. A live "draft" query (e.g. "Draft a notice about...") showing the
   Drafting Agent activate and produce a downloadable document
4. Open `graph/workflow.py` briefly to show the actual node/edge
   definition — this is the clearest way to prove the "multi-agent"
   claim is real code, not just marketing language in the README
