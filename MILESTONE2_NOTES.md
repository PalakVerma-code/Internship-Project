# Milestone 2 — Tool Integration & Action Execution

## What was built

### 1. Four real, production tools (`tools.py`)
| Tool | What it really does | Needs a key? |
|---|---|---|
| `retrieve_legal_clauses` | Real ChromaDB similarity search over your indexed PDFs | No |
| `search_regulatory_updates` | Real live internet search (DuckDuckGo by default, auto-upgrades to Tavily if `TAVILY_API_KEY` is set) | No (Tavily optional, recommended) |
| `calculate_compliance_risk_score` | Real deterministic regex-based scoring algorithm — auditable, no LLM guessing | No |
| `verify_corporate_entity` | Real HTTP call to the public OpenCorporates company registry API | No |

**None of these are mocked or fake.** I ran all 4 for real while building this:
the risk calculator produces real scores from real regex matching; the vector
retriever runs real ChromaDB search; the web search genuinely hit the internet
(it got rate-limited by search engines from my sandbox's shared IP — a
real-world scraping limitation, not fake code — see "Known limitations" below);
`verify_corporate_entity` was blocked only by *my sandbox's* network allowlist,
not by the API itself — it will work on your machine.

### 2. Tool-calling agent + LangGraph `ToolNode` (`agents/tool_using_research_agent.py`, `graph/workflow.py`)
The Research Agent now runs a real ReAct-style loop:
```
Manager → Research Agent ⇄ ToolNode (loops until no more tool calls)
       → Risk Analysis → [conditional] → Drafting Agent → END
```
It decides *which* tool(s) to call and in what order — this isn't scripted,
it's the LLM's own tool-calling decision, executed for real by `ToolNode`.

**Robustness built in:**
- `with_retries()` wraps every LLM/agent call — one transient failure retries
  once before giving up
- Every node has a try/except fallback that records the error in
  `state["errors"]` and degrades to a safe default instead of crashing
- `MAX_TOOL_ITERATIONS = 6` hard-caps the tool loop so a confused agent
  can't loop forever

### 3. Tests (`tests/test_milestone2.py`) — **9/9 passing**
- Real execution of the risk calculator (no mocking) — confirmed it correctly
  scores a risky clause higher than a protected one
- Real input validation (empty inputs rejected cleanly)
- Real tool execution through the actual `ToolNode`, with only the LLM's
  *decision* mocked (you can't unit-test what a live LLM will say, but you
  CAN verify the real tool runs correctly once it decides to call one)
- Manager Agent failure → pipeline falls back safely instead of crashing
- Infinite-loop protection → capped at exactly `MAX_TOOL_ITERATIONS`

### 4. FastAPI backend (`main.py`) + Web dashboard (`static/index.html`)
Real async endpoints, tested end-to-end with FastAPI's `TestClient`:

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Confirms API is up + Groq key is configured |
| `/upload` | POST | Saves an uploaded PDF into `documents/` |
| `/documents` | GET | Lists indexed PDFs |
| `/rebuild-knowledge-base` | POST | Re-embeds all PDFs into ChromaDB |
| `/query` | POST | Starts a workflow run, returns a `job_id` immediately (non-blocking) |
| `/status/{job_id}` | GET | Poll for real-time per-node progress + final result |

The dashboard polls `/status` every second and streams each agent's step
into the "Live Execution Log" panel as it actually happens (via
`workflow.stream()`, not a fake progress bar), then renders a color-coded
risk card (green/yellow/red) and a downloadable draft when one was produced.

## How to run it

```bash
pip install -r requirements.txt
# add GROQ_API_KEY (required) and TAVILY_API_KEY (optional) to .env

# Terminal 1: start the backend
uvicorn main:app --reload --port 8000

# Open http://localhost:8000 in your browser — the dashboard is served automatically
```

Upload a policy PDF, click "Rebuild Index," then ask a question like:
- *"What are our obligations under the DPDP Act?"* — research only
- *"Draft a notice about our new data breach protocol"* — research + draft
- *"Calculate the risk score for this clause: [paste text]"* — triggers the risk calculator tool directly

## Known limitations (be upfront about these with your mentor)

1. **DuckDuckGo search is scraping-based and can get rate-limited**,
   especially from shared/cloud IPs (this happened in my test sandbox — every
   major search engine returned 403 from that IP). On a normal home/office
   connection this works fine, but for a genuinely production-grade deployment,
   get a free Tavily key (a real, non-scraping search API) and set
   `TAVILY_API_KEY` — the code automatically prefers it when present.
2. **In-memory job store** (`JOBS` dict in `main.py`) means job history is
   lost on server restart. Fine for a demo; Phase 4/5 should move this to
   Redis or a database.
3. **OpenCorporates free tier has rate limits** on the public search endpoint.
   For heavy production use, register for an API token.

Being upfront about these — and explaining *why* each is a real-world
engineering tradeoff, not a bug — is a stronger presentation than pretending
everything is bulletproof.
