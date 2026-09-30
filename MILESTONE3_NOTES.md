# Milestone 3 — Agent Coordination & Memory Systems

## Run it (step by step)

```bash
# 1. Unzip and enter the project
unzip compliance_system_milestone3.zip
cd current_project          # (or whatever the unzipped folder is called)

# 2. Create a FRESH virtual environment (important - see "Version note" below)
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Configure keys
cp .env.example .env                # Windows: copy .env.example .env
#    then edit .env and set GROQ_API_KEY=...

# 5. Run the test suite (no API key or internet needed - LLMs are mocked)
python -m pytest tests/ -v          # expect: 24 passed

# 6. Start the backend + dashboard
uvicorn main:app --reload --port 8000
#    open http://localhost:8000
```

First real run: upload a policy PDF in the dashboard and click **Rebuild Index**.
The embedding model (~90 MB) downloads once on first use.

To keep conversations across server restarts, set `USE_SQLITE_MEMORY=true` in `.env`.

## What Milestone 3 added

| Requirement | Where | How |
|---|---|---|
| 4 specialised roles | `graph/workflow.py` | Your existing classes cover them: **Planning** = `ManagerAgent`, **Research** = `ToolUsingResearchAgent`, **Analysis** = `RiskAnalysisAgent`, **Decision/Drafting** = `DraftingAgent`. Nothing renamed. |
| Short-term memory | `memory/checkpointer.py` | Graph compiled with `MemorySaver` (default) or `SqliteSaver`. Same `thread_id` = conversation continues. |
| Long-term memory | `memory/long_term_memory.py` | Separate ChromaDB collection of every completed run (query, route, risk, draft yes/no). Written automatically by `log_memory_node`; searchable via new 5th tool `search_audit_history`. |
| Coordination | `graph/workflow.py` | Shared state + conditional handoffs (`out_of_scope` / `research_only` / `research_and_draft`); every path funnels through `log_memory` before `END`. |
| API | `main.py` | `/query` accepts + returns `thread_id`; new `/threads/{id}/history`, `/audit-log`, `/audit-log/search`. |
| Dashboard | `static/index.html` | New Conversation Memory panel and searchable Audit Log panel; live node-by-node log unchanged. |
| Tests | `tests/test_milestone3.py` | 9 tests: routing, thread persistence/isolation, backward compatibility, audit logging. |

## Decisions you should be able to defend

- **No new "PlanningAgent" class.** Your Manager Agent already plans the workflow path, and your README names four agents. Adding a fifth would duplicate it. If your mentor expects a literally separate Planning class, tell me and I'll split it out.
- **Audit logging includes out-of-scope requests.** A compliance system should record what it declined, too.
- **`thread_id` is optional.** Omitting it gives a fresh isolated run, so all Milestone 2 code and tests behave as before.

## Known limitations (say these before your mentor finds them)

1. **Only the Research Agent sees conversation history.** The Manager, Risk and Drafting agents receive just the current query text. A follow-up like *"Does that apply to customer data too?"* is classified by the Manager without context, so it could be misrouted (even to `out_of_scope`). In demos, phrase follow-ups with enough context ("Does the DPDP breach rule apply to customer data too?"). Fix for next iteration: pass recent messages into `ManagerAgent.route`.
2. **Agents coordinate through shared graph state, not direct messages.** That is how LangGraph is designed, but "agent-to-agent communication" here means state handoffs.
3. **Job store in `main.py` is still in-memory** (lost on restart). Thread memory and the audit log are not affected.
4. **Search/registry tools** carry over the Milestone 2 caveats (DuckDuckGo can be rate-limited; OpenCorporates free tier is limited).

## What was and wasn't verified

Verified in my sandbox: all 24 tests; the full submit → poll → history → audit-log API flow; SQLite memory surviving a simulated restart (separate Python processes); real ChromaDB writes/reads.

**Not** verified here (sandbox has no access to Groq or HuggingFace): live LLM answers and the real embedding model. Tests use mocked LLM decisions and a small fake embedding function, so they prove the wiring and memory logic, not answer quality or semantic-search quality. Please do one live run on your machine before presenting.

## Things that changed that you should know about

- **Package versions were upgraded** (`langgraph` 0.2.45 → 1.2.12, `langchain-core` 0.3 → 1.6, etc.). The `langchain-*` packages must move together, so use a fresh virtualenv rather than upgrading in place. Your existing code and all its tests passed unchanged on the new versions.
- **Removed `tests/test_multi_agent.py`.** It was a Phase 3 leftover that patched the wrong import path and tried to call the real Groq API; `test_milestone2.py` and `test_milestone3.py` cover the same ground.
- **Three bugs found and fixed while testing** (worth mentioning as real debugging):
  - A `patch()` applied before `importlib.reload()` is silently undone by the reload — the audit-log mock in `test_milestone2.py` now applies after it.
  - A function default argument (`persist_directory=AUDIT_PERSIST_DIRECTORY`) binds at definition time, so tests couldn't redirect the audit store and leaked data into `memory_db/`. Now resolved at call time.
  - Reusing one temp path across tests made ChromaDB fail with "readonly database"; each test now gets its own directory.
