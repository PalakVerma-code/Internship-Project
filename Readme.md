# Milestone 1 — Agent Foundation Development
### Project: Automated Enterprise Legal & Compliance Workflow System
 
## What's in this folder
- `agent.py` — the agent code (`ComplianceAssistantAgent`) — this is your core deliverable
- `dashboard.py` — visual web dashboard for presenting/demoing the agent
- `requirements.txt` — packages to install
- `.env.example` — rename to `.env` and add your Groq API key here
## How to run it (step by step)
 
1. Open a terminal inside this folder.
2. Create and activate a virtual environment:
```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
```
3. Install the packages:
```
   pip install -r requirements.txt
```
4. Get a free Groq API key at https://console.groq.com
5. Rename `.env.example` to `.env` and paste in your key.
6. Run the terminal version:
```
   python agent.py
```
   Or run the visual dashboard version (recommended for presenting):
```
   streamlit run dashboard.py
```
7. Try asking:
   - "What are our obligations under the DPDP Act for storing employee data?"
   - "Summarize POSH Act requirements for setting up an Internal Committee"
   - "What compliance filings does a private limited company need annually under the Companies Act 2013?"
## Presenting to your mentor
The dashboard (`streamlit run dashboard.py`) opens a browser window with:
- A chat interface to type live compliance questions
- A sidebar explaining the current pipeline **and your full 5-phase
  roadmap** — use this to show the mentor you understand where this fits
  in the bigger picture, not just this one piece
- An expandable "See the actual prompt sent to the LLM" panel — open this
  once during your demo to explain the prompt template concept
**Suggested demo flow:**
1. Point to the sidebar roadmap (10 seconds) — "this is Phase 1 of 5"
2. Ask 2-3 live compliance questions
3. Open the "See the actual prompt" panel once, to explain how the
   template fixes the agent's role and enforces a consistent answer
   structure (understanding → relevant law → guidance → disclaimer)
4. Be upfront about the current limitation: the agent answers from the
   model's general knowledge only — it does not yet read your company's
   actual policy PDFs or specific legal texts. That's exactly what Phase 2
   (RAG) adds next, so it doubles as your transition into the next
   milestone.
## How this maps to the Milestone 1 tasks
| Task from project doc | Where it is in the code |
|---|---|
| Configure LangChain and dependencies | `requirements.txt`, imports + `.env` loading at top of `agent.py` |
| Develop foundational AI agent | `ComplianceAssistantAgent` class |
| Prompt templates and interaction workflows | `self.prompt_template` and `handle_request()` |
| Basic testing interface | `run_cli()` in `agent.py`, plus `dashboard.py` for a visual version |
 
## What to submit / show your mentor
- `agent.py` and `dashboard.py`
- A screenshot or short recording of the dashboard running with 2-3 sample
  compliance questions
- A short paragraph (3-4 lines), in your own words, explaining: what the
  agent does now, and what Phase 2 (RAG over legal/policy documents) will
  add next
 