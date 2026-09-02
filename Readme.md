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

## How this maps to the Milestone 1 tasks
| Task from project doc | Where it is in the code |
|---|---|
| Configure LangChain and dependencies | `requirements.txt`, imports + `.env` loading at top of `agent.py` |
| Develop foundational AI agent | `ComplianceAssistantAgent` class |
| Prompt templates and interaction workflows | `self.prompt_template` and `handle_request()` |
| Basic testing interface | `run_cli()` in `agent.py`, plus `dashboard.py` for a visual version |
 
