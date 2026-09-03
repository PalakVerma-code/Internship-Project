# Milestone 1 — Agent Foundation Development

### Project: Automated Enterprise Legal & Compliance Workflow System

# Agile documentation for your reference.
- [Agile Documentation Sheet 1](https://docs.google.com/spreadsheets/d/1qOnqI21ksRY6J5UNZIjFbSMVg5eBb623yxsqVyT7-Nk/edit?usp=sharing)
- [Agile Documentation Sheet 2](https://docs.google.com/spreadsheets/d/1BB0N4jOuBphVZ4GsUnnPzYoYj2IJBlYZ/edit?usp=sharing&ouid=116718011743655261611&rtpof=true&sd=true)
- [Agile Documentation Sheet 3](https://docs.google.com/spreadsheets/d/12MiNxLmrmoSlD3EEVdP08wuH2aw84TuwhLnHOEumQHo/edit?usp=sharing)
 
## What's in this folder
- `agent.py` — the agent code (`ComplianceAssistantAgent`) — this is your core deliverable
- `dashboard.py` — visual web dashboard for presenting/demoing the agent
- `requirements.txt` — packages to install
- `.env.example` — rename to `.env` and add your Groq API key here

 ## project Screenshot
 <img width="1527" height="695" alt="Screenshot 2026-09-02 154312" src="https://github.com/user-attachments/assets/97704518-6d58-42ef-8cce-22b6cd638841" />
 <img width="1401" height="687" alt="Screenshot 2026-09-02 154513" src="https://github.com/user-attachments/assets/bb610acc-e072-4ac7-911b-a52d6079520d" />

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
 
