import os
import streamlit as st
from dotenv import load_dotenv
from agent import ComplianceAssistantAgent
 
load_dotenv()
 
st.set_page_config(
    page_title="Enterprise Legal & Compliance Agent - Milestone 1",
    page_icon="⚖️",
    layout="wide",
)
 
 
@st.cache_resource
def load_agent():
    return ComplianceAssistantAgent()
 
 
# ---------------------------------------------------------
# Sidebar: roadmap + architecture explanation (for presenting)
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚖️ Compliance Agent")
    st.caption("Automated Enterprise Legal & Compliance Workflow System")
 
    st.markdown("### Current pipeline (Phase 1)")
    st.markdown(
        """
        1. Employee/officer submits a compliance query
        2. Query is inserted into a **prompt template** that fixes the
           agent's role and required answer structure
        3. The filled prompt is sent to **Groq (LLaMA 3.3 70B)**
        4. Agent returns a structured response: understanding, relevant
           legal area, guidance, and a disclaimer
        """
    )
 
    st.markdown("### Full project roadmap")
    st.markdown(
        """
        - ✅ **Phase 1** — Single base agent, secure `.env` key management
          *(this demo)*
        - ⏳ **Phase 2** — RAG pipeline: index corporate policy PDFs and
          Indian legal frameworks (ChromaDB/FAISS)
        - ⏳ **Phase 3** — Multi-agent orchestration (LangGraph): Manager,
          Legal RAG Research, Risk Analysis, Drafting agents
        - ⏳ **Phase 4** — State & memory management (checkpointers +
          long-term vector logs)
        - ⏳ **Phase 5** — FastAPI REST layer + production UI dashboard
        """
    )
    st.divider()
    st.caption("⚠️ General guidance only — not a substitute for legal counsel")
 
# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------
st.title("Legal & Compliance Assistant — Foundation Demo")
st.caption("Milestone 1: Agent Environment Setup & Foundation Development")
 
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "No API key found. Create a `.env` file (see .env.example) "
        "and add your Groq API key before running this demo."
    )
    st.stop()
 
agent = load_agent()
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
user_input = st.chat_input(
    "Ask a compliance question (e.g. 'What are our obligations under the DPDP Act for storing employee data?')"
)
 
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    with st.chat_message("assistant"):
        with st.spinner("Agent is analyzing the query..."):
            filled_prompt = agent.build_prompt(user_input)
            response = agent.handle_request(user_input)
        st.markdown(response)
 
        with st.expander("🔍 See the actual prompt sent to the LLM"):
            st.code(filled_prompt, language="text")
 
    st.session_state.messages.append({"role": "assistant", "content": response})
 
if not st.session_state.messages:
    st.info(
        "👋 Try asking something like:\n\n"
        "- What are our obligations under the DPDP Act for storing employee data?\n"
        "- Summarize POSH Act requirements for setting up an Internal Committee\n"
        "- What compliance filings does a private limited company need annually under the Companies Act 2013?"
    )
 
 