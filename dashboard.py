"""
Phase 2 & 3 - Visual Multi-Agent Dashboard
------------------------------------------------
Shows the full multi-agent compliance workflow running live: which agent
is active, what documents were retrieved, the risk assessment, and the
final draft (if one was needed) - all in one browser dashboard.

Run with:
    streamlit run dashboard.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Enterprise Legal & Compliance Multi-Agent System",
    page_icon="⚖️",
    layout="wide",
)

RISK_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Unknown": "⚪"}


# ---------------------------------------------------------
# Sidebar: roadmap + document management
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚖️ Compliance System")
    st.caption("Automated Enterprise Legal & Compliance Workflow System")

    st.markdown("### Project roadmap")
    st.markdown(
        """
        - ✅ **Phase 1** — Base agent foundation
        - ✅ **Phase 2** — RAG: policy PDF ingestion + vector search
        - ✅ **Phase 3** — Multi-agent orchestration (LangGraph)
        - ⏳ **Phase 4** — Long-term memory & state checkpointing
        - ⏳ **Phase 5** — FastAPI REST layer + production UI
        """
    )
    st.divider()

    st.markdown("### 📄 Knowledge Base Documents")
    docs = [f for f in os.listdir("documents") if f.lower().endswith(".pdf")] if os.path.isdir("documents") else []
    if docs:
        for d in docs:
            st.markdown(f"- {d}")
    else:
        st.warning("No PDFs found in documents/")

    uploaded_file = st.file_uploader("Add a policy PDF", type=["pdf"])
    if uploaded_file is not None:
        os.makedirs("documents", exist_ok=True)
        save_path = os.path.join("documents", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name}")

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Re-indexing documents..."):
            from rag.vector_store import build_vector_store
            build_vector_store()
        st.success("Knowledge base rebuilt.")

    st.divider()
    st.caption("⚠️ General guidance only — not a substitute for legal counsel")


# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------
st.title("Legal & Compliance Multi-Agent System")
st.caption("Manager → Legal Research → Risk Analysis → Drafting (LangGraph)")

if not os.getenv("GROQ_API_KEY"):
    st.error(
        "No API key found. Create a `.env` file (see .env.example) "
        "and add your Groq API key before running this demo."
    )
    st.stop()

if not os.path.isdir("documents") or not [f for f in os.listdir("documents") if f.endswith(".pdf")]:
    st.warning("No policy PDFs found. Upload at least one PDF in the sidebar before asking a question.")

query = st.chat_input(
    "Ask a compliance question or request a draft (e.g. 'Draft a notice about our breach notification protocol')"
)

if query:
    st.chat_message("user").markdown(query)

    # Import here so Streamlit shows the API-key error above before
    # trying to construct agents that need it
    from graph.workflow import run_compliance_workflow

    pipeline_status = st.status("Running multi-agent pipeline...", expanded=True)

    with pipeline_status:
        st.write("🧭 **Manager Agent** — classifying request...")
        result = run_compliance_workflow(query)
        st.write(f"↳ Route selected: `{result['route']}`")

        st.write("📚 **Legal Research Agent** — searching indexed documents...")
        st.write(f"↳ Found {len(result['sources'])} relevant source(s)")

        st.write("⚠️ **Risk Analysis Agent** — assessing compliance risk...")
        st.write(f"↳ Risk Level: {RISK_COLORS.get(result['risk_level'], '⚪')} {result['risk_level']}")

        if result["draft"]:
            st.write("✍️ **Drafting Agent** — preparing formal document...")
            st.write("↳ Draft ready")

    pipeline_status.update(label="Pipeline complete", state="complete", expanded=False)

    # ---------------- Results ----------------
    with st.chat_message("assistant"):
        st.markdown("#### 📚 Research Findings")
        st.markdown(result["research_answer"])

        if result["sources"]:
            with st.expander(f"🔍 Sources ({len(result['sources'])})"):
                for src in result["sources"]:
                    st.markdown(f"- `{src['source']}`, page {src['page']}")

        st.markdown("#### ⚠️ Risk Assessment")
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Level", f"{RISK_COLORS.get(result['risk_level'], '⚪')} {result['risk_level']}")
        col2.markdown(f"**Key Risk Factors**\n\n{result['key_risk_factors']}")
        col3.markdown(f"**Recommended Urgency**\n\n{result['recommended_urgency']}")

        if result["draft"]:
            st.markdown("#### ✍️ Drafted Document")
            st.text_area("Ready-to-use draft", result["draft"], height=250)
            st.download_button(
                "Download draft as .txt",
                data=result["draft"],
                file_name="compliance_draft.txt",
            )

if not query:
    st.info(
        "👋 Try asking something like:\n\n"
        "- What are our obligations under the DPDP Act for storing employee data? *(research only)*\n"
        "- Draft a notice about our new data breach notification protocol *(research + draft)*\n"
        "- What does the POSH Act require for our Internal Committee? *(research only)*"
    )
