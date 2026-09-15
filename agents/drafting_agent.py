"""
Phase 3 - Drafting Agent
------------------------------
This agent's ONE job: turn research findings + risk analysis into a
ready-to-use formal document (notice, memo, email, or policy excerpt) in
professional business language, for the manager route "research_and_draft".
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class DraftingAgent:
    """Drafts a formal compliance communication from prior agent outputs."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(model=model_name, temperature=0.3)
        self.prompt_template = PromptTemplate(
            input_variables=["query", "research_findings", "risk_summary"],
            template=(
                "You are a Drafting Agent for corporate legal & compliance "
                "communications in an Indian enterprise. Draft a formal, "
                "professional document that fulfils the request below.\n\n"
                "Original Request: {query}\n\n"
                "Research Findings to incorporate:\n{research_findings}\n\n"
                "Risk Context to incorporate:\n{risk_summary}\n\n"
                "Write the document in a formal business tone, with a clear "
                "subject/title, body, and closing line. Do not include "
                "placeholder brackets like [Company Name] unless truly "
                "necessary - keep it as ready-to-use as possible."
            ),
        )

    def draft(self, query: str, research_findings: str, risk_summary: str) -> str:
        prompt = self.prompt_template.format(
            query=query,
            research_findings=research_findings,
            risk_summary=risk_summary,
        )
        response = self.llm.invoke(prompt)
        return response.content
