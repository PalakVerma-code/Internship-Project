"""Phase 3 - Drafting Agent"""
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class DraftingAgent:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(model=model_name, temperature=0.3)
        self.prompt_template = PromptTemplate(
            input_variables=["query", "research_findings", "risk_summary"],
            template=(
                "Draft a formal compliance document for an Indian enterprise.\n\n"
                "Request: {query}\nFindings:\n{research_findings}\nRisk Context:\n{risk_summary}\n\n"
                "Write in formal business tone with subject, body, and closing."
            ),
        )

    def draft(self, query: str, research_findings: str, risk_summary: str) -> str:
        prompt = self.prompt_template.format(query=query, research_findings=research_findings, risk_summary=risk_summary)
        response = self.llm.invoke(prompt)
        return response.content
