"""
Phase 3 - Risk Analysis Agent
----------------------------------
This agent's ONE job: take the research findings and evaluate how urgent /
risky the situation is for the company, so the response isn't just
informational but also flags what needs attention.
"""

from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class RiskAnalysisAgent:
    """Assesses compliance risk level based on research findings."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(model=model_name, temperature=0.1)
        self.prompt_template = PromptTemplate(
            input_variables=["query", "research_findings"],
            template=(
                "You are a Risk Analysis Agent for corporate compliance. "
                "Based on the query and research findings below, assess "
                "the compliance risk.\n\n"
                "Query: {query}\n\n"
                "Research Findings:\n{research_findings}\n\n"
                "Respond in EXACTLY this format:\n"
                "Risk Level: <Low/Medium/High>\n"
                "Key Risk Factors: <one short sentence>\n"
                "Recommended Urgency: <one short sentence on timeline>\n"
            ),
        )

    def _parse(self, raw_text: str) -> Dict[str, str]:
        result = {"risk_level": "Unknown", "key_risk_factors": "", "recommended_urgency": ""}
        for line in raw_text.splitlines():
            line = line.strip()
            if line.lower().startswith("risk level:"):
                result["risk_level"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("key risk factors:"):
                result["key_risk_factors"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("recommended urgency:"):
                result["recommended_urgency"] = line.split(":", 1)[1].strip()
        return result

    def analyze(self, query: str, research_findings: str) -> Dict[str, Any]:
        prompt = self.prompt_template.format(query=query, research_findings=research_findings)
        response = self.llm.invoke(prompt)
        parsed = self._parse(response.content)
        parsed["raw"] = response.content
        return parsed
