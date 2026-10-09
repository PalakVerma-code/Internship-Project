"""Phase 3 - Risk Analysis Agent"""
import os
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class RiskAnalysisAgent:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self.llm = ChatGroq(model=model_name, temperature=0.1, timeout=timeout)
        self.prompt_template = PromptTemplate(
            input_variables=["query", "research_findings"],
            template=(
                "Based on the query and findings, assess compliance risk.\n\n"
                "Query: {query}\nFindings:\n{research_findings}\n\n"
                "Respond EXACTLY as:\nRisk Level: <Low/Medium/High>\n"
                "Key Risk Factors: <one sentence>\nRecommended Urgency: <one sentence>\n"
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
