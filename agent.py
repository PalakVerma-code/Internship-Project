"""
Milestone 1: Agent Foundation Development
Project: Automated Enterprise Legal & Compliance Workflow System
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

class ComplianceAssistantAgent:
    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.2):
        self.llm = ChatGroq(model=model_name, temperature=temperature)
        self.prompt_template = PromptTemplate(
            input_variables=["request"],
            template=(
                "You are a Legal & Compliance Assistant AI agent for an Indian corporate enterprise.\n\n"
                "Employee/Officer Query: {request}\n\n"
                "Respond with: 1. Understanding 2. Relevant Area 3. Guidance 4. Disclaimer\n"
            ),
        )

    def build_prompt(self, request: str) -> str:
        return self.prompt_template.format(request=request)

    def handle_request(self, request: str) -> str:
        prompt = self.build_prompt(request)
        response = self.llm.invoke(prompt)
        return response.content
