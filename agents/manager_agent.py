"""Phase 3 - Manager/Router Agent"""
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class ManagerAgent:
    VALID_ROUTES = {"research_only", "research_and_draft", "out_of_scope"}
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(model=model_name, temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template=(
                "Classify this query into exactly ONE route:\n"
                "- research_only: user wants an explanation\n"
                "- research_and_draft: user wants a document drafted\n"
                "- out_of_scope: unrelated to legal, compliance, policy, risk, privacy, contracts, or regulation\n\n"
                "Classify by the meaning and intent of the full sentence, not by exact keywords. "
                "Treat natural-language requests to check a company, vendor, business, registration, "
                "incorporation, or legal status as research_only even when the user does not use formal legal terms.\n"
                "Query: {query}\n\nRespond with ONLY the route name."
            ),
        )

    def route(self, query: str) -> str:
        prompt = self.prompt_template.format(query=query)
        response = self.llm.invoke(prompt)
        route = response.content.strip().lower()
        if route not in self.VALID_ROUTES:
            return "research_only"
        return route
