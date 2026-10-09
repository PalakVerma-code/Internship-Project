"""Phase 3 - Manager/Router Agent"""
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class ManagerAgent:
    VALID_ROUTES = {"research_only", "research_and_draft", "out_of_scope"}
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self.llm = ChatGroq(model=model_name, temperature=0, timeout=timeout)
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template=(
                "Classify this query into exactly ONE route:\n"
                "- research_only: user wants an explanation, legal answer, issue-spotting, comparison, or analysis\n"
                "- research_and_draft: user wants a document drafted, revised, or reviewed\n"
                "- out_of_scope: clearly unrelated to legal, compliance, policy, risk, privacy, contracts, or business operations\n\n"
                "This is an enterprise legal assistant. Treat questions about contracts, vendors, procurement, "
                "employment, HR, privacy, data protection, cybersecurity, corporate governance, company law, "
                "regulatory obligations, litigation, disputes, intellectual property, finance controls, tax, "
                "audits, policies, due diligence, or business risk as in scope. Route them to research_only "
                "even when the available tools do not have a perfect specialist tool; the research agent should "
                "give a grounded answer, state limitations, and identify when human counsel is needed. "
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
