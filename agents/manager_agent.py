"""
Phase 3 - Manager / Router Agent
------------------------------------
This is the "team lead." It looks at the incoming query and decides which
specialist agents actually need to run.

Beginner notes:
- Not every query needs a formal drafted document. "What does the POSH Act
  require?" just needs research. "Draft a notice about our new data
  retention policy" needs research + a drafted document.
- The Manager keeps the system efficient by skipping steps that aren't
  needed, and it's the one place where routing logic lives (Single
  Responsibility Principle - each agent does ONE job).
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class ManagerAgent:
    """Classifies a query and decides the workflow route."""

    VALID_ROUTES = {"research_only", "research_and_draft"}

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(model=model_name, temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are a routing classifier for a Legal & Compliance "
                "workflow system. Classify the following employee query "
                "into exactly ONE of these two routes:\n\n"
                "- research_only: the user is asking a question and wants "
                "an explanation or clarification\n"
                "- research_and_draft: the user wants a document drafted "
                "(e.g. a notice, policy, email, letter, memo)\n\n"
                "Query: {query}\n\n"
                "Respond with ONLY the route name, nothing else."
            ),
        )

    def route(self, query: str) -> str:
        prompt = self.prompt_template.format(query=query)
        response = self.llm.invoke(prompt)
        route = response.content.strip().lower()

        # Safety net: if the LLM responds with anything unexpected,
        # default to the safer/more complete path.
        if route not in self.VALID_ROUTES:
            return "research_only"
        return route
