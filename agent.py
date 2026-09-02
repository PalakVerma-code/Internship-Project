import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
 
# ---------------------------------------------------------
# TASK 1: Configure LangChain and dependencies
# ---------------------------------------------------------
load_dotenv()
 
if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "No API key found. Create a .env file (copy .env.example) "
        "and paste your Groq API key into it. Get one free at "
        "https://console.groq.com"
    )
 
 
# ---------------------------------------------------------
# TASK 2 & 3: Foundational agent + prompt template
# ---------------------------------------------------------
class ComplianceAssistantAgent:
    """
    Phase 1 foundational agent for the Enterprise Legal & Compliance
    Workflow System.
 
    This agent currently answers using general legal/compliance reasoning
    from the language model. It does NOT yet reference specific uploaded
    company policy documents or Indian legal texts - that retrieval
    capability (RAG) is added in Phase 2.
    """
 
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
    ):
        # temperature is kept low (0.2) because compliance answers should be
        # consistent and cautious, not "creative"
        self.llm = ChatGroq(model=model_name, temperature=temperature)
 
        # The prompt template fixes the agent's role as a compliance
        # assistant and defines the exact structure every response follows.
        self.prompt_template = PromptTemplate(
            input_variables=["request"],
            template=(
                "You are a Legal & Compliance Assistant AI agent for an "
                "Indian corporate enterprise. Your job is to help employees "
                "and compliance officers understand policy questions, "
                "regulatory obligations, and internal process queries.\n\n"
                "Important: You must give general guidance only. You are "
                "not a licensed advocate, and for binding legal decisions "
                "the company should consult its legal counsel.\n\n"
                "Employee/Officer Query: {request}\n\n"
                "Respond in this structure:\n"
                "1. Understanding: restate what is being asked, in one line\n"
                "2. Relevant Area: name the likely compliance domain "
                "(e.g. Companies Act 2013, POSH Act, Labour Codes, Data "
                "Protection/DPDP Act, GST/Tax compliance, contract law, "
                "etc.)\n"
                "3. Guidance: a clear, practical answer or next step\n"
                "4. Disclaimer: one line reminding the user this is not a "
                "substitute for formal legal advice\n"
            ),
        )
 
    def build_prompt(self, request: str) -> str:
        """Fills the template with the actual user query."""
        return self.prompt_template.format(request=request)
 
    def handle_request(self, request: str) -> str:
        """
        The core interaction workflow:
        user query -> filled prompt -> Groq LLM -> structured response
        """
        prompt = self.build_prompt(request)
        response = self.llm.invoke(prompt)
        return response.content
 
 
# ---------------------------------------------------------
# TASK 4: Basic testing interface (simple command-line loop)
# ---------------------------------------------------------
def run_cli():
    agent = ComplianceAssistantAgent()
 
    print("=" * 65)
    print("Enterprise Legal & Compliance Agent - Milestone 1 Foundation")
    print("Type a compliance/policy question below. Type 'exit' to quit.")
    print("=" * 65)
 
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Ending session. Goodbye!")
            break
        if not user_input:
            continue
 
        response = agent.handle_request(user_input)
        print(f"\nAgent: {response}")
 
 
if __name__ == "__main__":
    run_cli()
 