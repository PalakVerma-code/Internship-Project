"""
Phase 3 - Multi-Agent Orchestration (LangGraph)
----------------------------------------------------
This file wires the 4 specialist agents into a single workflow graph.

Beginner notes on LangGraph:
- Think of it like a flowchart. Each "node" is one agent doing its job.
  Each "edge" is an arrow saying "after this agent finishes, go here next."
- "State" is a shared notebook that gets passed from node to node - each
  agent reads what it needs from it and writes its own result into it.
- A "conditional edge" is a fork in the flowchart: "if X, go left; if Y,
  go right" - this is how the Manager Agent's routing decision actually
  changes what happens next.

Flow:
    START -> Manager (decides route)
          -> Legal Research Agent (always runs - grounds everything in real docs)
          -> Risk Analysis Agent (always runs - flags urgency/risk)
          -> [conditional] -> Drafting Agent (only if route == research_and_draft)
          -> END
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from agents.manager_agent import ManagerAgent
from agents.legal_research_agent import LegalResearchAgent
from agents.risk_analysis_agent import RiskAnalysisAgent
from agents.drafting_agent import DraftingAgent


# ---------------------------------------------------------
# Shared state - the "notebook" passed between agents
# ---------------------------------------------------------
class ComplianceWorkflowState(TypedDict):
    query: str
    route: str
    research_answer: str
    sources: List[Dict[str, Any]]
    risk_level: str
    key_risk_factors: str
    recommended_urgency: str
    draft: Optional[str]


# ---------------------------------------------------------
# Instantiate agents once (reused across every query)
# ---------------------------------------------------------
_manager = ManagerAgent()
_researcher = LegalResearchAgent()
_risk_analyst = RiskAnalysisAgent()
_drafter = DraftingAgent()


# ---------------------------------------------------------
# Node functions - each one is a thin wrapper calling its agent
# ---------------------------------------------------------
def manager_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    route = _manager.route(state["query"])
    state["route"] = route
    return state


def research_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    result = _researcher.research(state["query"])
    state["research_answer"] = result["answer"]
    state["sources"] = result["sources"]
    return state


def risk_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    result = _risk_analyst.analyze(state["query"], state["research_answer"])
    state["risk_level"] = result["risk_level"]
    state["key_risk_factors"] = result["key_risk_factors"]
    state["recommended_urgency"] = result["recommended_urgency"]
    return state


def drafting_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    risk_summary = (
        f"Risk Level: {state['risk_level']}. "
        f"Key factors: {state['key_risk_factors']}. "
        f"Urgency: {state['recommended_urgency']}."
    )
    draft = _drafter.draft(state["query"], state["research_answer"], risk_summary)
    state["draft"] = draft
    return state


def _route_decision(state: ComplianceWorkflowState) -> str:
    """Used by the conditional edge to pick the next node after risk analysis."""
    if state["route"] == "research_and_draft":
        return "drafting_agent"
    return "end"


# ---------------------------------------------------------
# Build the graph
# ---------------------------------------------------------
def build_workflow():
    graph = StateGraph(ComplianceWorkflowState)

    graph.add_node("manager", manager_node)
    graph.add_node("legal_research", research_node)
    graph.add_node("risk_analysis", risk_node)
    graph.add_node("drafting_agent", drafting_node)

    graph.set_entry_point("manager")
    graph.add_edge("manager", "legal_research")
    graph.add_edge("legal_research", "risk_analysis")

    graph.add_conditional_edges(
        "risk_analysis",
        _route_decision,
        {"drafting_agent": "drafting_agent", "end": END},
    )
    graph.add_edge("drafting_agent", END)

    return graph.compile()


_compiled_workflow = None


def get_workflow():
    """Builds the graph once and reuses it (avoids rebuilding on every call)."""
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = build_workflow()
    return _compiled_workflow


def run_compliance_workflow(query: str) -> ComplianceWorkflowState:
    """Main entry point: runs a query through the full multi-agent workflow."""
    workflow = get_workflow()
    initial_state: ComplianceWorkflowState = {
        "query": query,
        "route": "",
        "research_answer": "",
        "sources": [],
        "risk_level": "",
        "key_risk_factors": "",
        "recommended_urgency": "",
        "draft": None,
    }
    final_state = workflow.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # Quick manual check: python graph/workflow.py
    result = run_compliance_workflow(
        "What are our obligations under the DPDP Act for storing employee data?"
    )
    print("Route:", result["route"])
    print("\nResearch Answer:\n", result["research_answer"])
    print("\nRisk Level:", result["risk_level"])
    print("Draft:", result["draft"])
