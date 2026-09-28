"""
Milestone 2 - Multi-Agent Workflow with Real Tool Integration
--------------------------------------------------------------------
Extends the Phase 3 graph with a genuine tool-calling loop:

    START -> Manager (routes)
          -> Tool-Using Research Agent <-> Tools (loops until no more tool calls)
          -> Risk Analysis Agent
          -> [conditional] -> Drafting Agent (only if route == research_and_draft)
          -> END

Beginner notes:
- `tools_condition` is a prebuilt LangGraph helper: it looks at the last
  message and returns "tools" if the LLM asked to call a tool, or "END"
  (well, a special marker) if it gave a final text answer instead. This
  is what creates the agent <-> tools loop.
- Retries: LLM/API calls can fail transiently (rate limits, timeouts). We
  wrap the risky calls in a small retry helper so one flaky network blip
  doesn't kill the whole pipeline.
- Graceful degradation: if a node fails even after retries, we don't
  crash the graph - we record the error in state and let the pipeline
  continue with whatever it has, so the user still gets a partial,
  honest answer instead of a stack trace.
"""

import time
import logging
from typing import TypedDict, List, Dict, Any, Optional, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, AIMessage

from agents.manager_agent import ManagerAgent
from agents.tool_using_research_agent import ToolUsingResearchAgent
from agents.risk_analysis_agent import RiskAnalysisAgent
from agents.drafting_agent import DraftingAgent
from tools import ALL_TOOLS

logger = logging.getLogger("compliance_workflow")
logging.basicConfig(level=logging.INFO)

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.5
MAX_TOOL_ITERATIONS = 6  # hard cap so a confused agent can't loop forever


def with_retries(func, *args, **kwargs):
    """Small retry wrapper for flaky LLM/API calls. Raises the last error if all retries fail."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {func}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_error


# ---------------------------------------------------------
# Shared state
# ---------------------------------------------------------
class ComplianceWorkflowState(TypedDict):
    query: str
    route: str
    messages: Annotated[List[AnyMessage], add_messages]  # tool-calling conversation
    research_answer: str
    risk_level: str
    key_risk_factors: str
    recommended_urgency: str
    draft: Optional[str]
    errors: List[str]


# ---------------------------------------------------------
# Agents (built once, reused across queries)
# ---------------------------------------------------------
_manager = ManagerAgent()
_research_agent = ToolUsingResearchAgent()
_risk_analyst = RiskAnalysisAgent()
_drafter = DraftingAgent()
_tool_node = ToolNode(ALL_TOOLS)


# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------
def manager_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    if not state.get("query") or not state["query"].strip():
        state["errors"] = state.get("errors", []) + ["Empty query received by Manager Agent."]
        state["route"] = "research_only"
        return state
    try:
        state["route"] = with_retries(_manager.route, state["query"])
    except Exception as e:
        logger.exception("Manager Agent failed after retries")
        state["errors"] = state.get("errors", []) + [f"Manager Agent error: {e}"]
        state["route"] = "research_only"  # safe fallback
    if not state.get("messages"):
        state["messages"] = _research_agent.start(state["query"])
    return state


def out_of_scope_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    """Stops unrelated questions before research, tools, risk, or drafting."""
    state["research_answer"] = (
        "This question is outside the scope of the legal and compliance "
        "workflow. Please ask about company policy, contracts, privacy, "
        "regulatory obligations, compliance risk, or related documents."
    )
    state["risk_level"] = "Not assessed"
    state["key_risk_factors"] = "No legal or compliance assessment was performed."
    state["recommended_urgency"] = "No action recommended for this out-of-scope request."
    state["draft"] = None
    return state


def research_agent_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    """Calls the tool-using LLM. May return a tool call OR a final answer."""
    try:
        response = with_retries(_research_agent.invoke, state["messages"])
    except Exception as e:
        logger.exception("Research Agent failed after retries")
        state["errors"] = state.get("errors", []) + [f"Research Agent error: {e}"]
        # Fall back to a plain text message so the graph can still terminate cleanly
        response = AIMessage(content=(
            "I was unable to complete research due to a technical error. "
            "Please consult internal compliance documentation directly."
        ))
    state["messages"] = [response]
    return state


def finalize_research_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    """Runs once the tool-calling loop is done; extracts the final text answer."""
    last_message = state["messages"][-1]
    state["research_answer"] = getattr(last_message, "content", "") or "No answer produced."
    return state


def risk_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    try:
        result = with_retries(_risk_analyst.analyze, state["query"], state["research_answer"])
        state["risk_level"] = result["risk_level"]
        state["key_risk_factors"] = result["key_risk_factors"]
        state["recommended_urgency"] = result["recommended_urgency"]
    except Exception as e:
        logger.exception("Risk Analysis Agent failed after retries")
        state["errors"] = state.get("errors", []) + [f"Risk Analysis Agent error: {e}"]
        state["risk_level"] = "Unknown"
        state["key_risk_factors"] = "Risk analysis unavailable due to a technical error."
        state["recommended_urgency"] = "Manual review recommended."
    return state


def drafting_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    risk_summary = (
        f"Risk Level: {state['risk_level']}. Key factors: {state['key_risk_factors']}. "
        f"Urgency: {state['recommended_urgency']}."
    )
    try:
        state["draft"] = with_retries(_drafter.draft, state["query"], state["research_answer"], risk_summary)
    except Exception as e:
        logger.exception("Drafting Agent failed after retries")
        state["errors"] = state.get("errors", []) + [f"Drafting Agent error: {e}"]
        state["draft"] = None  # graceful degradation - research/risk results are still returned
    return state


# ---------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------
def _after_research_condition(state: ComplianceWorkflowState) -> str:
    """
    Decides whether to call a tool, or move on to risk analysis.
    Also enforces MAX_TOOL_ITERATIONS as a safety cap.
    """
    tool_call_count = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
    if tool_call_count >= MAX_TOOL_ITERATIONS:
        logger.warning("Max tool iterations reached - forcing finalize.")
        return "finalize"

    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "finalize"


def _route_decision(state: ComplianceWorkflowState) -> str:
    if state["route"] == "research_and_draft":
        return "drafting_agent"
    return "end"


def _after_manager_condition(state: ComplianceWorkflowState) -> str:
    if state["route"] == "out_of_scope":
        return "out_of_scope"
    return "research_agent"


# ---------------------------------------------------------
# Build the graph
# ---------------------------------------------------------
def build_workflow():
    graph = StateGraph(ComplianceWorkflowState)

    graph.add_node("manager", manager_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("tools", _tool_node)
    graph.add_node("finalize_research", finalize_research_node)
    graph.add_node("risk_analysis", risk_node)
    graph.add_node("drafting_agent", drafting_node)

    graph.set_entry_point("manager")
    graph.add_conditional_edges(
        "manager",
        _after_manager_condition,
        {"out_of_scope": "out_of_scope", "research_agent": "research_agent"},
    )
    graph.add_edge("out_of_scope", END)

    graph.add_conditional_edges(
        "research_agent",
        _after_research_condition,
        {"tools": "tools", "finalize": "finalize_research"},
    )
    graph.add_edge("tools", "research_agent")  # loop back after a tool runs

    graph.add_edge("finalize_research", "risk_analysis")

    graph.add_conditional_edges(
        "risk_analysis",
        _route_decision,
        {"drafting_agent": "drafting_agent", "end": END},
    )
    graph.add_edge("drafting_agent", END)

    return graph.compile()


_compiled_workflow = None


def get_workflow():
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = build_workflow()
    return _compiled_workflow


def run_compliance_workflow(query: str) -> ComplianceWorkflowState:
    """Main entry point: runs a query through the full multi-agent + tools workflow."""
    workflow = get_workflow()
    initial_state: ComplianceWorkflowState = {
        "query": query,
        "route": "",
        "messages": [],
        "research_answer": "",
        "risk_level": "",
        "key_risk_factors": "",
        "recommended_urgency": "",
        "draft": None,
        "errors": [],
    }
    return workflow.invoke(initial_state)


def stream_compliance_workflow(query: str):
    """
    Generator version used by the FastAPI backend for real-time progress
    logs: yields (node_name, state_snapshot) after each node completes.
    """
    workflow = get_workflow()
    initial_state: ComplianceWorkflowState = {
        "query": query,
        "route": "",
        "messages": [],
        "research_answer": "",
        "risk_level": "",
        "key_risk_factors": "",
        "recommended_urgency": "",
        "draft": None,
        "errors": [],
    }
    for step in workflow.stream(initial_state):
        for node_name, node_state in step.items():
            yield node_name, node_state


if __name__ == "__main__":
    result = run_compliance_workflow(
        "What are our obligations under the DPDP Act for storing employee data?"
    )
    print("Route:", result["route"])
    print("\nResearch Answer:\n", result["research_answer"])
    print("\nRisk Level:", result["risk_level"])
    print("Errors:", result["errors"])
