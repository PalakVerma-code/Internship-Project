"""
Milestone 3 - Multi-Agent Workflow with Tools, Coordination & Memory
--------------------------------------------------------------------------
Builds on the Milestone 2 tool-calling graph by adding:

  1. Short-term memory: the graph is compiled WITH a checkpointer, so
     passing the same thread_id across multiple run_compliance_workflow()
     calls gives every agent the full prior conversation, not just the
     latest message.
  2. Long-term memory: a log_memory_node persists every completed run
     (including out-of-scope ones) into the audit-log vector store in
     memory/long_term_memory.py, and agents can search that history via
     the search_audit_history tool in tools.py.
  3. Agent coordination: the same 4 named agents from Milestone 1/2 -
     Manager Agent, Tool-Using Research Agent, Risk Analysis Agent,
     Drafting Agent - now cover the 4 core business roles this milestone
     asks for (Planning, Research, Analysis, Decision/Drafting). See the
     "Agent role mapping" note below.

    START -> Manager (routes: research_only / research_and_draft / out_of_scope)
          -> [out_of_scope] --------------------------------------------+
          -> Tool-Using Research Agent <-> Tools (loops until finalized) |
          -> Risk Analysis Agent                                        |
          -> [conditional] -> Drafting Agent (only if research_and_draft)|
          -> log_memory_node <-----------------------------------------+
          -> END

Agent role mapping (Milestone 3 "4 core business roles" requirement):
    Planning          -> ManagerAgent               (agents/manager_agent.py)
    Research           -> ToolUsingResearchAgent     (agents/tool_using_research_agent.py)
    Analysis           -> RiskAnalysisAgent          (agents/risk_analysis_agent.py)
    Decision/Drafting  -> DraftingAgent              (agents/drafting_agent.py)
These are the EXACT existing agent classes - nothing was renamed. The
Manager Agent already does the "planning" job (deciding what workflow path
this request needs), so Milestone 3 extends it and the graph around it
rather than introducing a redundant 5th "PlanningAgent" class.

Beginner notes:
- `tools_condition` is a prebuilt LangGraph helper: it looks at the last
  message and returns "tools" if the LLM asked to call a tool, or a
  final-answer marker otherwise. This is what creates the agent <-> tools
  loop.
- Retries: LLM/API calls can fail transiently (rate limits, timeouts). We
  wrap the risky calls in a small retry helper so one flaky network blip
  doesn't kill the whole pipeline.
- Graceful degradation: if a node fails even after retries, we don't
  crash the graph - we record the error in state and let the pipeline
  continue with whatever it has, so the user still gets a partial,
  honest answer instead of a stack trace.
"""

import time
import uuid
import logging
from typing import TypedDict, List, Dict, Any, Optional, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage

from agents.manager_agent import ManagerAgent
from agents.tool_using_research_agent import ToolUsingResearchAgent
from agents.risk_analysis_agent import RiskAnalysisAgent
from agents.drafting_agent import DraftingAgent
from tools import ALL_TOOLS
from memory.checkpointer import get_checkpointer, get_thread_config, save_message, get_messages
from memory.long_term_memory import log_decision_outcome

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
    thread_id: str
    route: str
    messages: Annotated[List[AnyMessage], add_messages]  # tool-calling + multi-turn conversation
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
    answer = getattr(last_message, "content", "") or ""
    if not answer.strip():
        try:
            recovery_prompt = SystemMessage(content=(
                "Produce the final answer now. Use the available tool results "
                "in the conversation, answer the user's question directly, "
                "and keep a simple question to 1-3 concise paragraphs or bullets. "
                "Do not call another tool and do not return an empty response."
            ))
            recovered = with_retries(
                _research_agent.llm.invoke,
                [recovery_prompt] + state["messages"],
            )
            answer = getattr(recovered, "content", "") or ""
        except Exception as e:
            logger.exception("Research answer recovery failed")
            state["errors"] = state.get("errors", []) + [f"Research answer recovery error: {e}"]

    state["research_answer"] = answer.strip() or (
        "The available compliance sources did not return a usable answer. "
        "Please try the question again or add a relevant document to the knowledge base."
    )
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


def log_memory_node(state: ComplianceWorkflowState) -> ComplianceWorkflowState:
    """
    Milestone 3 - Long-Term Memory: persists this run's outcome into the
    audit-log vector store (memory/long_term_memory.py), regardless of
    which path the run took (research-only, drafted, or out-of-scope) -
    every completed run becomes part of the searchable audit history.
    A logging failure is recorded as a soft error; it never blocks the
    user from getting their actual result.
    """
    try:
        log_decision_outcome(
            thread_id=state.get("thread_id", "unknown-thread"),
            query=state.get("query", ""),
            route=state.get("route", ""),
            risk_level=state.get("risk_level", ""),
            key_risk_factors=state.get("key_risk_factors", ""),
            recommended_urgency=state.get("recommended_urgency", ""),
            research_answer=state.get("research_answer", ""),
            had_draft=bool(state.get("draft")),
        )
    except Exception as e:
        logger.exception("Failed to write audit log entry")
        state["errors"] = state.get("errors", []) + [f"Audit logging error: {e}"]
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
    return "log_memory"


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
    graph.add_node("log_memory", log_memory_node)

    graph.set_entry_point("manager")
    graph.add_conditional_edges(
        "manager",
        _after_manager_condition,
        {"out_of_scope": "out_of_scope", "research_agent": "research_agent"},
    )
    graph.add_edge("out_of_scope", "log_memory")

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
        {"drafting_agent": "drafting_agent", "log_memory": "log_memory"},
    )
    graph.add_edge("drafting_agent", "log_memory")
    graph.add_edge("log_memory", END)

    # Milestone 3: compile WITH a checkpointer so thread_id-scoped runs
    # retain multi-turn conversational memory (short-term memory).
    return graph.compile(checkpointer=get_checkpointer())


_compiled_workflow = None


def get_workflow():
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = build_workflow()
    return _compiled_workflow


def _build_initial_state(query: str, thread_id: str) -> ComplianceWorkflowState:
    return {
        "query": query,
        "thread_id": thread_id,
        "route": "",
        "messages": [HumanMessage(content=query)],
        "research_answer": "",
        "risk_level": "",
        "key_risk_factors": "",
        "recommended_urgency": "",
        "draft": None,
        "errors": [],
    }


def run_compliance_workflow(query: str, thread_id: Optional[str] = None) -> ComplianceWorkflowState:
    """
    Main entry point: runs a query through the full multi-agent + tools +
    memory workflow.

    thread_id: pass the SAME thread_id across multiple calls to give the
    agents memory of the earlier turns in that conversation (short-term
    memory). Omit it (or pass None) for a one-off, isolated query - a
    fresh thread_id is generated automatically, matching the old
    single-turn behavior exactly.
    """
    workflow = get_workflow()
    resolved_thread_id = thread_id or str(uuid.uuid4())
    save_message(resolved_thread_id, "human", query)
    initial_state = _build_initial_state(query, resolved_thread_id)
    config = get_thread_config(resolved_thread_id)
    result = workflow.invoke(initial_state, config=config)
    save_message(resolved_thread_id, "ai", _conversation_response(result))
    return result


def stream_compliance_workflow(query: str, thread_id: Optional[str] = None):
    """
    Generator version used by the FastAPI backend for real-time progress
    logs: yields (node_name, state_snapshot) after each node completes.
    Same thread_id semantics as run_compliance_workflow().
    """
    workflow = get_workflow()
    resolved_thread_id = thread_id or str(uuid.uuid4())
    save_message(resolved_thread_id, "human", query)
    initial_state = _build_initial_state(query, resolved_thread_id)
    config = get_thread_config(resolved_thread_id)

    for step in workflow.stream(initial_state, config=config):
        for node_name, node_state in step.items():
            yield node_name, node_state

    final_state = workflow.get_state(config).values
    save_message(resolved_thread_id, "ai", _conversation_response(final_state))


def _conversation_response(state: Dict[str, Any]) -> str:
    """Create the durable assistant message shown when a saved thread is reopened."""
    sections = []
    if state.get("research_answer"):
        sections.append(f"Research findings:\n{state['research_answer']}")
    if state.get("risk_level"):
        sections.append(
            "Risk assessment:\n"
            f"Level: {state.get('risk_level', '')}\n"
            f"Key factors: {state.get('key_risk_factors', '')}\n"
            f"Urgency: {state.get('recommended_urgency', '')}"
        )
    if state.get("draft"):
        sections.append(f"Drafted document:\n{state['draft']}")
    return "\n\n".join(sections) or "No response was produced."


def get_thread_history(thread_id: str) -> List[Dict[str, Any]]:
    """
    Milestone 3 - reads back the short-term conversational memory for a
    thread: every human/AI message exchanged so far, for display in the
    dashboard's conversation panel.
    """
    stored_messages = get_messages(thread_id)
    if stored_messages:
        return [
            {"role": message.get("role", ""), "content": message.get("content", "")}
            for message in stored_messages
            if message.get("role") in ("human", "ai") and message.get("content")
        ]

    workflow = get_workflow()
    config = get_thread_config(thread_id)
    snapshot = workflow.get_state(config)
    if not snapshot or not snapshot.values:
        return []

    messages = snapshot.values.get("messages", [])
    history = []
    for m in messages:
        role = getattr(m, "type", "unknown")
        content = getattr(m, "content", "")
        if role in ("human", "ai") and content:
            history.append({"role": role, "content": content})
    return history


if __name__ == "__main__":
    result = run_compliance_workflow(
        "What are our obligations under the DPDP Act for storing employee data?"
    )
    print("Route:", result["route"])
    print("\nResearch Answer:\n", result["research_answer"])
    print("\nRisk Level:", result["risk_level"])
    print("Errors:", result["errors"])
