"""
Milestone 2 Test Suite
--------------------------
Covers:
1. Real tool functionality (the deterministic risk calculator, run for real)
2. Tool input validation (empty inputs handled cleanly)
3. Dynamic tool-call execution through the LangGraph ToolNode (mocked LLM
   decision, REAL tool execution)
4. Error fallback: a failing agent degrades gracefully instead of crashing
5. Safety cap: a "confused" agent that never stops calling tools is capped

Run with:
    pytest tests/test_milestone2.py -v

Note: Tests 1-2 exercise the REAL algorithmic tool with no mocking at all.
Tests 3-5 mock the Groq LLM decision-making (since that needs a real API
key + network) but let the actual tool functions execute for real - this
is what "verifying tool call accuracy" means in an automated test: you
can't assert what a live LLM will decide, but you CAN assert that once it
decides, the correct real tool runs and the graph behaves correctly.
"""

import sys
import os
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("GROQ_API_KEY", "fake_key_for_testing")

from langchain_core.messages import AIMessage


# =====================================================================
# 1. Real tool tests (no mocking - genuine function execution)
# =====================================================================
def test_risk_calculator_flags_high_risk_contract():
    from tools import calculate_compliance_risk_score

    text = (
        "The vendor shall indemnify the company against all claims. "
        "This agreement includes an unlimited liability clause involving "
        "personal data and a data breach notification requirement."
    )
    result = calculate_compliance_risk_score.invoke({"contract_text": text})
    assert "Risk Score" in result
    assert "High" in result or "Medium" in result


def test_risk_calculator_reduces_score_for_protective_clauses():
    from tools import calculate_compliance_risk_score

    risky_text = "unlimited liability indemnify penalty"
    protected_text = "governing law arbitration force majeure"

    risky_result = calculate_compliance_risk_score.invoke({"contract_text": risky_text})
    protected_result = calculate_compliance_risk_score.invoke({"contract_text": protected_text})

    def extract_score(text):
        return int(text.split("Risk Score: ")[1].split("/")[0])

    assert extract_score(risky_text and risky_result) > extract_score(protected_text and protected_result)


def test_risk_calculator_rejects_empty_input():
    from tools import calculate_compliance_risk_score

    result = calculate_compliance_risk_score.invoke({"contract_text": ""})
    assert result.startswith("ERROR")


def test_verify_corporate_entity_rejects_empty_input():
    from tools import verify_corporate_entity

    result = verify_corporate_entity.invoke({"company_name": ""})
    assert result.startswith("ERROR")


def test_manager_routes_company_registration_query_to_research():
    from agents.manager_agent import ManagerAgent

    manager = ManagerAgent.__new__(ManagerAgent)
    manager.llm = MagicMock()
    manager.llm.invoke.return_value = MagicMock(content="research_only")
    manager.prompt_template = MagicMock()
    manager.prompt_template.format.return_value = "prompt"

    assert manager.route("Verify whether Tata Consultancy Services is registered in India") == "research_only"


def test_manager_uses_intent_for_natural_language_company_query():
    from agents.manager_agent import ManagerAgent

    manager = ManagerAgent.__new__(ManagerAgent)
    manager.llm = MagicMock()
    manager.llm.invoke.return_value = MagicMock(content="research_only")
    manager.prompt_template = MagicMock()
    manager.prompt_template.format.return_value = "prompt"

    assert manager.route("Can you check if this business is legitimate?") == "research_only"


def test_search_regulatory_updates_degrades_gracefully_on_failure():
    """Even if every search backend fails, the tool must return a string, never raise."""
    from tools import search_regulatory_updates

    with patch("tools._search_with_duckduckgo", side_effect=Exception("network down")):
        result = search_regulatory_updates.invoke({"query": "test query"})
        assert isinstance(result, str)
        assert result.startswith("ERROR")


def test_retrieve_legal_clauses_handles_missing_vector_store():
    from tools import retrieve_legal_clauses

    with patch("tools.get_or_build_vector_store", side_effect=FileNotFoundError("no docs")):
        result = retrieve_legal_clauses.invoke({"query": "test"})
        assert result.startswith("ERROR")


# =====================================================================
# 2. Tool-calling loop through the real LangGraph ToolNode
# =====================================================================
def _patch_all_agent_llms(stack: ExitStack):
    """Patches ChatGroq in every agent module. Returns dict of mock instances."""
    mocks = {}
    for name, path in [
        ("manager", "agents.manager_agent.ChatGroq"),
        ("research", "agents.tool_using_research_agent.ChatGroq"),
        ("risk", "agents.risk_analysis_agent.ChatGroq"),
        ("drafting", "agents.drafting_agent.ChatGroq"),
    ]:
        mock_cls = stack.enter_context(patch(path))
        instance = MagicMock()
        if name == "research":
            instance.bind_tools.return_value = instance
        mock_cls.return_value = instance
        mocks[name] = instance
    return mocks


def test_tool_call_executes_real_tool_and_feeds_result_back():
    """
    Confirms that when the LLM 'decides' to call a tool, the graph runs
    the REAL tool function (not a mock) and the real output is passed
    back into the conversation for the agent to use.
    """
    import importlib
    import graph.workflow as wf

    with ExitStack() as stack:
        mocks = _patch_all_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="research_only")
        mocks["research"].invoke.side_effect = [
            AIMessage(content="", tool_calls=[{
                "name": "calculate_compliance_risk_score",
                "args": {"contract_text": "unlimited liability indemnify"},
                "id": "call_1",
            }]),
            AIMessage(content="This clause is High risk based on the calculated score."),
        ]
        mocks["risk"].invoke.return_value = MagicMock(
            content="Risk Level: High\nKey Risk Factors: liability\nRecommended Urgency: now"
        )

        importlib.reload(wf)
        result = wf.run_compliance_workflow("Assess this clause")

        tool_messages = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
        assert len(tool_messages) == 1
        assert "Risk Score" in tool_messages[0].content  # proves the REAL tool ran
        assert result["errors"] == []


def test_manager_failure_falls_back_without_crashing():
    """A failing Manager Agent (simulated API outage) must not crash the pipeline."""
    import importlib
    import graph.workflow as wf

    with ExitStack() as stack:
        mocks = _patch_all_agent_llms(stack)
        mocks["manager"].invoke.side_effect = Exception("Simulated Groq outage")
        mocks["research"].invoke.return_value = AIMessage(content="Fallback answer")
        mocks["risk"].invoke.return_value = MagicMock(
            content="Risk Level: Unknown\nKey Risk Factors: n/a\nRecommended Urgency: n/a"
        )

        importlib.reload(wf)
        result = wf.run_compliance_workflow("Test query")

        assert result["route"] == "research_only"  # safe fallback, not a crash
        assert any("Manager Agent error" in e for e in result["errors"])


def test_tool_loop_respects_max_iteration_safety_cap():
    """An agent that never stops calling tools must be forcibly capped, not loop forever."""
    import itertools
    import importlib
    import graph.workflow as wf

    with ExitStack() as stack:
        mocks = _patch_all_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="research_only")

        counter = itertools.count()

        def always_call_tool(*args, **kwargs):
            i = next(counter)
            return AIMessage(content="", tool_calls=[{
                "name": "calculate_compliance_risk_score",
                "args": {"contract_text": f"clause {i}"},
                "id": f"call_{i}",
            }])

        mocks["research"].invoke.side_effect = always_call_tool
        mocks["risk"].invoke.return_value = MagicMock(
            content="Risk Level: Medium\nKey Risk Factors: x\nRecommended Urgency: y"
        )

        importlib.reload(wf)
        result = wf.run_compliance_workflow("test loop protection")

        tool_call_msgs = [m for m in result["messages"] if getattr(m, "tool_calls", None)]
        assert len(tool_call_msgs) == wf.MAX_TOOL_ITERATIONS


if __name__ == "__main__":
    test_risk_calculator_flags_high_risk_contract()
    test_risk_calculator_reduces_score_for_protective_clauses()
    test_risk_calculator_rejects_empty_input()
    test_verify_corporate_entity_rejects_empty_input()
    test_search_regulatory_updates_degrades_gracefully_on_failure()
    test_retrieve_legal_clauses_handles_missing_vector_store()
    test_tool_call_executes_real_tool_and_feeds_result_back()
    test_manager_failure_falls_back_without_crashing()
    test_tool_loop_respects_max_iteration_safety_cap()
    print("All Milestone 2 tests passed.")
