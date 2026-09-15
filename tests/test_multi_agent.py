"""
Test script for Phase 3: Multi-agent LangGraph workflow.

These tests MOCK the Groq LLM and vector store, so they run instantly,
for free, and without needing a real API key. This lets you (and your
mentor) verify the AGENT LOGIC AND WIRING is correct, separately from
whether the actual LLM gives a good answer.

Run with:
    pytest tests/test_multi_agent.py -v

Mocking note (important if you extend these tests):
Each agent module does `from langchain_groq import ChatGroq` at the top,
which creates its OWN local reference to the class. So we must patch
"agents.manager_agent.ChatGroq" (where it's USED), not
"langchain_groq.ChatGroq" (where it's DEFINED) - patching the definition
site has no effect once a module has already imported its own reference.
"""

import sys
import os
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GROQ_API_KEY", "fake_key_for_testing")


def _make_fake_vector_store(content="The company must notify within 24 hours."):
    fake_doc = MagicMock()
    fake_doc.metadata = {"source": "sample_compliance_policy.pdf", "page": 0}
    fake_doc.page_content = content
    fake_store = MagicMock()
    fake_store.similarity_search.return_value = [fake_doc]
    return fake_store


def test_manager_routes_draft_request_correctly():
    from agents.manager_agent import ManagerAgent

    with patch("agents.manager_agent.ChatGroq") as MockLLM:
        instance = MagicMock()
        instance.invoke.return_value = MagicMock(content="research_and_draft")
        MockLLM.return_value = instance

        manager = ManagerAgent()
        route = manager.route("Draft a notice about our new leave policy")
        assert route == "research_and_draft"


def test_manager_falls_back_safely_on_invalid_llm_output():
    from agents.manager_agent import ManagerAgent

    with patch("agents.manager_agent.ChatGroq") as MockLLM:
        instance = MagicMock()
        instance.invoke.return_value = MagicMock(content="not_a_real_route")
        MockLLM.return_value = instance

        manager = ManagerAgent()
        route = manager.route("Some ambiguous query")
        assert route == "research_only"  # safe default


def test_research_agent_returns_answer_and_sources():
    from agents.legal_research_agent import LegalResearchAgent

    with patch("agents.legal_research_agent.get_or_build_vector_store", return_value=_make_fake_vector_store()):
        with patch("agents.legal_research_agent.ChatGroq") as MockLLM:
            instance = MagicMock()
            instance.invoke.return_value = MagicMock(content="Notification required within 24 hours.")
            MockLLM.return_value = instance

            agent = LegalResearchAgent()
            result = agent.research("What is the breach notification window?")
            assert "24 hours" in result["answer"]
            assert len(result["sources"]) == 1
            assert result["sources"][0]["source"] == "sample_compliance_policy.pdf"


def test_risk_agent_parses_structured_output():
    from agents.risk_analysis_agent import RiskAnalysisAgent

    with patch("agents.risk_analysis_agent.ChatGroq") as MockLLM:
        instance = MagicMock()
        instance.invoke.return_value = MagicMock(
            content="Risk Level: High\nKey Risk Factors: Tight deadline\nRecommended Urgency: Immediate"
        )
        MockLLM.return_value = instance

        agent = RiskAnalysisAgent()
        result = agent.analyze("query", "findings")
        assert result["risk_level"] == "High"
        assert result["key_risk_factors"] == "Tight deadline"


def _reload_workflow_with_mocked_agents(llm_responses):
    """
    Helper: patches ChatGroq in all 4 agent modules simultaneously, then
    reloads graph.workflow so its module-level agent instances are rebuilt
    using the mocks (workflow.py builds its agents once at import time).
    """
    import importlib
    import graph.workflow as wf

    stack = ExitStack()
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = llm_responses

    for module_path in [
        "agents.manager_agent.ChatGroq",
        "agents.legal_research_agent.ChatGroq",
        "agents.risk_analysis_agent.ChatGroq",
        "agents.drafting_agent.ChatGroq",
    ]:
        mock_cls = stack.enter_context(patch(module_path))
        mock_cls.return_value = mock_llm_instance

    importlib.reload(wf)
    return wf, stack


def test_full_workflow_drafts_when_routed_to_draft():
    with patch("agents.legal_research_agent.get_or_build_vector_store", return_value=_make_fake_vector_store()):
        wf, stack = _reload_workflow_with_mocked_agents([
            MagicMock(content="research_and_draft"),  # manager
            MagicMock(content="Research findings here."),  # research
            MagicMock(content="Risk Level: High\nKey Risk Factors: X\nRecommended Urgency: Y"),  # risk
            MagicMock(content="Drafted memo text."),  # drafting
        ])
        try:
            result = wf.run_compliance_workflow("Draft a memo about X")
            assert result["route"] == "research_and_draft"
            assert result["draft"] == "Drafted memo text."
        finally:
            stack.close()


def test_full_workflow_skips_draft_when_research_only():
    with patch("agents.legal_research_agent.get_or_build_vector_store", return_value=_make_fake_vector_store()):
        wf, stack = _reload_workflow_with_mocked_agents([
            MagicMock(content="research_only"),  # manager
            MagicMock(content="Research findings here."),  # research
            MagicMock(content="Risk Level: Low\nKey Risk Factors: X\nRecommended Urgency: Y"),  # risk
            # NOTE: only 3 responses - if drafting agent incorrectly runs,
            # this test will fail with a StopIteration error, proving the
            # conditional edge is broken.
        ])
        try:
            result = wf.run_compliance_workflow("What does X require?")
            assert result["route"] == "research_only"
            assert result["draft"] is None
        finally:
            stack.close()


if __name__ == "__main__":
    test_manager_routes_draft_request_correctly()
    test_manager_falls_back_safely_on_invalid_llm_output()
    test_research_agent_returns_answer_and_sources()
    test_risk_agent_parses_structured_output()
    test_full_workflow_drafts_when_routed_to_draft()
    test_full_workflow_skips_draft_when_research_only()
    print("All multi-agent tests passed.")
