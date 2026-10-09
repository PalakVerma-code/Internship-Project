"""Milestone 3 tests for the current Supabase-backed memory design."""

import importlib
import os
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("GROQ_API_KEY", "fake_key_for_testing")

from langchain_core.messages import AIMessage


class FakeSupabase:
    """Small deterministic Supabase substitute for unit tests."""

    def __init__(self):
        self.rows = []

    def table(self, name):
        assert name == "audit_logs"
        return FakeAuditQuery(self)


class FakeAuditQuery:
    def __init__(self, client):
        self.client = client
        self.payload = None
        self.search_term = ""
        self.max_rows = None
        self.newest_first = False

    def insert(self, payload):
        self.payload = payload
        return self

    def select(self, _columns):
        return self

    def or_(self, expression):
        self.search_term = expression.split("ilike.%", 1)[-1].split("%", 1)[0].lower()
        return self

    def order(self, _column, desc=False):
        self.newest_first = desc
        return self

    def limit(self, value):
        self.max_rows = value
        return self

    def execute(self):
        if self.payload is not None:
            row = {"id": str(len(self.client.rows) + 1), **self.payload}
            self.client.rows.append(row)
            return MagicMock(data=[row])

        rows = list(self.client.rows)
        if self.search_term:
            rows = [
                row for row in rows
                if self.search_term in row["query"].lower()
                or self.search_term in row["research_summary"].lower()
            ]
        if self.newest_first:
            rows.reverse()
        if self.max_rows is not None:
            rows = rows[: self.max_rows]
        return MagicMock(data=rows)


def _patch_agent_llms(stack):
    mocks = {}
    for name, path in [
        ("manager", "agents.manager_agent.ChatGroq"),
        ("research", "agents.tool_using_research_agent.ChatGroq"),
        ("risk", "agents.risk_analysis_agent.ChatGroq"),
        ("drafting", "agents.drafting_agent.ChatGroq"),
    ]:
        mock_class = stack.enter_context(patch(path))
        instance = MagicMock()
        if name == "research":
            instance.bind_tools.return_value = instance
        mock_class.return_value = instance
        mocks[name] = instance
    return mocks


def _reload_workflow(stack):
    import memory.checkpointer as checkpointer
    import graph.workflow as workflow

    checkpointer._checkpointer = None
    checkpointer._sqlite_conn = None
    workflow = importlib.reload(workflow)
    stack.enter_context(patch.object(workflow, "log_decision_outcome", return_value="1"))
    return workflow


def _patch_message_store(stack):
    messages = {}

    def save_message(thread_id, role, content):
        messages.setdefault(thread_id, []).append({"role": role, "content": content})

    def get_messages(thread_id):
        return messages.get(thread_id, [])

    stack.enter_context(patch("graph.workflow.save_message", side_effect=save_message))
    stack.enter_context(patch("graph.workflow.get_messages", side_effect=get_messages))


def test_out_of_scope_route_skips_research_and_drafting():
    with ExitStack() as stack:
        mocks = _patch_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="out_of_scope")
        workflow = _reload_workflow(stack)

        result = workflow.run_compliance_workflow("What is a good biryani recipe?")

        assert result["route"] == "out_of_scope"
        assert result["draft"] is None
        mocks["research"].invoke.assert_not_called()
        mocks["risk"].invoke.assert_not_called()


def test_out_of_scope_route_is_audited():
    with ExitStack() as stack:
        mocks = _patch_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="out_of_scope")
        workflow = _reload_workflow(stack)
        audit = workflow.log_decision_outcome

        workflow.run_compliance_workflow("What is a good biryani recipe?")

        audit.assert_called_once()
        assert audit.call_args.kwargs["route"] == "out_of_scope"


def test_same_thread_id_accumulates_conversation_history():
    with ExitStack() as stack:
        mocks = _patch_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="research_only")
        mocks["risk"].invoke.return_value = MagicMock(
            content="Risk Level: Low\nKey Risk Factors: x\nRecommended Urgency: y"
        )
        mocks["research"].invoke.side_effect = [
            AIMessage(content="First answer."),
            AIMessage(content="Second answer."),
        ]
        workflow = _reload_workflow(stack)
        _patch_message_store(stack)

        workflow.run_compliance_workflow("First question", thread_id="thread-memory-A")
        result = workflow.run_compliance_workflow("Second question", thread_id="thread-memory-A")

        human_messages = [
            message.content for message in result["messages"]
            if getattr(message, "type", "") == "human"
        ]
        assert human_messages == ["First question", "Second question"]


def test_different_thread_id_starts_fresh():
    with ExitStack() as stack:
        mocks = _patch_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="research_only")
        mocks["risk"].invoke.return_value = MagicMock(
            content="Risk Level: Low\nKey Risk Factors: x\nRecommended Urgency: y"
        )
        mocks["research"].invoke.return_value = AIMessage(content="Answer.")
        workflow = _reload_workflow(stack)

        workflow.run_compliance_workflow("Question A", thread_id="thread-isolation-A")
        result = workflow.run_compliance_workflow("Question B", thread_id="thread-isolation-B")

        human_messages = [
            message.content for message in result["messages"]
            if getattr(message, "type", "") == "human"
        ]
        assert human_messages == ["Question B"]


def test_supabase_audit_log_search_and_recent_decisions():
    fake_supabase = FakeSupabase()
    with patch("memory.long_term_memory.supabase", fake_supabase):
        import memory.long_term_memory as memory
        importlib.reload(memory)
        memory.supabase = fake_supabase

        memory.log_decision_outcome(
            thread_id="t1", query="DPDP breach notification rules", route="research_only",
            risk_level="High", key_risk_factors="Short window", recommended_urgency="Immediate",
            research_answer="Notify the authority promptly.", had_draft=False,
        )
        memory.log_decision_outcome(
            thread_id="t2", query="Draft a leave policy notice", route="research_and_draft",
            risk_level="Low", key_risk_factors="Standard HR", recommended_urgency="Routine",
            research_answer="Use the approved leave policy.", had_draft=True,
        )

        results = memory.search_past_decisions("breach notification", k=2)
        recent = memory.list_recent_decisions(limit=10)

        assert len(results) == 1
        assert results[0]["risk_level"] == "High"
        assert len(recent) == 2


def test_search_audit_history_tool_uses_supabase_memory():
    fake_supabase = FakeSupabase()
    with patch("memory.long_term_memory.supabase", fake_supabase):
        import memory.long_term_memory as memory
        importlib.reload(memory)
        memory.supabase = fake_supabase
        memory.log_decision_outcome(
            thread_id="t1", query="POSH Internal Committee requirements", route="research_only",
            risk_level="Medium", key_risk_factors="Committee composition",
            recommended_urgency="This quarter", research_answer="Needs a Presiding Officer.",
            had_draft=False,
        )

        from tools import search_audit_history
        result = search_audit_history.invoke({"query": "Internal Committee"})

        assert "POSH" in result or "Committee" in result
        assert not result.startswith("ERROR")
