"""
Milestone 3 Test Suite
--------------------------
Covers:
1. Agent coordination: Manager Agent's out_of_scope route correctly skips
   research/risk/drafting and still logs an audit entry
2. Short-term memory: the SAME thread_id across two calls accumulates
   conversation history (agents see prior turns); a DIFFERENT thread_id
   starts fresh
3. Long-term memory: log_decision_outcome / search_past_decisions /
   list_recent_decisions work against a real (temp, isolated) ChromaDB
   collection - exactly one audit entry is written per completed run,
   including out-of-scope ones
4. Checkpointer plumbing: get_thread_history() correctly reads back a
   thread's conversation for the dashboard's memory panel

Run with:
    pytest tests/test_milestone3.py -v

Embedding note: these tests patch `get_embedding_model` with a small
deterministic fake (hash-based) embedding function. This is NOT testing
semantic search quality - it exists purely so these tests run instantly,
offline, and reproducibly, without downloading the real
sentence-transformers model or needing network access. The real model is
already exercised by rag/vector_store.py in normal use.

Isolation note: each test gets its OWN brand-new temp directory via
tempfile.mkdtemp() rather than reusing one fixed path with rmtree between
tests. ChromaDB can cache a client handle keyed by path; deleting and
recreating the SAME path within one process risks a stale handle pointing
at now-gone files (this produced a real "attempt to write a readonly
database" error during development). A fresh path per test avoids it
entirely.
"""

import sys
import os
import shutil
import hashlib
import tempfile
import importlib
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("GROQ_API_KEY", "fake_key_for_testing")

from langchain_core.messages import AIMessage

TEST_AUDIT_DIR_PREFIX = "test_milestone3_audit_"


def _fresh_audit_dir():
    return tempfile.mkdtemp(prefix=TEST_AUDIT_DIR_PREFIX)


class FakeEmbeddings:
    """Deterministic, offline, hash-based embeddings - see module docstring."""

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        words = text.lower().split()
        vec = [0.0] * 32
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            vec[h % 32] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


def _patch_all_agent_llms(stack: ExitStack):
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


def _reload_workflow_with_fake_embeddings(stack: ExitStack, audit_dir: str):
    """
    Sets up a fresh, isolated long-term memory store AND reloads
    graph/workflow.py so its module-level agent singletons pick up the
    ChatGroq mocks. Order matters here (see test_milestone2.py's
    _patch_audit_logging_after_reload note): the fake embeddings patch
    must be applied BEFORE reload, because long_term_memory's module-level
    get_audit_store() singleton is what actually calls get_embedding_model
    - reload of graph.workflow does NOT re-import long_term_memory's
    internals, only its own top-level names, so this one is safe to patch
    pre-reload (unlike log_decision_outcome in test_milestone2.py).
    """
    import memory.long_term_memory as ltm
    ltm._audit_store = None  # reset the singleton so the new directory takes effect
    ltm.AUDIT_PERSIST_DIRECTORY = audit_dir
    stack.enter_context(patch("memory.long_term_memory.get_embedding_model", return_value=FakeEmbeddings()))

    import graph.workflow as wf
    importlib.reload(wf)
    return wf, ltm


# =====================================================================
# 1. Agent coordination (Manager Agent's out_of_scope routing)
# =====================================================================
def test_out_of_scope_route_skips_research_and_drafting():
    audit_dir = _fresh_audit_dir()
    with ExitStack() as stack:
        mocks = _patch_all_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="out_of_scope")

        wf, ltm = _reload_workflow_with_fake_embeddings(stack, audit_dir)
        result = wf.run_compliance_workflow("What's a good recipe for biryani?")

        assert result["route"] == "out_of_scope"
        assert result["draft"] is None
        mocks["research"].invoke.assert_not_called()
        mocks["risk"].invoke.assert_not_called()
    shutil.rmtree(audit_dir, ignore_errors=True)


def test_out_of_scope_route_still_gets_audit_logged():
    """Milestone 3 requirement: EVERY completed run is part of the audit
    trail, including ones the Manager Agent rejected as out of scope."""
    audit_dir = _fresh_audit_dir()
    with ExitStack() as stack:
        mocks = _patch_all_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="out_of_scope")

        wf, ltm = _reload_workflow_with_fake_embeddings(stack, audit_dir)
        wf.run_compliance_workflow("What's a good recipe for biryani?")

        entries = ltm.list_recent_decisions(limit=10)
        assert len(entries) == 1
        assert entries[0]["route"] == "out_of_scope"
    shutil.rmtree(audit_dir, ignore_errors=True)


# =====================================================================
# 2. Short-term memory (thread-level conversation persistence)
# =====================================================================
def test_same_thread_id_accumulates_conversation_history():
    audit_dir = _fresh_audit_dir()
    with ExitStack() as stack:
        mocks = _patch_all_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="research_only")
        mocks["risk"].invoke.return_value = MagicMock(
            content="Risk Level: Medium\nKey Risk Factors: x\nRecommended Urgency: y"
        )

        wf, ltm = _reload_workflow_with_fake_embeddings(stack, audit_dir)

        mocks["research"].invoke.return_value = AIMessage(content="First answer.")
        result1 = wf.run_compliance_workflow("First question", thread_id="thread-A")
        assert len(result1["messages"]) == 2  # 1 human + 1 ai

        mocks["research"].invoke.return_value = AIMessage(content="Second answer, building on the first.")
        result2 = wf.run_compliance_workflow("Second question", thread_id="thread-A")

        human_msgs = [m.content for m in result2["messages"] if getattr(m, "type", "") == "human"]
        assert human_msgs == ["First question", "Second question"]
    shutil.rmtree(audit_dir, ignore_errors=True)


def test_different_thread_id_starts_fresh_with_no_shared_history():
    audit_dir = _fresh_audit_dir()
    with ExitStack() as stack:
        mocks = _patch_all_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="research_only")
        mocks["risk"].invoke.return_value = MagicMock(
            content="Risk Level: Low\nKey Risk Factors: x\nRecommended Urgency: y"
        )
        mocks["research"].invoke.return_value = AIMessage(content="Answer.")

        wf, ltm = _reload_workflow_with_fake_embeddings(stack, audit_dir)

        wf.run_compliance_workflow("Question in thread B", thread_id="thread-B")
        result = wf.run_compliance_workflow("Question in thread C", thread_id="thread-C")

        human_msgs = [m.content for m in result["messages"] if getattr(m, "type", "") == "human"]
        assert human_msgs == ["Question in thread C"]  # no leakage from thread-B
    shutil.rmtree(audit_dir, ignore_errors=True)


def test_omitting_thread_id_behaves_as_isolated_single_turn():
    """Backward compatibility: callers that don't pass thread_id (like the
    Milestone 2 test suite) must keep getting fresh, isolated runs."""
    audit_dir = _fresh_audit_dir()
    with ExitStack() as stack:
        mocks = _patch_all_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="research_only")
        mocks["risk"].invoke.return_value = MagicMock(
            content="Risk Level: Low\nKey Risk Factors: x\nRecommended Urgency: y"
        )
        mocks["research"].invoke.return_value = AIMessage(content="Answer.")

        wf, ltm = _reload_workflow_with_fake_embeddings(stack, audit_dir)

        result1 = wf.run_compliance_workflow("Solo question 1")
        result2 = wf.run_compliance_workflow("Solo question 2")

        assert len(result1["messages"]) == 2
        assert len(result2["messages"]) == 2  # NOT 4 - no accidental shared thread
    shutil.rmtree(audit_dir, ignore_errors=True)


def test_get_thread_history_reads_back_full_conversation():
    audit_dir = _fresh_audit_dir()
    with ExitStack() as stack:
        mocks = _patch_all_agent_llms(stack)
        mocks["manager"].invoke.return_value = MagicMock(content="research_only")
        mocks["risk"].invoke.return_value = MagicMock(
            content="Risk Level: Low\nKey Risk Factors: x\nRecommended Urgency: y"
        )

        wf, ltm = _reload_workflow_with_fake_embeddings(stack, audit_dir)

        mocks["research"].invoke.return_value = AIMessage(content="Answer A.")
        wf.run_compliance_workflow("Question A", thread_id="thread-D")
        mocks["research"].invoke.return_value = AIMessage(content="Answer B.")
        wf.run_compliance_workflow("Question B", thread_id="thread-D")

        history = wf.get_thread_history("thread-D")
        assert len(history) == 4
        assert history[0]["role"] == "human"
        assert history[0]["content"] == "Question A"
        assert history[-1]["content"] == "Answer B."

        assert wf.get_thread_history("never-used-thread") == []
    shutil.rmtree(audit_dir, ignore_errors=True)


# =====================================================================
# 3. Long-term memory (audit log store, tested directly - no mocked LLMs)
# =====================================================================
def test_log_and_search_past_decisions_real_chromadb():
    audit_dir = _fresh_audit_dir()
    with patch("memory.long_term_memory.get_embedding_model", return_value=FakeEmbeddings()):
        import memory.long_term_memory as ltm
        ltm._audit_store = None
        ltm.AUDIT_PERSIST_DIRECTORY = audit_dir

        ltm.log_decision_outcome(
            thread_id="t1", query="What are DPDP breach notification rules?",
            route="research_only", risk_level="High", key_risk_factors="Short window",
            recommended_urgency="Immediate", research_answer="Notify within 24 hours.",
            had_draft=False,
        )
        ltm.log_decision_outcome(
            thread_id="t2", query="Draft a new leave policy notice",
            route="research_and_draft", risk_level="Low", key_risk_factors="Standard HR",
            recommended_urgency="Routine", research_answer="20 days annual leave.",
            had_draft=True,
        )

        results = ltm.search_past_decisions("data breach notification", k=2)
        assert len(results) == 2
        assert results[0]["risk_level"] == "High"  # closest semantic match ranks first

        recent = ltm.list_recent_decisions(limit=10)
        assert len(recent) == 2
    shutil.rmtree(audit_dir, ignore_errors=True)


def test_search_past_decisions_empty_store_returns_empty_list_not_error():
    audit_dir = _fresh_audit_dir()
    with patch("memory.long_term_memory.get_embedding_model", return_value=FakeEmbeddings()):
        import memory.long_term_memory as ltm
        ltm._audit_store = None
        ltm.AUDIT_PERSIST_DIRECTORY = audit_dir

        results = ltm.search_past_decisions("anything", k=3)
        assert results == []
    shutil.rmtree(audit_dir, ignore_errors=True)


def test_search_audit_history_tool_wraps_long_term_memory():
    """Confirms the 5th tool (tools.py) correctly surfaces past decisions
    to the agent as formatted text."""
    audit_dir = _fresh_audit_dir()
    with patch("memory.long_term_memory.get_embedding_model", return_value=FakeEmbeddings()):
        import memory.long_term_memory as ltm
        ltm._audit_store = None
        ltm.AUDIT_PERSIST_DIRECTORY = audit_dir

        ltm.log_decision_outcome(
            thread_id="t1", query="POSH Internal Committee requirements",
            route="research_only", risk_level="Medium", key_risk_factors="Committee composition",
            recommended_urgency="This quarter", research_answer="Needs a woman Presiding Officer.",
            had_draft=False,
        )

        from tools import search_audit_history
        result = search_audit_history.invoke({"query": "internal committee POSH"})
        assert "POSH" in result or "Committee" in result
        assert not result.startswith("ERROR")
    shutil.rmtree(audit_dir, ignore_errors=True)


if __name__ == "__main__":
    test_out_of_scope_route_skips_research_and_drafting()
    test_out_of_scope_route_still_gets_audit_logged()
    test_same_thread_id_accumulates_conversation_history()
    test_different_thread_id_starts_fresh_with_no_shared_history()
    test_omitting_thread_id_behaves_as_isolated_single_turn()
    test_get_thread_history_reads_back_full_conversation()
    test_log_and_search_past_decisions_real_chromadb()
    test_search_past_decisions_empty_store_returns_empty_list_not_error()
    test_search_audit_history_tool_wraps_long_term_memory()
    print("All Milestone 3 tests passed.")
