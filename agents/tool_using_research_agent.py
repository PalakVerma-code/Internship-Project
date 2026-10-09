"""
Milestone 2 - Tool-Using Legal Research Agent
--------------------------------------------------
Replaces the Phase 3 "just retrieve from vector store" research agent with
a real ReAct-style agent that can DECIDE which of the 4 tools to call
(retrieve from documents, search the live web, calculate a risk score, or
verify a company), possibly calling several in sequence, before producing
a final answer.

Beginner notes on "tool calling":
- We give the LLM a list of tool descriptions. When it decides it needs
  one, instead of writing a normal text reply, it writes a structured
  "tool_call" (function name + arguments).
- LangGraph's `ToolNode` is a ready-made node that takes that tool_call,
  actually runs the real Python function, and puts the result back into
  the conversation as a new message.
- The agent then sees that tool result and decides: call another tool, or
  answer the user directly. This can loop several times - that loop is
  what makes this an "agent" rather than a single LLM call.
"""

import os
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage

from tools import ALL_TOOLS

SYSTEM_PROMPT = (
    "You are the Legal Research Agent in an Enterprise Legal & Compliance "
    "Workflow System for an Indian corporate enterprise. You have access "
    "to 4 tools:\n"
    "- retrieve_legal_clauses: search indexed internal policy/legal PDFs\n"
    "- search_regulatory_updates: search the live internet for recent changes\n"
    "- calculate_compliance_risk_score: score contract text algorithmically\n"
    "- verify_corporate_entity: check a company's registration status\n\n"
    "Always try retrieve_legal_clauses first for policy/legal questions. "
    "Only use search_regulatory_updates if the question concerns something "
    "recent that internal documents would not cover. Use "
    "calculate_compliance_risk_score only when actual contract/clause text "
    "is provided. Use verify_corporate_entity only when a specific company "
    "name needs to be checked. Once you have enough information, answer in "
    "proportion to the user's request. For a simple factual question, give "
    "a direct answer in 1-3 short paragraphs or bullets, with only the "
    "source detail needed to support it. Do not turn a simple question into "
    "a research memo, restate the question, or add generic background. For "
    "legal analysis, risk assessment, comparison, or formal documents, give "
    "the fuller structured findings those tasks require. Use source references "
    "only for sources actually used. "
    "If the user asks for a letter, notice, memo, or other document, do "
    "not draft that document here; provide the factual findings for the "
    "separate Drafting Agent. Do not call tools unnecessarily."
)


class ToolUsingResearchAgent:
    """Wraps a Groq LLM bound to the 4 real tools, for use as a LangGraph node."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self.llm = ChatGroq(model=model_name, temperature=0.1, timeout=timeout)
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

    def invoke(self, messages: List[AnyMessage]) -> AnyMessage:
        """
        Takes the current conversation (including any prior tool results)
        and returns the next message - either another tool call, or a
        final text answer.
        """
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        return self.llm_with_tools.invoke(full_messages)

    def start(self, query: str) -> List[AnyMessage]:
        """Builds the initial message list for a new query."""
        return [HumanMessage(content=query)]
