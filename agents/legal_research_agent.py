"""
Phase 3 - Legal RAG Research Agent
--------------------------------------
This agent's ONE job: search the indexed policy/legal PDFs (via the Phase 2
vector store) and answer the query using ONLY what it actually finds there,
citing sources. This is what makes answers grounded in your real documents
instead of the LLM just guessing.
"""

from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from rag.vector_store import get_or_build_vector_store


class LegalResearchAgent:
    """Retrieves relevant document chunks and synthesizes a grounded answer."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b", top_k: int = 4):
        self.llm = ChatGroq(model=model_name, temperature=0.1)
        self.top_k = top_k
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "You are a Legal Research Agent. Answer the query using "
                "ONLY the context below, which is extracted from real "
                "corporate policy and legal framework documents. If the "
                "context does not contain enough information, say so "
                "clearly instead of guessing.\n\n"
                "Context:\n{context}\n\n"
                "Query: {query}\n\n"
                "Answer in proportion to the query. For a simple question, "
                "give a direct answer in no more than 3 short paragraphs or "
                "bullets and include only the necessary source reference. "
                "Do not restate the question, add generic background, or "
                "turn a simple answer into a research memo. For detailed "
                "legal analysis, provide a fuller structured answer grounded "
                "in the context above."
            ),
        )

    def _format_context(self, chunks: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(chunks, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[Source {i}: {source}, page {page}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def research(self, query: str) -> Dict[str, Any]:
        """
        Returns a dict with:
          - answer: the synthesized, grounded answer
          - sources: list of {source, page} the answer was based on
        """
        vector_store = get_or_build_vector_store()
        chunks = vector_store.similarity_search(query, k=self.top_k)

        if not chunks:
            return {
                "answer": "No relevant documents were found in the knowledge base for this query.",
                "sources": [],
            }

        context = self._format_context(chunks)
        prompt = self.prompt_template.format(query=query, context=context)
        response = self.llm.invoke(prompt)

        sources = [
            {"source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", "?")}
            for doc in chunks
        ]

        return {"answer": response.content, "sources": sources}
