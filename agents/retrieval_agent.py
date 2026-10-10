"""
RetrievalAgent — LangChain edition.

Replaces the hand-rolled ReAct loop with LangChain's:
  create_openai_tools_agent  (uses OpenAI-style function calling)
  AgentExecutor              (drives the Reason-Act loop)

The agent still autonomously searches the BM25 index up to 3 times,
refining its query between rounds, and returns a SearchResult.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from base_agent import create_llm
from messages import SearchTask, SearchResult
from retriever import BM25Retriever
import config


SYSTEM_PROMPT = """You are a specialized text retrieval agent for a Porsche 911 (993) repair manual.

Your sole responsibility is to find the most relevant text passages for a given search task.

Tool available:
  bm25_search(query) — keyword search over the manual. Returns ranked passages with page numbers.

Strategy:
  1. Read the search task carefully. Identify the key technical concepts and part names.
  2. Formulate ONE precise BM25 query using specific automotive terminology.
     Good: "cylinder head bolt torque Nm tightening sequence"
     Bad:  "how tight should I make the bolts"
  3. Run the search. If results are completely off-topic, try ONE alternative query.
  4. STOP after at most 2 searches. Do not keep retrying with different phrasings.
  5. Write a concise summary of what you found and on which pages.

IMPORTANT: You may search AT MOST 2 times. After 2 searches, always write your summary
and stop — even if you are not fully satisfied with the results. The visual agent will
handle anything that requires reading diagrams or images."""


class RetrievalAgent:
    """
    Autonomous BM25 retrieval agent powered by LangChain's AgentExecutor.
    """

    def __init__(self, retriever: BM25Retriever):
        self.retriever = retriever
        self._collected_chunks: List[dict] = []
        self._last_query: str = ""
        self._llm = create_llm()
        self._executor = self._build_executor()

    # -- Build the LangChain AgentExecutor -----------------------------------

    def _build_executor(self) -> AgentExecutor:
        # Define the BM25 tool as a closure so it can write to self._*
        @tool
        def bm25_search(query: str) -> str:
            """
            Keyword search over the Porsche 911 (993) repair manual.
            Returns the top matching text passages with page numbers and relevance scores.
            Use specific part names, procedures, and measurements.
            Example: 'piston ring gap clearance specification mm'
            """
            self._last_query = query
            chunks = self.retriever.retrieve(query, top_k=config.TOP_K_CHUNKS)
            self._collected_chunks.extend(chunks)

            if not chunks:
                return "No results found for this query."

            parts = [
                f"[Page {c['page']} | score: {c['score']}]\n{c['text']}"
                for c in chunks
            ]
            return "\n\n---\n\n".join(parts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self._llm, [bm25_search], prompt)
        return AgentExecutor(
            agent=agent,
            tools=[bm25_search],
            max_iterations=10,          # enough for up to 4 searches + thinking + summary
            handle_parsing_errors=True,
            verbose=True,
        )

    # -- Public interface ----------------------------------------------------

    def run(self, task: SearchTask) -> SearchResult:
        """Execute the retrieval task autonomously. Returns a SearchResult."""
        self._collected_chunks = []
        self._last_query = task.query

        user_input = (
            f"Search task: {task.query}\n"
            f"Why this is needed: {task.rationale or 'Answer a user question about the Porsche 993 manual.'}\n\n"
            "Search the manual and provide a summary of the relevant information you find."
        )

        result = self._executor.invoke({"input": user_input})
        summary = result.get("output", "")

        # Deduplicate by chunk_id
        seen: set = set()
        unique_chunks: List[dict] = []
        for chunk in self._collected_chunks:
            cid = chunk.get("chunk_id", id(chunk))
            if cid not in seen:
                seen.add(cid)
                unique_chunks.append(chunk)

        # If the agent hit max iterations without producing a summary,
        # build one from the collected chunks so the orchestrator still gets
        # useful evidence rather than an empty string.
        if not summary or "stopped" in summary.lower():
            if unique_chunks:
                pages = sorted({c["page"] for c in unique_chunks})
                top = unique_chunks[0]
                summary = (
                    f"Retrieved content from pages {pages}. "
                    f"Most relevant (page {top['page']}): {top['text'][:400]}"
                )
            else:
                summary = "No relevant information found for this query."

        return SearchResult(
            original_query=task.query,
            final_query=self._last_query,
            chunks=unique_chunks,
            agent_summary=summary,
        )
