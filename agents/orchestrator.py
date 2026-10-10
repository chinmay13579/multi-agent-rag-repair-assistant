"""
OrchestratorAgent — LangChain edition.

Replaces the hand-rolled _tool_loop() with:
  create_openai_tools_agent  — binds sub-agent callables as LangChain tools
  AgentExecutor              — drives the Reason-Act loop

Each sub-agent is exposed as a @tool closure so the orchestrator can
delegate to them using OpenAI-style function calling.

Multi-turn history is maintained manually and injected into the agent
via the "chat_history" MessagesPlaceholder on every call.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Callable, List, Optional

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from base_agent import create_llm
from messages import (
    SearchTask, SearchResult,
    VisualTask, VisualResult,
    SynthesisTask, FinalAnswer,
    AgentCall,
)
import config


SYSTEM_PROMPT = """You are the Orchestrator of a multi-agent system for the Porsche 911 (993) Repair Manual.

Your team of specialised agents:
  • RetrievalAgent  — searches manual text using BM25 (autonomous, has its own reasoning loop).
  • VisualAgent     — analyses a rendered page image with a vision model.
  • SynthesisAgent  — composes the final cited answer from all gathered evidence.

Your tools map directly to these agents:
  call_retrieval_agent(query, rationale)   →  delegates to RetrievalAgent
  call_visual_agent(page_number, focus)    →  delegates to VisualAgent
  finalize(ready)                          →  delegates to SynthesisAgent (call LAST)

Decision framework:
  1. Decompose the user question into information needs.
  2. Call call_retrieval_agent ONCE with a focused query to locate relevant pages.
  3. For ANY question involving diagrams, part numbers, item numbers, exploded views,
     schematics, or tables — call call_visual_agent on the relevant pages immediately
     after retrieval. Always check BOTH the diagram page AND the page immediately after
     it (which usually contains the parts legend/table with numbered designations).
     Example: exhaust system diagram is on page 44, parts table is on page 46 — check both.
  4. Call finalize as soon as you have evidence from retrieval + visual.

STRICT RULES:
  - Call call_retrieval_agent AT MOST 2 times. If the first search returns relevant pages,
    move on immediately — do not keep retrying.
  - For part number / diagram label questions: after retrieval, call call_visual_agent on
    the diagram page AND on page+2 (the parts table page). Both are needed.
  - Do NOT retry retrieval more than once. Pivot to call_visual_agent instead.
  - Call finalize after at most 2 retrievals + 2 visual calls. Do not delay.
  - You are NOT responsible for writing the final answer — that is SynthesisAgent's job."""


class OrchestratorAgent:
    """
    Coordinates RetrievalAgent, VisualAgent, and SynthesisAgent via
    LangChain's AgentExecutor with OpenAI-style tool calling.
    """

    def __init__(
        self,
        retrieval_fn: Callable[[SearchTask], SearchResult],
        visual_fn: Callable[[VisualTask], VisualResult],
        synthesis_fn: Callable[[SynthesisTask], FinalAnswer],
    ):
        self._retrieval_fn = retrieval_fn
        self._visual_fn    = visual_fn
        self._synthesis_fn = synthesis_fn
        self._llm = create_llm()

        # Per-turn state (reset in ask())
        self._search_results: List[SearchResult] = []
        self._visual_results: List[VisualResult]  = []
        self._question: str = ""
        self._final_answer: Optional[FinalAnswer]  = None
        self._trace: List[AgentCall] = []

        # Persistent multi-turn chat history (HumanMessage / AIMessage pairs)
        self.history: List = []

        # Build executor once; tools reference self so they always use current state
        self._executor = self._build_executor()

    # -- Build AgentExecutor -------------------------------------------------

    def _build_executor(self) -> AgentExecutor:
        @tool
        def call_retrieval_agent(query: str, rationale: str = "") -> str:
            """
            Delegate a text search task to the RetrievalAgent.
            The agent autonomously searches the manual using BM25, may refine its
            query, and returns the most relevant passages with page numbers and a summary.
            query: The search topic (technical terms preferred).
            rationale: Why this information is needed to answer the question.
            """
            return self._handle_retrieval(query, rationale)

        @tool
        def call_visual_agent(page_number: int, focus: str) -> str:
            """
            Delegate visual analysis to the VisualAgent.
            The agent renders the specified page as a high-resolution image and uses a
            vision model to describe diagrams, exploded views, tables, and callouts.
            page_number: The 1-indexed page number to analyse.
            focus: What aspect of the page to focus on.
            """
            return self._handle_visual(page_number, focus)

        @tool
        def finalize(ready: bool) -> str:
            """
            Trigger the SynthesisAgent to compose the final answer from all collected
            evidence. Call this only when you have gathered sufficient text and/or
            visual evidence to answer the question.
            ready: Must be true to proceed.
            """
            return self._handle_finalize(ready)

        tools = [call_retrieval_agent, call_visual_agent, finalize]

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self._llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=config.MAX_AGENT_ITERATIONS,
            handle_parsing_errors=True,
            verbose=True,
        )

    # -- Public interface ----------------------------------------------------

    def ask(self, question: str):
        """
        Process a user question through the full multi-agent pipeline.

        Returns
        -------
        answer : str
        trace  : List[AgentCall]
        """
        self._question       = question
        self._search_results = []
        self._visual_results = []
        self._final_answer   = None
        self._trace          = []

        result = self._executor.invoke({
            "input":        question,
            "chat_history": self.history,
        })

        # Persist this turn in history for future turns
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=result.get("output", "")))

        answer = (
            self._final_answer.answer
            if self._final_answer
            else (
                "The orchestrator could not produce a complete answer. "
                "Please try rephrasing your question."
            )
        )
        return answer, self._trace

    def reset(self) -> None:
        """Clear per-session history."""
        self.history = []

    # -- Sub-agent dispatch methods -----------------------------------------

    def _handle_retrieval(self, query: str, rationale: str = "") -> str:
        print(f"    [OrchestratorAgent] → RetrievalAgent: '{query}'")
        task   = SearchTask(query=query, rationale=rationale)
        result = self._retrieval_fn(task)
        self._search_results.append(result)

        self._trace.append(AgentCall(
            agent="RetrievalAgent",
            input_summary=f"query='{query}'",
            output_preview=result.agent_summary[:120],
        ))

        return (
            f"[RetrievalAgent completed]\n"
            f"Final query used: '{result.final_query}'\n"
            f"Pages found: {sorted({c['page'] for c in result.chunks})}\n"
            f"Summary: {result.agent_summary}"
        )

    def _handle_visual(self, page_number: int, focus: str) -> str:
        print(f"    [OrchestratorAgent] → VisualAgent: page {page_number}, focus='{focus}'")
        task   = VisualTask(page_number=page_number, focus=focus)
        result = self._visual_fn(task)
        self._visual_results.append(result)

        preview = result.description[:120].replace("\n", " ")
        self._trace.append(AgentCall(
            agent="VisualAgent",
            input_summary=f"page={page_number}, focus='{focus}'",
            output_preview=preview,
        ))

        return (
            f"[VisualAgent completed — page {page_number}]\n"
            f"{result.description[:600]}"
        )

    def _handle_finalize(self, ready: bool = True) -> str:
        if not ready:
            return "[finalize skipped — continue gathering evidence]"

        print(f"    [OrchestratorAgent] → SynthesisAgent")
        task = SynthesisTask(
            user_question=self._question,
            search_results=self._search_results,
            visual_results=self._visual_results,
        )
        self._final_answer = self._synthesis_fn(task)

        self._trace.append(AgentCall(
            agent="SynthesisAgent",
            input_summary=(
                f"{len(self._search_results)} search result(s), "
                f"{len(self._visual_results)} visual result(s)"
            ),
            output_preview=self._final_answer.answer[:120],
        ))

        return (
            f"[SynthesisAgent completed — answer produced "
            f"({len(self._final_answer.answer)} chars, "
            f"sources: pages {self._final_answer.source_pages})]"
        )
