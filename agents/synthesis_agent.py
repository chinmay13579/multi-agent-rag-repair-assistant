"""
SynthesisAgent — LangChain edition.

Replaces the raw _chat() call with a LangChain LCEL chain:

    ChatPromptTemplate | ChatOpenAI | StrOutputParser

No tools, no loop — pure reasoning over the collected evidence.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from base_agent import create_llm
from messages import SynthesisTask, FinalAnswer, SearchResult, VisualResult


SYSTEM_PROMPT = """You are a technical synthesis agent for automotive repair information.

You receive a user question and all evidence collected from a Porsche 911 (993) repair
manual by other specialised agents. Your job is to synthesise this evidence into a
single, accurate answer.

Guidelines:
  • Only use information explicitly present in the provided evidence. Never add facts from your own knowledge.
  • Always cite the page number for every claim: e.g. "According to page 14…"
  • Quote exact values directly from the evidence: torque figures, clearances, part numbers.
  • If text and visual evidence conflict, highlight the discrepancy.
  • Structure multi-step procedures as numbered lists.
  • If the evidence does not contain enough information to fully answer the question,
    clearly say so — do not fill gaps with assumptions or general automotive knowledge.

STRICT RULES:
  • If a specific value (torque spec, measurement, part number) is not explicitly stated
    in the evidence, do NOT include it in your answer.
  • If the evidence is vague or unclear on a point, say "not clearly stated in the manual"
    rather than guessing.
  • It is better to give a short accurate answer than a long answer with invented details.
  • Wrong technical values can cause serious engine damage — accuracy is critical."""


class SynthesisAgent:
    """
    Pure-reasoning agent: LCEL chain (prompt | LLM | parser), no tools.
    """

    def __init__(self):
        self._llm = create_llm(max_tokens=1800)
        self._chain = self._build_chain()

    # -- Build LCEL chain ----------------------------------------------------

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        return prompt | self._llm | StrOutputParser()

    # -- Public interface ----------------------------------------------------

    def run(self, task: SynthesisTask) -> FinalAnswer:
        """Synthesise all gathered evidence into a final answer."""
        evidence_block = self._build_evidence_block(
            task.search_results, task.visual_results
        )

        user_input = (
            f"User question:\n{task.user_question}\n\n"
            f"Evidence gathered from the repair manual:\n\n{evidence_block}\n\n"
            "Produce a comprehensive, cited answer to the user's question."
        )

        answer = self._chain.invoke({"input": user_input})

        source_pages = sorted(
            {c["page"] for sr in task.search_results for c in sr.chunks}
            | {vr.page_number for vr in task.visual_results}
        )

        return FinalAnswer(answer=answer, source_pages=source_pages)

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _build_evidence_block(
        search_results: List[SearchResult],
        visual_results: List[VisualResult],
    ) -> str:
        parts: List[str] = []

        for i, sr in enumerate(search_results, start=1):
            header = f"=== Text Evidence #{i} (query: '{sr.final_query}') ==="
            chunks_text = "\n---\n".join(
                f"[Page {c['page']}]\n{c['text']}" for c in sr.chunks
            )
            agent_note = f"\nRetrieval agent summary: {sr.agent_summary}"
            parts.append(f"{header}\n{chunks_text}{agent_note}")

        for vr in visual_results:
            parts.append(
                f"=== Visual Evidence (Page {vr.page_number}) ===\n{vr.description}"
            )

        if not parts:
            return "No evidence was gathered from the manual."

        return "\n\n".join(parts)
