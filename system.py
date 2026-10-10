"""
MultiAgentSystem — wires all LangChain agents together.

Instantiation order is identical to the original:
  1. PDF is processed (text chunks + page images cached on disk).
  2. BM25Retriever (LangChain-backed) is built from the text chunks.
  3. Leaf agents (RetrievalAgent, VisualAgent, SynthesisAgent) are instantiated.
  4. OrchestratorAgent is injected with references to the leaf agents' .run() methods.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pdf_processor import extract_chunks, render_pages
from retriever import BM25Retriever
from agents.retrieval_agent import RetrievalAgent
from agents.visual_agent import VisualAgent
from agents.synthesis_agent import SynthesisAgent
from agents.orchestrator import OrchestratorAgent


class MultiAgentSystem:
    """
    Entry point for the LangChain-powered multi-agent repair manual assistant.

    Usage
    -----
    sys = MultiAgentSystem(pdf_path="path/to/manual.pdf")
    answer, trace = sys.ask("What is the valve clearance for the intake valves?")
    """

    def __init__(self, pdf_path: str):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Step 1: PDF indexing (cached after first run)
        print("[system] Indexing PDF …")
        chunks      = extract_chunks(pdf_path)
        image_index = render_pages(pdf_path)
        print(f"[system] {len(chunks)} text chunks, {len(image_index)} page images ready.")

        # Step 2: Build LangChain BM25 retrieval index
        retriever = BM25Retriever(chunks)

        # Step 3: Instantiate leaf agents
        self.retrieval_agent = RetrievalAgent(retriever=retriever)
        self.visual_agent    = VisualAgent(image_index=image_index)
        self.synthesis_agent = SynthesisAgent()

        # Step 4: Inject into orchestrator
        self.orchestrator = OrchestratorAgent(
            retrieval_fn=self.retrieval_agent.run,
            visual_fn=self.visual_agent.run,
            synthesis_fn=self.synthesis_agent.run,
        )

        print("[system] All agents ready.\n")

    def ask(self, question: str):
        return self.orchestrator.ask(question)

    def reset(self) -> None:
        self.orchestrator.reset()
