"""
Inter-agent message protocol.

Every task sent to a sub-agent and every result returned is a typed dataclass.
This makes agent boundaries explicit and the data flow easy to inspect/log.
"""

from dataclasses import dataclass, field
from typing import List
import uuid


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Task messages  (Orchestrator → Sub-agent)
# ---------------------------------------------------------------------------

@dataclass
class SearchTask:
    """Sent to the RetrievalAgent."""
    query: str
    rationale: str = ""   # Why this information is needed — helps the agent focus
    max_results: int = 4


@dataclass
class VisualTask:
    """Sent to the VisualAgent."""
    page_number: int
    focus: str            # What aspect to focus on


@dataclass
class SynthesisTask:
    """Sent to the SynthesisAgent."""
    user_question: str
    search_results: List   # List[SearchResult]
    visual_results: List   # List[VisualResult]


# ---------------------------------------------------------------------------
# Result messages  (Sub-agent → Orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Returned by the RetrievalAgent."""
    original_query: str
    final_query: str          # May differ if the agent reformulated it
    chunks: List[dict]        # [{"page": int, "text": str, "score": float}]
    agent_summary: str        # The agent's own prose summary of what it found


@dataclass
class VisualResult:
    """Returned by the VisualAgent."""
    page_number: int
    description: str          # Detailed visual description from the vision model


@dataclass
class FinalAnswer:
    """Returned by the SynthesisAgent."""
    answer: str
    source_pages: List[int]


# ---------------------------------------------------------------------------
# Execution trace (for CLI display)
# ---------------------------------------------------------------------------

@dataclass
class AgentCall:
    """Records one sub-agent invocation for display."""
    agent: str
    input_summary: str
    output_preview: str       # First ~120 chars of result
    msg_id: str = field(default_factory=_new_id)
