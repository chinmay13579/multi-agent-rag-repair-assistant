"""
VisualAgent — LangChain edition.

Replaces the raw OpenAI client call with LangChain's ChatOpenAI,
using LangChain message types (SystemMessage / HumanMessage) for the
multimodal vision request.  No tool loop needed — a single focused
vision call suffices.
"""

import base64
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from messages import VisualTask, VisualResult
from pdf_processor import get_page_image_path
import config


SYSTEM_PROMPT = """You are a visual analysis specialist for automotive repair manual pages.

You analyze rendered pages from the Porsche 911 (993) repair manual and transcribe
all technically relevant information EXACTLY as it appears in the image.

When you analyze a page, report:
  • All diagrams and schematics — describe the components and their relationships.
  • Exploded-view drawings — list every labelled part with its callout number.
  • Tables — transcribe all rows and columns EXACTLY (torque specs, clearances, etc.).
  • Arrows, callouts, and annotations — transcribe EXACTLY what they say.
  • Step numbers and sequences — transcribe the steps EXACTLY as written.
  • Any embedded text (headers, captions, footnotes) — copy word for word.

STRICT RULES:
  - Only report what you can directly READ in the image. Do not infer, guess, or fill in gaps.
  - If text is unclear or unreadable, say "unreadable" — do not guess the content.
  - Do not add context, explanations, or information from your own knowledge.
  - If a specific value (torque, measurement, part number) is not clearly visible, do not state it.
  - A technician depends on your accuracy — wrong values can cause serious damage."""


class VisualAgent:
    """
    Vision-model agent that analyses a rendered PDF page using LangChain's ChatOpenAI.
    """

    def __init__(self, image_index: list):
        self.image_index = image_index
        self._llm = ChatOpenAI(
            model=config.VISION_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=0.1,
            max_tokens=1200,
        )

    def run(self, task: VisualTask) -> VisualResult:
        """Analyse the specified page and return a VisualResult."""
        img_path = get_page_image_path(self.image_index, task.page_number)

        if not img_path or not Path(img_path).exists():
            return VisualResult(
                page_number=task.page_number,
                description=f"[VisualAgent] Image for page {task.page_number} is not available.",
            )

        img_b64 = self._encode_image(img_path)

        user_prompt = (
            f"You are analyzing page {task.page_number} of a Porsche 911 (993) repair manual.\n"
            f"Focus particularly on: {task.focus}\n\n"
            "Provide a thorough, technically precise description of all visual content on this page."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "high",
                    },
                },
            ]),
        ]

        # Retry up to 3 times on rate-limit errors
        for attempt in range(3):
            try:
                response = self._llm.invoke(messages)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    raise

        description = (
            response.content.strip()
            if response.content
            else "[VisualAgent] No description returned by vision model."
        )
        return VisualResult(page_number=task.page_number, description=description)

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
