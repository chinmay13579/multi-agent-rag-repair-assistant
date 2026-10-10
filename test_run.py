"""
Quick smoke-test: indexes test.pdf, asks one question, prints the answer.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import MultiAgentSystem

PDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../repair_assistant_v2/repair_assistant/test.pdf",
)

if __name__ == "__main__":
    print("=== LangChain Repair Assistant — Smoke Test ===\n")
    system = MultiAgentSystem(pdf_path=PDF_PATH)

    questions = [
        "What is this document about? Give a brief overview.",
        "What are the key procedures or specifications mentioned?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        print("-" * 60)
        answer, trace = system.ask(q)

        print("Agent trace:")
        for step in trace:
            print(f"  {step.agent}({step.input_summary})")
            print(f"    → {step.output_preview[:100]} …")

        print(f"\nAnswer:\n{answer}\n")
        print("=" * 60)
