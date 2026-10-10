import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from system import MultiAgentSystem

PDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../repair_assistant_v2/repair_assistant/test.pdf",
)

system = MultiAgentSystem(pdf_path=PDF_PATH)
question = "What should be done while adjusting parking brake?"
print(f"\nQuestion: {question}\n" + "="*60)
answer, trace = system.ask(question)
print("\nAgent trace:")
for step in trace:
    print(f"  {step.agent}({step.input_summary})")
    print(f"    -> {step.output_preview[:120]} ...")
print(f"\nAnswer:\n{answer}")
