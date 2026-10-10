"""
CLI entry point for the LangChain Multi-Agent Repair Manual System.

Usage:
    python assistant.py [--pdf path/to/manual.pdf]

Commands during the interactive session:
    reset   — clear conversation history
    trace   — show the agent call trace from the last question
    help    — show available commands
    quit    — exit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from system import MultiAgentSystem


BOLD   = "\033[1m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
DIM    = "\033[2m"
RESET  = "\033[0m"

AGENT_COLOURS = {
    "OrchestratorAgent": "\033[35m",
    "RetrievalAgent":    "\033[34m",
    "VisualAgent":       "\033[33m",
    "SynthesisAgent":    "\033[32m",
}


def _print_banner() -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 64}{RESET}")
    print(f"{BOLD}{CYAN}  Porsche 911 (993) Repair Manual — LangChain Multi-Agent{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 64}{RESET}\n")


def _print_agent_trace(trace: list) -> None:
    if not trace:
        return
    print(f"\n{DIM}Agent execution trace:{RESET}")
    for step, call in enumerate(trace, start=1):
        colour = AGENT_COLOURS.get(call.agent, CYAN)
        print(
            f"{DIM}  {step}. {colour}{BOLD}{call.agent}{RESET}"
            f"{DIM}({call.input_summary}){RESET}"
        )
        print(f"{DIM}     → {call.output_preview} …{RESET}")
    print()


def _separator() -> None:
    print(f"{DIM}{'─' * 64}{RESET}")


def run(pdf_path: str) -> None:
    _print_banner()
    print(f"  PDF          : {pdf_path}")
    print(f"  Orchestrator : {config.TEXT_MODEL}")
    print(f"  Vision       : {config.VISION_MODEL}")
    _separator()
    print()

    system = MultiAgentSystem(pdf_path=pdf_path)
    last_trace: list = []

    print(f"{GREEN}Ready!{RESET}  Type a question, or: {BOLD}reset{RESET} / {BOLD}trace{RESET} / {BOLD}help{RESET} / {BOLD}quit{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in {"quit", "exit", "q"}:
            print("Bye!")
            break

        if cmd == "reset":
            system.reset()
            last_trace = []
            print(f"{DIM}[conversation history cleared]{RESET}\n")
            continue

        if cmd == "trace":
            if last_trace:
                _print_agent_trace(last_trace)
            else:
                print(f"{DIM}[no trace available yet]{RESET}\n")
            continue

        if cmd == "help":
            print(
                f"{DIM}"
                "  <question>  — ask anything about the repair manual\n"
                "  reset       — start a new conversation\n"
                "  trace       — re-display the agent call trace from the last question\n"
                "  help        — show this message\n"
                "  quit        — exit\n"
                f"{RESET}"
            )
            continue

        _separator()
        print(f"{DIM}[orchestrator planning …]{RESET}\n")

        answer, trace = system.ask(user_input)
        last_trace = trace

        _print_agent_trace(trace)

        print(f"{BOLD}{CYAN}Answer:{RESET}")
        print(answer)
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Porsche 911 (993) Repair Manual — LangChain Multi-Agent Assistant"
    )
    parser.add_argument(
        "--pdf",
        default=config.PDF_PATH,
        help="Path to the repair manual PDF",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(pdf_path=args.pdf)
