"""Command-line interface for hoodaAgents."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Sequence

from dotenv import load_dotenv

from hooda_agents.agent import AgentError, AgentResult, build_agent
from hooda_agents.client import OllamaError
from hooda_agents.config import Settings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local-first hoodaAgents Ollama agent"
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="optional one-shot prompt; omit it to start the interactive shell",
    )
    parser.add_argument("--model", help="override OLLAMA_MODEL")
    parser.add_argument("--session", default="default", help="local memory session")
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="disable local conversation persistence",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="display model thinking traces when the model provides them",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete one-shot result as JSON",
    )
    return parser.parse_args(argv)


def _display(result: AgentResult, *, show_thinking: bool) -> None:
    if show_thinking and result.thinking:
        print("\nThinking:")
        for thought in result.thinking:
            print(thought)
    if result.tool_events:
        called = ", ".join(event.name for event in result.tool_events)
        print(f"Tools: {called}")
    print(result.text)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    try:
        settings = Settings.from_env()
        settings = replace(
            settings,
            model=args.model or settings.model,
            memory_enabled=settings.memory_enabled and not args.no_memory,
        )
        agent = build_agent(settings)
    except (ValueError, OSError) as exc:
        print(f"Configuration error: {exc}")
        return 2

    one_shot_prompt = " ".join(args.prompt).strip()
    if one_shot_prompt:
        try:
            result = agent.run(one_shot_prompt, session_id=args.session)
        except (AgentError, OllamaError, ValueError) as exc:
            print(f"Agent error: {exc}")
            return 1
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
        else:
            _display(result, show_thinking=args.show_thinking)
        return 0

    print(
        f"hoodaAgents | model={settings.model} | session={args.session}\n"
        "Commands: /clear, /tools, /exit"
    )
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            print("Goodbye.")
            return 0
        if prompt == "/clear":
            agent.clear_memory(args.session)
            print("Local session memory cleared.")
            continue
        if prompt == "/tools":
            print(", ".join(agent.tools.names()) or "No tools enabled")
            continue

        try:
            result = agent.run(prompt, session_id=args.session)
            print("\nAgent: ", end="")
            _display(result, show_thinking=args.show_thinking)
        except (AgentError, OllamaError, ValueError) as exc:
            print(f"Agent error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
