"""Run live capability evaluations against an installed Ollama model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hooda_agents import Settings, build_agent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model override (default: OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(__file__).with_name("scenarios.json"),
    )
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=0.75,
        help="required passing fraction from 0 through 1",
    )
    return parser.parse_args(argv)


def evaluate_scenario(agent, scenario):
    result = agent.run(
        scenario["prompt"],
        session_id=f"eval-{scenario['id']}",
    )
    actual_tools = [event.name for event in result.tool_events]
    expected_tools = scenario.get("expected_tools", [])
    required_text = scenario.get("answer_contains", [])

    tools_pass = set(expected_tools).issubset(actual_tools)
    answer_lower = result.text.lower()
    answer_pass = all(
        expected.lower() in answer_lower for expected in required_text
    )
    return {
        "id": scenario["id"],
        "passed": tools_pass and answer_pass,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "answer": result.text,
        "steps": result.steps,
        "tool_errors": [
            event.as_dict()
            for event in result.tool_events
            if event.status == "error"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 <= args.minimum_score <= 1:
        print("--minimum-score must be between 0 and 1", file=sys.stderr)
        return 2

    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    settings = Settings.from_env()
    settings = replace(
        settings,
        model=args.model or settings.model,
        memory_enabled=False,
        temperature=0.0,
    )
    agent = build_agent(settings)

    results = [evaluate_scenario(agent, scenario) for scenario in scenarios]
    passed = sum(result["passed"] for result in results)
    score = passed / len(results) if results else 0.0
    report = {
        "model": settings.model,
        "score": score,
        "passed": passed,
        "total": len(results),
        "results": results,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if score >= args.minimum_score else 1


if __name__ == "__main__":
    raise SystemExit(main())
