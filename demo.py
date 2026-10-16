#!/usr/bin/env python3
"""Demo: load a .agi file and run the agent through several interactions.

Shows the full lifecycle:
1. Load and validate the .agi package
2. Initialize the kernel with all 7 capability modules
3. Run through several agent loop iterations
4. Print state, action log, and module statistics
"""

import json
import sys

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)

sys.stdout.reconfigure(line_buffering=True)

from runtime.kernel import AGIKernel
from runtime.loader import load_agi


def print_header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step_summary(result: dict) -> None:
    for event in result["events"]:
        phase = event["phase"]
        if phase == "perceive":
            print(f"  [perceive] {event['percepts']} percept(s)")
        elif phase == "understand":
            print(f"  [understand] intent={event['intent']}")
        elif phase == "select_goal":
            print(f"  [goal] {event.get('goal', 'None')}")
        elif phase == "plan":
            print(f"  [plan] {event['steps']} steps")
        elif phase == "decide":
            print(f"  [decide] tool={event['tool']}")
        elif phase == "act":
            status = "ok" if event["success"] else "FAIL"
            print(f"  [act] {event['tool']} -> {status}")
        elif phase == "reward":
            print(f"  [learn] reward={event['total']:.2f}")
        elif phase == "communicate":
            print(f"  [respond] {event['response'][:80]}")
        elif phase == "goal_completed":
            print(f"  [DONE] {event['goal']}")


def main() -> None:
    print_header("AGI Format Demo -- Loading skeleton-agent")

    # 1. Load the .agi package
    package = load_agi("skeleton/")

    print(f"  Loaded: {package.name} v{package.version}")
    print(f"  Format: .agi v{package.format_version}")
    print(f"  Modules: perception, reasoning, memory, language, action, learning, goals")

    # 2. Initialize the kernel
    print_header("Initializing AGI Kernel")
    kernel = AGIKernel(package)

    # 3. Run through interactions
    interactions = [
        "What is the capital of France?",
        "Write a Python function to compute Fibonacci numbers",
        "Analyze the trade-offs between microservices and monoliths",
    ]

    for i, user_input in enumerate(interactions, 1):
        print_header(f"Interaction {i}/{len(interactions)}")
        print(f"  User: {user_input}\n")
        result = kernel.step(user_input)
        print_step_summary(result)

    # 4. Print final state
    print_header("Final Agent State")
    print(f"  Total steps:       {kernel.total_steps}")
    print(f"  Episodic memories: {len(kernel.memory.episodes)}")
    print(f"  Learning updates:  {kernel.learning.update_count}")
    avg = kernel.learning.total_reward / kernel.learning.update_count if kernel.learning.update_count else 0
    print(f"  Avg reward:        {avg:.3f}")
    print(f"  Actions taken:     {len(kernel.action.action_log)}")

    print_header("Action Log")
    for i, entry in enumerate(kernel.action.action_log, 1):
        print(f"  {i}. [{entry['tool']}] {entry['explanation']}")

    print_header("Module Dependency Graph")
    print("  perception ─────────────> memory")
    print("       │                      │")
    print("       └──> reasoning <───────┘")
    print("              │    │")
    print("         language  │")
    print("              │    v")
    print("              └─> action ──> learning")
    print("                    │          │")
    print("                  goals <──────┘")
    print("                    │")
    print("               [alignment]")
    print("               [escalation]")

    print_header("Done")
    print("  The skeleton agent demonstrates the .agi file format's")
    print("  architecture: 7 composable capability modules wired")
    print("  together by a declarative goal loop, with safety")
    print("  constraints enforced at every action boundary.")
    print()
    print("  To make this a real agent, replace the stub modules")
    print("  with actual implementations:")
    print("    - reasoning: LLM + world model + planner")
    print("    - memory: vector DB + knowledge graph")
    print("    - perception: vision/audio encoders")
    print("    - learning: LoRA + replay + EWC")
    print("    - action: sandboxed tool execution")
    print()


if __name__ == "__main__":
    main()
