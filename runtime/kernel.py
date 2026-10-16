"""AGI Runtime Kernel -- executes the agent loop defined in a .agi package."""

import structlog

from runtime.loader import AGIPackage
from runtime.modules import (
    ActionModule,
    GoalsModule,
    LanguageModule,
    LearningModule,
    MemoryModule,
    PerceptionModule,
    ReasoningModule,
)
from runtime.types import State

log = structlog.get_logger()


class AGIKernel:
    """The runtime kernel. Loads modules and executes the agent loop.

    This is the 'operating system' for the .agi file. It:
    1. Loads the package and initializes all capability modules
    2. Wires up the data flow between modules
    3. Executes the goal loop as defined in goal_loop.json
    4. Enforces safety constraints at every action boundary
    """

    def __init__(self, package: AGIPackage):
        self.package = package
        self.goal_loop_def = package.load_goal_loop()

        # Initialize all capability modules
        log.info("kernel_initializing", agent=package.name, version=package.version)
        self.perception = PerceptionModule(package)
        self.reasoning = ReasoningModule(package)
        self.memory = MemoryModule(package)
        self.language = LanguageModule(package)
        self.action = ActionModule(package)
        self.learning = LearningModule(package)
        self.goals = GoalsModule(package)

        # Agent state
        self.state = State()
        self.alive = True
        self.total_steps = 0

        log.info("kernel_ready", agent=package.name)

    def step(self, user_input: str | dict | None = None) -> dict:
        """Execute one full iteration of the agent loop.

        This follows the loop defined in goals/goal_loop.json:
        perceive -> remember -> understand -> select_goal -> plan -> execute -> learn

        Returns a dict summarizing what happened.
        """
        self.total_steps += 1
        step_log = {
            "step": self.total_steps,
            "events": [],
        }

        log.info("step_begin", step=self.total_steps)

        # 1. Perceive
        percepts = self.perception.observe(user_input)
        step_log["events"].append({"phase": "perceive", "percepts": len(percepts)})

        # 2. Remember (integrate percepts)
        self.state = self.memory.update(self.state, percepts)
        step_log["events"].append({"phase": "remember", "step_count": self.state.step_count})

        # 3. Understand (if text input)
        semantic_frame = None
        if user_input and isinstance(user_input, str):
            semantic_frame = self.language.understand(user_input)
            step_log["events"].append({"phase": "understand", "intent": semantic_frame.get("intent")})

        # 4. Select goal (if needed)
        if self.state.current_goal is None or self.state.current_goal.status != "active":
            goal = self.goals.select(self.state, semantic_frame)
            if goal:
                self.state.current_goal = goal
                step_log["events"].append({"phase": "select_goal", "goal": goal.description})
            else:
                step_log["events"].append({"phase": "select_goal", "goal": None})
                log.info("step_end_no_goal", step=self.total_steps)
                return step_log

        # 5. Plan
        if self.state.current_plan is None or self.state.current_plan.done:
            plan = self.reasoning.plan(self.state, self.state.current_goal)
            self.state.current_plan = plan
            step_log["events"].append({"phase": "plan", "steps": len(plan.steps)})

        # 6. Execute plan steps
        plan = self.state.current_plan
        actions_taken = 0

        while not plan.done:
            current_step = plan.current_step
            if current_step is None:
                break

            # 6a. Decide action
            action = self.reasoning.decide(self.state, current_step)
            step_log["events"].append({
                "phase": "decide",
                "tool": action.tool,
                "explanation": action.explanation,
            })

            # 6b. Check alignment
            alignment = self.goals.check_alignment(action)
            if not alignment.approved:
                step_log["events"].append({
                    "phase": "alignment_rejected",
                    "violations": alignment.violated_constraints,
                })
                log.warning(
                    "action_blocked",
                    violations=alignment.violated_constraints,
                )
                # Check escalation
                escalation = self.goals.check_escalation(self.state, action)
                if escalation.should_escalate:
                    step_log["events"].append({
                        "phase": "escalation",
                        "trigger": escalation.trigger_id,
                    })
                break

            # 6c. Execute action
            observation = self.action.execute(action)
            step_log["events"].append({
                "phase": "act",
                "tool": action.tool,
                "success": observation.success,
                "result_preview": str(observation.result)[:200],
            })

            # 6d. Observe result (update memory)
            self.state = self.memory.update(self.state, observation)

            # 6e. Compute reward
            reward = self.action.compute_reward(self.state, action, observation)
            step_log["events"].append({
                "phase": "reward",
                "total": reward.total,
                "components": reward.components,
            })

            # 6f. Learn
            self.learning.update(self.state, action, observation, reward)

            # 6g. Generate response if needed
            response = self.language.generate(self.state, {"result": observation.result})
            step_log["events"].append({"phase": "communicate", "response": response})

            # 6h. Advance plan
            current_step.completed = True
            plan.current_step_index += 1
            actions_taken += 1

            # 6i. Replan check
            if self.reasoning.should_replan(self.state, plan):
                step_log["events"].append({"phase": "replan_triggered"})
                self.state.current_plan = None
                break

        # Check if goal is complete
        if plan.done:
            self.state.current_goal.status = "completed"
            self.state.current_plan = None
            step_log["events"].append({"phase": "goal_completed", "goal": self.state.current_goal.description})
            self.state.current_goal = None

        log.info("step_end", step=self.total_steps, actions=actions_taken)
        return step_log

    def run_interactive(self, max_steps: int = 100) -> None:
        """Run the agent in interactive mode, accepting user input from stdin."""
        print(f"\n{'='*60}")
        print(f"  AGI Runtime Kernel v0.1.0")
        print(f"  Agent: {self.package.name} v{self.package.version}")
        print(f"  Format: .agi v{self.package.format_version}")
        print(f"{'='*60}")
        print(f"\n  Modules loaded: perception, reasoning, memory,")
        print(f"                  language, action, learning, goals")
        print(f"\n  Type a message to interact. Type 'quit' to exit.")
        print(f"  Type 'status' to see agent state.")
        print(f"  Type 'log' to see action log.")
        print(f"{'='*60}\n")

        step = 0
        while self.alive and step < max_steps:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nShutting down.")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Shutting down.")
                break
            if user_input.lower() == "status":
                self._print_status()
                continue
            if user_input.lower() == "log":
                self._print_action_log()
                continue

            result = self.step(user_input)
            self._print_step_result(result)
            step += 1

    def _print_status(self) -> None:
        """Print current agent state."""
        print(f"\n--- Agent Status ---")
        print(f"  Steps completed: {self.total_steps}")
        print(f"  Current goal: {self.state.current_goal.description if self.state.current_goal else 'None'}")
        print(f"  Current plan: {f'{self.state.current_plan.current_step_index}/{len(self.state.current_plan.steps)} steps' if self.state.current_plan else 'None'}")
        print(f"  Episodic memories: {len(self.memory.episodes)}")
        print(f"  Learning updates: {self.learning.update_count}")
        avg = self.learning.total_reward / self.learning.update_count if self.learning.update_count > 0 else 0
        print(f"  Avg reward: {avg:.3f}")
        print(f"--------------------\n")

    def _print_action_log(self) -> None:
        """Print action history."""
        if not self.action.action_log:
            print("\n  No actions taken yet.\n")
            return
        print(f"\n--- Action Log ({len(self.action.action_log)} actions) ---")
        for i, entry in enumerate(self.action.action_log[-10:], 1):
            print(f"  {i}. [{entry['tool']}] {entry['explanation']}")
        print(f"--------------------\n")

    def _print_step_result(self, result: dict) -> None:
        """Print a summary of what happened in this step."""
        print()
        for event in result["events"]:
            phase = event["phase"]
            if phase == "perceive":
                print(f"  [perceive] {event['percepts']} percept(s) received")
            elif phase == "select_goal":
                if event["goal"]:
                    print(f"  [goal] {event['goal']}")
                else:
                    print(f"  [goal] No goal selected")
            elif phase == "plan":
                print(f"  [plan] Created {event['steps']}-step plan")
            elif phase == "act":
                status = "ok" if event["success"] else "FAILED"
                print(f"  [act] {event['tool']} -> {status}")
                if event.get("result_preview"):
                    print(f"         {event['result_preview']}")
            elif phase == "reward":
                print(f"  [learn] reward={event['total']:.2f}")
            elif phase == "communicate":
                print(f"  [response] {event['response']}")
            elif phase == "goal_completed":
                print(f"  [done] Goal completed: {event['goal']}")
            elif phase == "alignment_rejected":
                print(f"  [BLOCKED] Alignment violation: {event['violations']}")
            elif phase == "escalation":
                print(f"  [ESCALATE] Trigger: {event['trigger']}")
        print()
